"""
ETL script for building apartment jeonse (deposit lease) prediction dataset.
Rebuilds merged_dataset.csv from raw Excel files.
"""

import os
import unicodedata
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_ROW_DIR = os.path.join(PROJECT_ROOT, "data", "row")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_PATH = os.path.join(DATASET_DIR, "merged_dataset.csv")

YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
EXCEL_HEADER_ROW = 12
MATCH_MAX_DAYS = 365


# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
def _data_dir():
    if os.path.isdir(DATA_RAW_DIR):
        return DATA_RAW_DIR
    if os.path.isdir(DATA_ROW_DIR):
        return DATA_ROW_DIR
    return DATA_RAW_DIR


def _normalize_filename(name):
    return unicodedata.normalize("NFC", name)


# ---------------------------------------------------------------------------
# Load Excel
# ---------------------------------------------------------------------------
def load_sale_for_year(data_dir, year):
    base = f"아파트(매매)_실거래가_{year}.xlsx"
    for f in os.listdir(data_dir):
        if f.endswith(".xlsx") and _normalize_filename(f) == _normalize_filename(base):
            path = os.path.join(data_dir, f)
            return pd.read_excel(path, header=EXCEL_HEADER_ROW)
    raise FileNotFoundError(base)


def load_jeonse_for_year(data_dir, year):
    base = f"아파트(전세)_실거래가_{year}.xlsx"
    for f in os.listdir(data_dir):
        if f.endswith(".xlsx") and _normalize_filename(f) == _normalize_filename(base):
            path = os.path.join(data_dir, f)
            df = pd.read_excel(path, header=EXCEL_HEADER_ROW)

            if "전월세구분" in df.columns:
                df = df[df["전월세구분"].astype(str).str.strip() == "전세"]

            return df

    raise FileNotFoundError(base)


# ---------------------------------------------------------------------------
# apt_id 생성
# ---------------------------------------------------------------------------
def add_apt_id(df):
    out = df.copy()

    out["apt_id"] = (
        out["시군구"].astype(str).str.replace(" ", "", regex=False)
        + "_"
        + pd.to_numeric(out["본번"], errors="coerce").fillna(0).astype(int).astype(str)
        + "_"
        + pd.to_numeric(out["부번"], errors="coerce").fillna(0).astype(int).astype(str)
    )

    return out


# ---------------------------------------------------------------------------
# Column rename
# ---------------------------------------------------------------------------
SALE_RENAME = {
    "전용면적(㎡)": "area",
    "층": "floor",
    "건축년도": "constructionYear",
    "거래금액(만원)": "salePrice",
}

JEONSE_RENAME = {
    "전용면적(㎡)": "area",
    "층": "floor",
    "건축년도": "constructionYear",
    "보증금(만원)": "jeonsePrice",
}


def to_numeric(s):
    # pandas 3는 엑셀 문자열을 object가 아니라 str dtype으로 읽는다.
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False)

    return pd.to_numeric(s, errors="coerce")


def parse_contract_date(df):
    ym = pd.to_numeric(df["계약년월"], errors="coerce")
    day = pd.to_numeric(df["계약일"], errors="coerce").fillna(1).clip(lower=1, upper=31)
    return pd.to_datetime(
        {
            "year": (ym // 100).astype("Int64"),
            "month": (ym % 100).astype("Int64"),
            "day": day.astype("Int64"),
        },
        errors="coerce",
    )


def normalize_sale(df):
    df = df.rename(columns=SALE_RENAME)

    df["salePrice"] = to_numeric(df["salePrice"])
    df["area"] = to_numeric(df["area"])
    df["floor"] = to_numeric(df["floor"])
    df["constructionYear"] = to_numeric(df["constructionYear"])
    df["saleDate"] = parse_contract_date(df)
    df["saleYear"] = df["saleDate"].dt.year

    return df


def normalize_jeonse(df):
    df = df.rename(columns=JEONSE_RENAME)

    df["jeonsePrice"] = to_numeric(df["jeonsePrice"])
    df["area"] = to_numeric(df["area"])
    df["floor"] = to_numeric(df["floor"])
    df["constructionYear"] = to_numeric(df["constructionYear"])
    df["jeonseDate"] = parse_contract_date(df)
    df["year"] = df["jeonseDate"].dt.year
    df["jeonseYm"] = df["jeonseDate"].dt.year * 100 + df["jeonseDate"].dt.month

    return df


# ---------------------------------------------------------------------------
# Merge: 전세일 이전 365일 이내 매매 중 가장 가까운 건
# ---------------------------------------------------------------------------
def merge_sale_jeonse_asof(all_sale, all_jeonse):
    sale = all_sale.dropna(subset=["saleDate", "salePrice", "apt_id", "area_group"]).copy()
    jeonse = all_jeonse.dropna(subset=["jeonseDate", "jeonsePrice", "apt_id", "area_group"]).copy()

    sale = sale.sort_values(["apt_id", "area_group", "saleDate", "salePrice"])
    sale = sale.drop_duplicates(subset=["apt_id", "area_group", "saleDate"], keep="last")

    jeonse = jeonse.sort_values(["apt_id", "area_group", "jeonseDate", "jeonsePrice"])
    jeonse = jeonse.drop_duplicates(subset=["apt_id", "area_group", "jeonseDate"], keep="last")

    sale_cols = ["apt_id", "area_group", "saleDate", "salePrice", "saleYear"]
    # merge_asof는 on 키가 전역 정렬이어야 한다.
    jeonse = jeonse.sort_values(["jeonseDate", "apt_id", "area_group"])
    sale_match = sale[sale_cols].sort_values(["saleDate", "apt_id", "area_group"])

    merged = pd.merge_asof(
        jeonse,
        sale_match,
        left_on="jeonseDate",
        right_on="saleDate",
        by=["apt_id", "area_group"],
        direction="backward",
        tolerance=pd.Timedelta(days=MATCH_MAX_DAYS),
    )
    merged = merged.dropna(subset=["salePrice", "saleDate"])
    return merged


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_features(df):
    df = df.copy()

    df["price_per_m2"] = df["salePrice"] / df["area"]
    df["jeonseRatio"] = df["jeonsePrice"] / df["salePrice"]
    df["match_gap_days"] = (df["jeonseDate"] - df["saleDate"]).dt.days
    df["match_gap_year"] = df["saleYear"] - df["year"]
    df["buildingAge"] = df["year"] - df["constructionYear"]

    return df


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
REGIONS = ["분당"]

FINAL_COLUMNS = [
    "apt_id",
    "region",
    "apartmentName",
    "dong",
    "area",
    "area_group",
    "floor",
    "constructionYear",
    "buildingAge",
    "salePrice",
    "saleYear",
    "saleDate",
    "jeonsePrice",
    "jeonseDate",
    "jeonseYm",
    "price_per_m2",
    "jeonseRatio",
    "match_gap_days",
    "match_gap_year",
    "year",
    "last_jeonse_ratio",
    "price_percentile_in_dong",
    "last_3_mean_jeonse_ratio",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():

    base_dir = _data_dir()

    all_merged = []

    for region in REGIONS:

        region_dir = os.path.join(base_dir, region)

        if not os.path.isdir(region_dir):
            region_dir = base_dir

        all_sale_list = []
        all_jeonse_list = []

        for year in YEARS:

            try:
                sale_df = load_sale_for_year(region_dir, year)
                jeonse_df = load_jeonse_for_year(region_dir, year)

            except FileNotFoundError:
                continue

            sale_df = add_apt_id(sale_df)
            jeonse_df = add_apt_id(jeonse_df)

            sale_df = normalize_sale(sale_df)
            jeonse_df = normalize_jeonse(jeonse_df)

            sale_df["area_group"] = sale_df["area"].round()
            jeonse_df["area_group"] = jeonse_df["area"].round()

            all_sale_list.append(sale_df)
            all_jeonse_list.append(jeonse_df)

        if not all_sale_list:
            continue

        all_sale = pd.concat(all_sale_list, ignore_index=True)
        all_jeonse = pd.concat(all_jeonse_list, ignore_index=True)

        combined = merge_sale_jeonse_asof(all_sale, all_jeonse)

        combined["region"] = region
        combined["dong"] = combined["시군구"].astype(str).str.split().str[-1]
        combined["apartmentName"] = combined["단지명"]

        combined = add_features(combined)

        # 같은 동·같은 계약년월 안에서만 평당가 순위 (하반기가 상반기 검증에 유입되지 않게)
        combined["price_percentile_in_dong"] = combined.groupby(
            ["dong", "jeonseYm"]
        )["price_per_m2"].rank(pct=True)

        combined = combined.sort_values(["apt_id", "area_group", "jeonseDate"])

        combined["last_jeonse_ratio"] = (
            combined.groupby(["apt_id", "area_group"])["jeonseRatio"].shift(1)
        )

        combined["last_3_mean_jeonse_ratio"] = (
            combined.groupby(["apt_id", "area_group"])["jeonseRatio"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        combined["last_3_mean_jeonse_ratio"] = combined["last_3_mean_jeonse_ratio"].fillna(
            combined["last_jeonse_ratio"]
        )

        combined = combined.dropna(subset=["last_jeonse_ratio"])
        combined = combined[[c for c in FINAL_COLUMNS if c in combined.columns]]
        combined = combined.dropna(subset=["salePrice", "jeonsePrice", "jeonseDate", "saleDate"])

        combined = combined[
            (combined["jeonseRatio"] >= 0.1) &
            (combined["jeonseRatio"] <= 0.95)
        ]

        all_merged.append(combined)

    combined = pd.concat(all_merged, ignore_index=True)

    os.makedirs(DATASET_DIR, exist_ok=True)

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Dataset rows:", len(combined))
    if "jeonseYm" in combined.columns:
        print(combined["jeonseYm"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
