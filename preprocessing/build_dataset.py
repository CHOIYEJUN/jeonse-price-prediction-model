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

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
EXCEL_HEADER_ROW = 12


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
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False)

    return pd.to_numeric(s, errors="coerce")


def normalize_sale(df, year):
    df = df.rename(columns=SALE_RENAME)

    df["salePrice"] = to_numeric(df["salePrice"])
    df["area"] = to_numeric(df["area"])
    df["floor"] = to_numeric(df["floor"])
    df["constructionYear"] = to_numeric(df["constructionYear"])

    df["saleYear"] = year

    return df


def normalize_jeonse(df, year):
    df = df.rename(columns=JEONSE_RENAME)

    df["jeonsePrice"] = to_numeric(df["jeonsePrice"])
    df["area"] = to_numeric(df["area"])
    df["floor"] = to_numeric(df["floor"])
    df["constructionYear"] = to_numeric(df["constructionYear"])

    df["year"] = year

    return df


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------
SALE_YEAR_TOLERANCE = 1


def merge_sale_jeonse_expanded(all_sale, all_jeonse):

    sale_agg = all_sale.drop_duplicates(
        subset=["apt_id", "area_group", "saleYear"]
    )

    jeonse_agg = all_jeonse.drop_duplicates(
        subset=["apt_id", "area_group", "year"]
    )

    merged = pd.merge(
        jeonse_agg,
        sale_agg,
        on=["apt_id", "area_group"],
        how="inner",
        suffixes=("", "_s"),
    )

    merged = merged[
        (merged["saleYear"] - merged["year"]).abs() <= SALE_YEAR_TOLERANCE
    ]

    merged = merged[[c for c in merged.columns if not c.endswith("_s")]]

    return merged


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_features(df):

    df = df.copy()

    df["price_per_m2"] = df["salePrice"] / df["area"]

    df["jeonseRatio"] = df["jeonsePrice"] / df["salePrice"]

    df["match_gap_year"] = df["saleYear"] - df["year"]

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
    "jeonsePrice",
    "price_per_m2",
    "jeonseRatio",
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

            sale_df = normalize_sale(sale_df, year)
            jeonse_df = normalize_jeonse(jeonse_df, year)

            sale_df["buildingAge"] = year - sale_df["constructionYear"]
            jeonse_df["buildingAge"] = year - jeonse_df["constructionYear"]

            sale_df["area_group"] = sale_df["area"].round()
            jeonse_df["area_group"] = jeonse_df["area"].round()

            all_sale_list.append(sale_df)
            all_jeonse_list.append(jeonse_df)

        if not all_sale_list:
            continue

        all_sale = pd.concat(all_sale_list)
        all_jeonse = pd.concat(all_jeonse_list)

        combined = merge_sale_jeonse_expanded(all_sale, all_jeonse)

        combined["region"] = region

        combined["dong"] = combined["시군구"].astype(str).str.split().str[-1]

        combined["apartmentName"] = combined["단지명"]

        combined = add_features(combined)

        # Submarket: 동·연도 내 평당가 상위 몇 % (고급 단지 → 낮은 전세가율 패턴 학습)
        combined["price_percentile_in_dong"] = combined.groupby(["dong", "year"])["price_per_m2"].rank(pct=True)

        combined = combined.sort_values(["apt_id", "area_group", "year"])

        combined["last_jeonse_ratio"] = (
            combined.groupby(["apt_id", "area_group"])["jeonseRatio"]
            .shift(1)
        )

        # Rolling: 같은 단지·면적대 직전 3건 평균 전세가율 (moving average)
        combined["last_3_mean_jeonse_ratio"] = (
            combined.groupby(["apt_id", "area_group"])["jeonseRatio"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        combined["last_3_mean_jeonse_ratio"] = combined["last_3_mean_jeonse_ratio"].fillna(combined["last_jeonse_ratio"])

        combined = combined.dropna(subset=["last_jeonse_ratio"])

        combined = combined[[c for c in FINAL_COLUMNS if c in combined.columns]]

        combined = combined.dropna(subset=["salePrice", "jeonsePrice"])

        combined = combined[
            (combined["jeonseRatio"] >= 0.1) &
            (combined["jeonseRatio"] <= 0.95)
        ]

        all_merged.append(combined)

    combined = pd.concat(all_merged)

    os.makedirs(DATASET_DIR, exist_ok=True)

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Dataset rows:", len(combined))


if __name__ == "__main__":
    main()