"""
ETL script for building apartment jeonse (deposit lease) prediction dataset.
Rebuilds merged_dataset.csv from raw Excel files in data/raw/ (or data/row/).
"""

import os
import unicodedata
import pandas as pd


# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_ROW_DIR = os.path.join(PROJECT_ROOT, "data", "row")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_PATH = os.path.join(DATASET_DIR, "merged_dataset.csv")

YEARS = [2020,2021,2022, 2023, 2024, 2025]
EXCEL_HEADER_ROW = 12


def _data_dir() -> str:
    """Return data directory: data/raw if present, else data/row (e.g. macOS typo)."""
    if os.path.isdir(DATA_RAW_DIR):
        return DATA_RAW_DIR
    if os.path.isdir(DATA_ROW_DIR):
        return DATA_ROW_DIR
    return DATA_RAW_DIR


def _normalize_filename(name: str) -> str:
    """NFC normalize for Korean filename matching on macOS (NFD)."""
    return unicodedata.normalize("NFC", name)


# ---------------------------------------------------------------------------
# STEP 1 — Load Excel files (per year: 매매 + 전세)
# ---------------------------------------------------------------------------
def load_sale_for_year(data_dir: str, year: int) -> pd.DataFrame:
    """Load 아파트(매매)_실거래가_YEAR.xlsx."""
    base = f"아파트(매매)_실거래가_{year}.xlsx"
    for f in os.listdir(data_dir):
        if f.endswith(".xlsx") and _normalize_filename(f) == _normalize_filename(base):
            path = os.path.join(data_dir, f)
            return pd.read_excel(path, header=EXCEL_HEADER_ROW)
    raise FileNotFoundError(f"Sale file not found for year {year}: {base}")


def load_jeonse_for_year(data_dir: str, year: int) -> pd.DataFrame:
    """Load 아파트(전세)_실거래가_YEAR.xlsx. Keeps only rows where 전월세구분 == '전세' (excludes 월세)."""
    base = f"아파트(전세)_실거래가_{year}.xlsx"
    for f in os.listdir(data_dir):
        if f.endswith(".xlsx") and _normalize_filename(f) == _normalize_filename(base):
            path = os.path.join(data_dir, f)
            df = pd.read_excel(path, header=EXCEL_HEADER_ROW)
            if "전월세구분" in df.columns:
                df = df[df["전월세구분"].astype(str).str.strip() == "전세"]
            return df
    raise FileNotFoundError(f"Jeonse file not found for year {year}: {base}")


# ---------------------------------------------------------------------------
# STEP 2 — Create apartment ID: 시군구(공백제거) + "_" + 본번 + "_" + 부번
# Example: 경기도 성남시 분당구 정자동, 본번 0121, 부번 0000 → 경기도성남시분당구정자동_121_0
# ---------------------------------------------------------------------------
def add_apt_id(df: pd.DataFrame) -> pd.DataFrame:
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
# STEP 3 — Normalize numeric fields & rename columns
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


def to_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == object or s.dtype.name == "string":
        s = s.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def normalize_sale(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.rename(columns=SALE_RENAME)
    df["salePrice"] = to_numeric(df["salePrice"])
    df["area"] = to_numeric(df["area"])
    df["floor"] = to_numeric(df["floor"])
    df["constructionYear"] = to_numeric(df["constructionYear"])
    df["saleYear"] = year
    return df


def normalize_jeonse(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.rename(columns=JEONSE_RENAME)
    df["jeonsePrice"] = to_numeric(df["jeonsePrice"])
    df["area"] = to_numeric(df["area"])
    df["floor"] = to_numeric(df["floor"])
    df["constructionYear"] = to_numeric(df["constructionYear"])
    df["year"] = year
    return df


# ---------------------------------------------------------------------------
# STEP 4 — Additional features: buildingAge, area_group (in main loop)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# STEP 5 — Merge sale and jeonse: allow sale year ±1 from jeonse year (data expansion)
# ---------------------------------------------------------------------------
SALE_YEAR_TOLERANCE = 1  # 전세 연도 기준 매매 데이터를 ±1년까지 매칭


def merge_sale_jeonse_expanded(
    all_sale: pd.DataFrame, all_jeonse: pd.DataFrame
) -> pd.DataFrame:
    """Merge jeonse with sale where sale year is in [jeonse_year - 1, jeonse_year + 1]."""
    # Sale: one row per (apt_id, area_group, saleYear)
    sale_agg = all_sale.drop_duplicates(
        subset=["apt_id", "area_group", "saleYear"], keep="first"
    )
    # Jeonse: one row per (apt_id, area_group, year)
    jeonse_agg = all_jeonse.drop_duplicates(
        subset=["apt_id", "area_group", "year"], keep="first"
    )
    # Merge on (apt_id, area_group) only → then filter by year range
    merged = pd.merge(
        jeonse_agg,
        sale_agg,
        on=["apt_id", "area_group"],
        how="inner",
        suffixes=("", "_s"),
    )
    # Keep only rows where sale year is within ±1 of jeonse year
    merged = merged[(merged["saleYear"] - merged["year"]).abs() <= SALE_YEAR_TOLERANCE]
    # Drop sale duplicate columns (area_s, floor_s, constructionYear_s, buildingAge_s)
    merged = merged[[c for c in merged.columns if not c.endswith("_s")]]
    return merged


# ---------------------------------------------------------------------------
# STEP 6 — Feature engineering: price_per_m2, jeonseRatio
# ---------------------------------------------------------------------------
def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_per_m2"] = df["salePrice"] / df["area"]
    df["jeonseRatio"] = df["jeonsePrice"] / df["salePrice"]
    return df


# ---------------------------------------------------------------------------
# STEP 7 — Final columns; dong = last token of 시군구; saleYear = 매매 계약 연도
# ---------------------------------------------------------------------------
FINAL_COLUMNS = [
    "apt_id",
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
    "year",
]


def main() -> None:
    data_dir = _data_dir()
    print(f"Data directory: {data_dir}")
    print(f"Merge: jeonse × sale on (apt_id, area_group), sale year ±{SALE_YEAR_TOLERANCE} from jeonse year")

    all_sale_list = []
    all_jeonse_list = []

    for year in YEARS:
        # STEP 1 — Load
        sale_df = load_sale_for_year(data_dir, year)
        jeonse_df = load_jeonse_for_year(data_dir, year)

        # STEP 2 — apt_id
        sale_df = add_apt_id(sale_df)
        jeonse_df = add_apt_id(jeonse_df)

        # STEP 3 — Rename & numeric
        sale_df = normalize_sale(sale_df, year)
        jeonse_df = normalize_jeonse(jeonse_df, year)

        # STEP 4 — buildingAge, area_group
        sale_df["buildingAge"] = year - sale_df["constructionYear"]
        sale_df["area_group"] = sale_df["area"].round().astype("Int64")
        jeonse_df["buildingAge"] = year - jeonse_df["constructionYear"]
        jeonse_df["area_group"] = jeonse_df["area"].round().astype("Int64")

        all_sale_list.append(sale_df)
        all_jeonse_list.append(jeonse_df)

    # STEP 5 — Single merge: jeonse × sale on (apt_id, area_group), sale year ±1 from jeonse year
    all_sale = pd.concat(all_sale_list, ignore_index=True)
    all_jeonse = pd.concat(all_jeonse_list, ignore_index=True)
    combined = merge_sale_jeonse_expanded(all_sale, all_jeonse)

    # STEP 6 — Ratios
    combined = add_ratios(combined)

    # STEP 7 — dong (last part of 시군구), apartmentName (단지명), final columns
    combined["dong"] = combined["시군구"].astype(str).str.split().str[-1]
    combined["apartmentName"] = (
        combined["단지명"] if "단지명" in combined.columns else combined.get("apartmentName", "")
    )
    combined = combined[[c for c in FINAL_COLUMNS if c in combined.columns]]

    # Drop rows with invalid prices and outlier jeonse ratio
    combined = combined.dropna(subset=["salePrice", "jeonsePrice", "area"])
    combined = combined[(combined["salePrice"] > 0) & (combined["area"] > 0)]
    combined = combined[
        (combined["jeonseRatio"] >= 0.1) & (combined["jeonseRatio"] <= 0.95)
    ]

    # STEP 9 — Save
    os.makedirs(DATASET_DIR, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved to: {OUTPUT_PATH}")

    # STEP 10 — Statistics
    n_rows = len(combined)
    n_apt = combined["apt_id"].nunique()
    year_dist = combined["year"].value_counts().sort_index()

    print()
    print("Dataset rows:", n_rows)
    print("Unique apartments:", n_apt)
    print("Year distribution:")
    for y, c in year_dist.items():
        print(f"  {y}: {c}")


if __name__ == "__main__":
    main()
