"""국토부 아파트 실거래 엑셀을 읽어 학습용 merged_dataset.csv를 만든다.

흐름
    1. 연도별 매매·전세 엑셀을 읽는다 (전월세 중 전세만).
    2. 단지(apt_id)와 면적대(area_group)가 같은 거래를 모은다.
    3. 각 전세에, 그 날짜 이전 365일 안 매매 중 가장 가까운 한 건을 붙인다.
    4. 전세가율, 시차, 직전 전세가율 등을 만들고 CSV로 저장한다.
"""

import os
import unicodedata
import pandas as pd


# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
# __file__ = 이 스크립트 위치. dirname 두 번이면 프로젝트 루트
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_ROW_DIR = os.path.join(PROJECT_ROOT, "data", "row")  # 예전 폴더 이름 호환
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_PATH = os.path.join(DATASET_DIR, "merged_dataset.csv")

YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]  # 읽을 연도 목록
EXCEL_HEADER_ROW = 12  # 실거래 엑셀에서 열 이름이 있는 행 (0부터 세면 12)
MATCH_MAX_DAYS = 365  # 전세보다 앞선 매매를 찾을 때 허용하는 최대 일수


# ---------------------------------------------------------------------------
# 자료 폴더
# ---------------------------------------------------------------------------
def _data_dir():
    """원천 엑셀이 있는 폴더를 고른다.

    data/raw 가 있으면 그것을, 없으면 예전 이름 data/row 를 쓴다.
    둘 다 없어도 raw 경로를 돌려 주고, 이후 파일 읽기에서 오류가 난다.
    """
    if os.path.isdir(DATA_RAW_DIR):
        return DATA_RAW_DIR
    if os.path.isdir(DATA_ROW_DIR):
        return DATA_ROW_DIR
    return DATA_RAW_DIR


def _normalize_filename(name):
    """한글 파일명 정규화(NFC). mac/윈도우에서 같은 글자가 다르게 저장되는 것을 맞춘다.

    name: 파일 이름 문자열
    반환: 유니코드 NFC로 맞춘 문자열
    """
    return unicodedata.normalize("NFC", name)


# ---------------------------------------------------------------------------
# 엑셀 읽기
# ---------------------------------------------------------------------------
def _region_search_dirs(data_dir):
    """지역 폴더와 그 안의 아파트/ 폴더에서 엑셀을 찾는다."""
    dirs = [data_dir, os.path.join(data_dir, "아파트")]
    return [d for d in dirs if os.path.isdir(d)]


def _find_excel(data_dir, base):
    """파일명이 base와 같은 엑셀 경로를 돌려준다."""
    want = _normalize_filename(base)
    for folder in _region_search_dirs(data_dir):
        for f in os.listdir(folder):
            if f.endswith(".xlsx") and _normalize_filename(f) == want:
                return os.path.join(folder, f)
    raise FileNotFoundError(base)


def load_sale_for_year(data_dir, year):
    """해당 연도 매매 엑셀 한 장을 DataFrame으로 읽는다.

    data_dir: 지역 폴더 (예: data/row/분당)
    year: 2020 같은 정수
    반환: 헤더가 잡힌 매매 표
    """
    path = _find_excel(data_dir, f"아파트(매매)_실거래가_{year}.xlsx")
    return pd.read_excel(path, header=EXCEL_HEADER_ROW)


def load_jeonse_for_year(data_dir, year):
    """해당 연도 전월세 엑셀에서 전세 행만 읽는다.

    data_dir, year: load_sale_for_year와 같다
    반환: 전세만 남긴 표
    """
    path = _find_excel(data_dir, f"아파트(전세)_실거래가_{year}.xlsx")
    df = pd.read_excel(path, header=EXCEL_HEADER_ROW)

    # 월세를 빼고 전세만 남긴다
    if "전월세구분" in df.columns:
        df = df[df["전월세구분"].astype(str).str.strip() == "전세"]

    return df


# ---------------------------------------------------------------------------
# 단지 식별자
# ---------------------------------------------------------------------------
def add_apt_id(df):
    """시군구+본번+부번을 이어 단지 키(apt_id)를 만든다.

    df: 원천 엑셀을 읽은 표. 시군구·본번·부번 열이 있어야 한다.
    반환: apt_id 열이 추가된 복사본
    """
    out = df.copy()

    # 공백을 없애 같은 주소를 한 키로 묶는다
    out["apt_id"] = (
        out["시군구"].astype(str).str.replace(" ", "", regex=False)
        + "_"
        + pd.to_numeric(out["본번"], errors="coerce").fillna(0).astype(int).astype(str)
        + "_"
        + pd.to_numeric(out["부번"], errors="coerce").fillna(0).astype(int).astype(str)
    )

    return out


# ---------------------------------------------------------------------------
# 한글 열 이름 → 영문
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
    """쉼표가 있는 금액 문자열을 숫자로 바꾼다.

    s: pandas Series (한 열)
    반환: 숫자 Series. 변환 실패는 NaN (errors="coerce")

    pandas 3는 엑셀 문자열을 object가 아니라 str dtype으로 읽는다.
    """
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False)

    return pd.to_numeric(s, errors="coerce")


def parse_contract_date(df):
    """계약년월(202607)과 계약일(10)을 날짜형으로 합친다.

    df: 계약년월·계약일 열이 있는 표
    반환: datetime64 시리즈. 잘못된 값은 NaT
    """
    ym = pd.to_numeric(df["계약년월"], errors="coerce")  # 예: 202607
    day = pd.to_numeric(df["계약일"], errors="coerce").fillna(1).clip(lower=1, upper=31)
    return pd.to_datetime(
        {
            "year": (ym // 100).astype("Int64"),  # 202607 // 100 = 2026
            "month": (ym % 100).astype("Int64"),  # 202607 % 100 = 7
            "day": day.astype("Int64"),
        },
        errors="coerce",
    )


def normalize_sale(df):
    """매매 표의 열 이름과 숫자·날짜를 학습에 쓰기 쉽게 정리한다.

    df: load_sale_for_year 결과
    반환: salePrice, saleDate, saleYear 등이 있는 표
    """
    df = df.rename(columns=SALE_RENAME)

    df["salePrice"] = to_numeric(df["salePrice"])
    df["area"] = to_numeric(df["area"])
    df["floor"] = to_numeric(df["floor"])
    df["constructionYear"] = to_numeric(df["constructionYear"])
    df["saleDate"] = parse_contract_date(df)
    df["saleYear"] = df["saleDate"].dt.year

    return df


def normalize_jeonse(df):
    """전세 표의 열 이름과 숫자·날짜를 정리한다.

    df: load_jeonse_for_year 결과
    반환: jeonsePrice, jeonseDate, year, jeonseYm 등이 있는 표
    """
    df = df.rename(columns=JEONSE_RENAME)

    df["jeonsePrice"] = to_numeric(df["jeonsePrice"])
    df["area"] = to_numeric(df["area"])
    df["floor"] = to_numeric(df["floor"])
    df["constructionYear"] = to_numeric(df["constructionYear"])
    df["jeonseDate"] = parse_contract_date(df)
    df["year"] = df["jeonseDate"].dt.year
    # jeonseYm: 202607처럼 연·월을 한 정수로. 학습/검증/평가 분할에 쓴다
    df["jeonseYm"] = df["jeonseDate"].dt.year * 100 + df["jeonseDate"].dt.month

    return df


# ---------------------------------------------------------------------------
# 결합: 전세일 이전 365일 이내 매매 중 가장 가까운 건
# ---------------------------------------------------------------------------
def merge_sale_jeonse_asof(all_sale, all_jeonse):
    """전세 한 건에, 같은 단지·면적대에서 그 날짜 이전의 최근 매매를 붙인다.

    all_sale: 여러 해 매매를 이어 붙인 표
    all_jeonse: 여러 해 전세를 이어 붙인 표
    반환: 매매가가 붙은 전세 표. 짝이 없으면 그 행은 버린다.

    merge_asof
        날짜가 가장 가까운 행을 붙이는 조인.
        direction="backward": 전세보다 늦은 매매는 쓰지 않는다 (미래 누수 방지).
    """
    sale = all_sale.dropna(subset=["saleDate", "salePrice", "apt_id", "area_group"]).copy()
    jeonse = all_jeonse.dropna(subset=["jeonseDate", "jeonsePrice", "apt_id", "area_group"]).copy()

    sale = sale.sort_values(["apt_id", "area_group", "saleDate", "salePrice"])
    # 같은 날 같은 단지·면적이면 마지막 금액만 남긴다
    sale = sale.drop_duplicates(subset=["apt_id", "area_group", "saleDate"], keep="last")

    jeonse = jeonse.sort_values(["apt_id", "area_group", "jeonseDate", "jeonsePrice"])
    jeonse = jeonse.drop_duplicates(subset=["apt_id", "area_group", "jeonseDate"], keep="last")

    sale_cols = ["apt_id", "area_group", "saleDate", "salePrice", "saleYear"]
    # merge_asof는 on 키(날짜)가 전역으로 정렬되어 있어야 한다
    jeonse = jeonse.sort_values(["jeonseDate", "apt_id", "area_group"])
    sale_match = sale[sale_cols].sort_values(["saleDate", "apt_id", "area_group"])

    merged = pd.merge_asof(
        jeonse,
        sale_match,
        left_on="jeonseDate",
        right_on="saleDate",
        by=["apt_id", "area_group"],  # 단지·면적대가 같은 후보만
        direction="backward",
        tolerance=pd.Timedelta(days=MATCH_MAX_DAYS),
    )
    merged = merged.dropna(subset=["salePrice", "saleDate"])
    return merged


# ---------------------------------------------------------------------------
# 파생 변수
# ---------------------------------------------------------------------------
def add_features(df):
    """매매가·전세가에서 비율·시차·연식을 계산한다.

    df: 결합이 끝난 표
    반환: 파생 열이 추가된 복사본
    """
    df = df.copy()

    df["price_per_m2"] = df["salePrice"] / df["area"]  # 단위면적당 매매가
    df["jeonseRatio"] = df["jeonsePrice"] / df["salePrice"]  # 종속변수
    df["match_gap_days"] = (df["jeonseDate"] - df["saleDate"]).dt.days  # 짝의 일수 차이
    df["match_gap_year"] = df["saleYear"] - df["year"]  # 연 단위 차이
    df["buildingAge"] = df["year"] - df["constructionYear"]  # 전세 시점 기준 연식

    return df


# ---------------------------------------------------------------------------
# 저장할 열
# ---------------------------------------------------------------------------
# data/row 아래 지역 폴더 이름. 분당은 2020~, 수정구·중원구는 2023~ 파일이 있다.
REGIONS = ["분당", "수정구", "중원구"]

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
# 진입점
# ---------------------------------------------------------------------------
def main():
    """연도별 엑셀을 모아 결합·파생 변수를 만들고 CSV로 저장한다."""

    base_dir = _data_dir()  # 원천 엑셀 폴더

    all_merged = []  # 지역별로 만든 표를 나중에 세로로 이어 붙인다

    for region in REGIONS:

        region_dir = os.path.join(base_dir, region)

        if not os.path.isdir(region_dir):
            region_dir = base_dir  # 하위 폴더가 없으면 바로 아래에서 찾는다

        all_sale_list = []  # 연도별 매매 표
        all_jeonse_list = []

        for year in YEARS:

            try:
                sale_df = load_sale_for_year(region_dir, year)
                jeonse_df = load_jeonse_for_year(region_dir, year)

            except FileNotFoundError:
                continue  # 그 해 파일이 없으면 건너뛴다

            sale_df = add_apt_id(sale_df)
            jeonse_df = add_apt_id(jeonse_df)

            sale_df = normalize_sale(sale_df)
            jeonse_df = normalize_jeonse(jeonse_df)

            # 면적대: 전용면적을 반올림. 같은 단지라도 평수가 다르면 따로 묶는다
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
        # 시군구 마지막 토큰을 동으로 쓴다. 예: "성남시 분당구 정자동" → "정자동"
        combined["dong"] = combined["시군구"].astype(str).str.split().str[-1]
        combined["apartmentName"] = combined["단지명"]

        combined = add_features(combined)

        # 같은 동·같은 계약년월 안에서만 평당가 순위 (다른 달 정보가 섞이지 않게)
        combined["price_percentile_in_dong"] = combined.groupby(
            ["dong", "jeonseYm"]
        )["price_per_m2"].rank(pct=True)

        combined = combined.sort_values(["apt_id", "area_group", "jeonseDate"])

        # shift(1): 바로 이전 행의 전세가율. 미래 값을 쓰지 않는다
        combined["last_jeonse_ratio"] = (
            combined.groupby(["apt_id", "area_group"])["jeonseRatio"].shift(1)
        )

        # 직전 최대 3건 평균. transform은 그룹 결과를 원래 행 수에 맞춰 되돌린다
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

        # 1차 극단값. 학습 단계의 분위수 필터와는 별개
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
        encoding="utf-8-sig"  # 엑셀에서 한글이 깨지지 않게 BOM을 붙인다
    )

    print("Dataset rows:", len(combined))
    if "jeonseYm" in combined.columns:
        print(combined["jeonseYm"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
