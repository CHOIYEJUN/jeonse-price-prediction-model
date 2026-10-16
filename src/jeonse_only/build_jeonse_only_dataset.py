"""전세 실거래만으로 다음 보증금 예측용 표를 만든다.

매매가·전세가율은 만들지 않는다.
같은 단지(apt_id)와 면적대(area_group)의 거래를 시간순으로 놓고,
각 행에 그보다 앞선 거래만으로 계산한 자기회귀 피처를 붙인다.

첫 거래는 직전 값이 없어 정답 행이 될 수 없다.
다만 그 다음 거래의 과거가 되므로 표에서는 지우지 않는다.
학습 스크립트가 정답 행만 가려 쓴다.
"""

import os
import sys

import pandas as pd

# 엑셀 로더는 src/jeonse_ratio/build_dataset.py 에 있다. 매매 결합 함수는 쓰지 않는다.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RATIO_DIR = os.path.join(_SRC_DIR, "jeonse_ratio")
sys.path.insert(0, _RATIO_DIR)

from build_dataset import (
    DATASET_DIR,
    REGIONS,
    YEARS,
    _data_dir,
    add_apt_id,
    load_jeonse_for_year,
    normalize_jeonse,
)


OUTPUT_PATH = os.path.join(DATASET_DIR, "jeonse_only_dataset.csv")

# 시퀀스를 다시 만들 때 필요한 키(apt_id, area_group)를 포함한다.
FINAL_COLUMNS = [
    "apt_id",
    "region",
    "apartmentName",
    "dong",
    "area",
    "area_group",
    "floor",
    "buildingAge",
    "jeonseDate",
    "jeonseYm",
    "jeonsePrice",
    "last_jeonse_price",
    "last_3_mean_jeonse_price",
    "days_since_last",
    "last_price_per_m2",
]


def load_region_jeonse(region_dir, region):
    """한 지역 폴더의 연도별 전세 엑셀을 이어 붙인다.

    region_dir: data/row/분당 같은 폴더
    region: 표에 남길 지역 이름
    반환: 정리된 전세 표. 파일이 없으면 None
    """
    frames = []
    for year in YEARS:
        try:
            raw = load_jeonse_for_year(region_dir, year)
        except FileNotFoundError:
            continue
        raw = add_apt_id(raw)
        raw = normalize_jeonse(raw)
        raw["area_group"] = raw["area"].round()
        raw["region"] = region
        frames.append(raw)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def clean_jeonse(df):
    """금액·면적·날짜가 없는 행을 버리고, 같은 날 중복을 하나 남긴다.

    df: load_region_jeonse 결과
    반환: 시간순으로 정렬된 표
    """
    df = df.copy()
    df["apartmentName"] = df["단지명"].astype(str).str.strip()
    df["dong"] = df["시군구"].astype(str).str.split().str[-1].str.strip()
    df["buildingAge"] = df["year"] - df["constructionYear"]

    needed = [
        "jeonseDate",
        "jeonsePrice",
        "area",
        "floor",
        "buildingAge",
        "apt_id",
        "area_group",
        "apartmentName",
        "dong",
        "jeonseYm",
    ]
    df = df.dropna(subset=needed)
    df = df[(df["jeonsePrice"] > 0) & (df["area"] > 0)]

    df = df.sort_values(["apt_id", "area_group", "jeonseDate", "jeonsePrice"])
    # 같은 단지·면적·날짜는 한 건만 남긴다. 시퀀스에 같은 날이 두 번 들어가지 않게.
    df = df.drop_duplicates(subset=["apt_id", "area_group", "jeonseDate"], keep="last")
    return df.reset_index(drop=True)


def add_autoregressive_features(df):
    """직전 보증금·직전 3건 평균·경과 일수·직전 평당가를 붙인다.

    전부 shift(1)이라 같은 행의 이번 보증금은 들어가지 않는다.
    df: clean_jeonse 결과. apt_id, area_group, jeonseDate 순으로 정렬되어 있어야 한다.
    반환: 피처가 추가된 복사본
    """
    df = df.sort_values(["apt_id", "area_group", "jeonseDate"]).copy()
    # 이번 거래의 평당가. 저장하지 않고, 한 칸 밀어 직전 값만 남긴다.
    df["price_per_m2"] = df["jeonsePrice"] / df["area"]
    group = df.groupby(["apt_id", "area_group"], sort=False)

    df["last_jeonse_price"] = group["jeonsePrice"].shift(1)
    df["last_3_mean_jeonse_price"] = group["jeonsePrice"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    previous_date = group["jeonseDate"].shift(1)
    df["days_since_last"] = (df["jeonseDate"] - previous_date).dt.days
    df["last_price_per_m2"] = group["price_per_m2"].shift(1)
    df = df.drop(columns=["price_per_m2"])

    return df


def main():
    """지역별 전세 엑셀을 모아 자기회귀 피처를 붙이고 CSV로 저장한다."""
    base_dir = _data_dir()
    parts = []

    for region in REGIONS:
        region_dir = os.path.join(base_dir, region)
        if not os.path.isdir(region_dir):
            print(f"건너뜀 (폴더 없음): {region}")
            continue

        loaded = load_region_jeonse(region_dir, region)
        if loaded is None:
            print(f"건너뜀 (전세 엑셀 없음): {region}")
            continue

        cleaned = clean_jeonse(loaded)
        featured = add_autoregressive_features(cleaned)
        parts.append(featured)
        with_history = int(featured["last_jeonse_price"].notna().sum())
        print(f"{region}: 전세 {len(featured)}건, 직전 거래 있음 {with_history}건")

    if not parts:
        raise RuntimeError("전세 엑셀을 찾지 못했습니다.")

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(
        ["apt_id", "area_group", "jeonseDate"]
    ).reset_index(drop=True)

    negative_gap = combined["days_since_last"].dropna() < 0
    if bool(negative_gap.any()):
        raise RuntimeError("직전 거래보다 날짜가 앞선 행이 있습니다. 정렬을 확인하세요.")

    combined = combined[[c for c in FINAL_COLUMNS if c in combined.columns]]

    os.makedirs(DATASET_DIR, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    supervised = combined["last_jeonse_price"].notna().sum()
    print("Dataset rows:", len(combined))
    print("Rows with history (학습 후보):", int(supervised))
    print("저장:", OUTPUT_PATH)
    if "jeonseYm" in combined.columns:
        print(combined["jeonseYm"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
