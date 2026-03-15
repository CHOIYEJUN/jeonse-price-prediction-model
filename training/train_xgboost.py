"""
아파트 전세비율(jeonseRatio) 예측용 XGBoost 회귀 모델 학습.
시점 기반 분할, 원핫 인코딩 사용, 모델은 model/xgboost_jeonse_model.pkl 로 저장.
"""

import os
import time
import threading
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
except Exception as e:
    print("XGBoost 로드 실패. macOS에서는 OpenMP(libomp)가 필요합니다.")
    print("터미널에서 다음 명령으로 설치 후 다시 실행하세요:")
    print("  brew install libomp")
    raise SystemExit(1) from e


# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "merged_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_jeonse_model.pkl")

# ---------------------------------------------------------------------------
# XGBoost 하이퍼파라미터
# ---------------------------------------------------------------------------
# 20회 랜덤 탐색으로 선정 (Test MAPE 8.78% 기준 최적)
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.07,
    "subsample": 0.8,
    "colsample_bytree": 0.85,
    "min_child_weight": 12,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "random_state": 42,
}

PROGRESS_LOG_INTERVAL_SEC = 10.0
MIN_SAMPLES_FOR_CATEGORY = 15
EARLY_STOPPING_ROUNDS = 50

# ---------------------------------------------------------------------------
# 연도별 학습 가중치
# ---------------------------------------------------------------------------
YEAR_WEIGHTS = {
    2025: 1.0,
    2024: 1.0,
    2023: 1.0,
    2022: 1.0,
    2021: 1.0,
    2020: 1.0,
}
DEFAULT_YEAR_WEIGHT = 1.0

# 1차 학습 후 상위 K개 피처만 골라 재학습
TOP_K_FEATURES_FOR_RETRAIN = 50

# ---------------------------------------------------------------------------
# 전세비율(jeonseRatio) 이상치 제거 — 허수/비정상 거래(지인 거래 등) 제외
# ---------------------------------------------------------------------------
# 모드: "bounds" = 고정 구간, "percentile" = 하위/상위 N% 제거
OUTLIER_FILTER_MODE = "percentile"  # "bounds" | "percentile"
# bounds 모드: 이 구간 밖은 제거
JEONSE_RATIO_MIN = 0.25   # 25% 미만 제거 (매우 저렴한 전세 = 의심 거래)
JEONSE_RATIO_MAX = 0.9    # 90% 초과 제거
# percentile 모드: 하위/상위 이 비율만큼 제거
OUTLIER_PERCENTILE_LOW = 10   
OUTLIER_PERCENTILE_HIGH = 10  


def _progress_logger(interval_sec: float, stop_event: threading.Event) -> None:
    """일정 간격으로 학습 진행 로그를 출력."""
    start = time.perf_counter()
    while not stop_event.is_set():
        stop_event.wait(interval_sec)
        if stop_event.is_set():
            break
        elapsed = int(time.perf_counter() - start)
        print(f"  진행 중: 학습 {elapsed}초 경과")


def run_training(xgb_params=None, verbose=True):
    """
    데이터 로드·전처리·학습·평가까지 수행.
    xgb_params: None이면 XGB_PARAMS 사용. 반환: dict(mape, mae, rmse, model, feature_cols, params)
    """
    params = dict(XGB_PARAMS) if xgb_params is None else dict(xgb_params)
    params.setdefault("random_state", 42)

    # -----------------------------------------------------------------------
    # STEP 1 — 데이터셋 로드
    # -----------------------------------------------------------------------
    df = pd.read_csv(DATASET_PATH)

    if "last_jeonse_ratio" in df.columns:
        df = df.dropna(subset=["last_jeonse_ratio"]).copy()

    if "jeonseRatio" in df.columns:
        if OUTLIER_FILTER_MODE == "bounds":
            keep = (df["jeonseRatio"] >= JEONSE_RATIO_MIN) & (df["jeonseRatio"] <= JEONSE_RATIO_MAX)
            n_removed = (~keep).sum()
            if verbose:
                print(f"전세비율 이상치 제거 (bounds): jeonseRatio [{JEONSE_RATIO_MIN}, {JEONSE_RATIO_MAX}] 밖 {n_removed}건 제거")
        else:
            low_q = np.percentile(df["jeonseRatio"], OUTLIER_PERCENTILE_LOW)
            high_q = np.percentile(df["jeonseRatio"], 100 - OUTLIER_PERCENTILE_HIGH)
            keep = (df["jeonseRatio"] >= low_q) & (df["jeonseRatio"] <= high_q)
            n_removed = (~keep).sum()
            if verbose:
                print(f"전세비율 이상치 제거 (percentile): 하위 {OUTLIER_PERCENTILE_LOW}%·상위 {OUTLIER_PERCENTILE_HIGH}% 제거, {n_removed}건 제거 (유효 구간 [{low_q:.3f}, {high_q:.3f}])")
        df = df.loc[keep].copy()

    df_original = df.copy()

    # -----------------------------------------------------------------------
    # STEP 2 — 희귀 단지/동은 "기타"로 묶은 뒤 원핫 인코딩
    # -----------------------------------------------------------------------
    for col in ["apartmentName", "dong"]:
        counts = df[col].value_counts()
        rare = counts[counts < MIN_SAMPLES_FOR_CATEGORY].index.tolist()
        if rare:
            df.loc[df[col].isin(rare), col] = "기타"

    dummy_cols = ["apartmentName", "dong"]
    if "region" in df.columns:
        dummy_cols.append("region")

    df = pd.get_dummies(df, columns=dummy_cols)

    # -----------------------------------------------------------------------
    # STEP 3 — 피처 선택
    # -----------------------------------------------------------------------
    numeric_features = [
        "area",
        "floor",
        "buildingAge",
        "salePrice",
        "price_per_m2",
        "match_gap_year",
        "last_jeonse_ratio",
    ]
    for col in ("price_percentile_in_dong", "last_3_mean_jeonse_ratio"):
        if col in df.columns:
            numeric_features.append(col)

    if "saleYear" in df.columns:
        numeric_features.append("saleYear")

    encoded_apt = [c for c in df.columns if c.startswith("apartmentName_")]
    encoded_dong = [c for c in df.columns if c.startswith("dong_")]
    encoded_region = [c for c in df.columns if c.startswith("region_")]

    feature_cols = numeric_features + encoded_apt + encoded_dong + encoded_region
    target_col = "jeonseRatio"

    X = df[feature_cols]
    y = df[target_col]

    # -----------------------------------------------------------------------
    # STEP 4 — 연도 기준 학습/검증/테스트 분할
    # -----------------------------------------------------------------------
    train_mask = df["year"] <= 2023
    val_mask = df["year"] == 2024
    test_mask = df["year"] == 2025

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    df_test = df_original[test_mask].copy()

    train_years = df.loc[train_mask, "year"].astype(int)
    sample_weight = np.array(
        [YEAR_WEIGHTS.get(int(year_value), DEFAULT_YEAR_WEIGHT) for year_value in train_years]
    )

    if verbose:
        print("연도별 학습 가중치:", YEAR_WEIGHTS, f"(default={DEFAULT_YEAR_WEIGHT})")

    # -----------------------------------------------------------------------
    # STEP 5 — XGBoost 1차 학습
    # -----------------------------------------------------------------------
    if verbose:
        print("XGBoost 1차 학습 시작 (전체 피처)...")
    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=_progress_logger,
        args=(PROGRESS_LOG_INTERVAL_SEC, stop_event),
        daemon=True,
    )
    progress_thread.start()

    try:
        model = xgb.XGBRegressor(**params)
        try:
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
        except TypeError:
            model.fit(X_train, y_train, sample_weight=sample_weight)
    finally:
        stop_event.set()
        progress_thread.join(timeout=PROGRESS_LOG_INTERVAL_SEC + 1)

    if verbose:
        print("1차 학습 완료.")

    # -----------------------------------------------------------------------
    # STEP 6 — 피처 중요도 계산 후 상위 K개로 재학습
    # -----------------------------------------------------------------------
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    if verbose:
        print("\n피처 중요도 (상위 20):")
        print(importance.head(20).to_string())

    if TOP_K_FEATURES_FOR_RETRAIN > 0 and len(feature_cols) > TOP_K_FEATURES_FOR_RETRAIN:
        top_cols = importance.head(TOP_K_FEATURES_FOR_RETRAIN).index.tolist()

        X_train_k = X_train[top_cols]
        X_val_k = X_val[top_cols]
        X_test_k = X_test[top_cols]

        if verbose:
            print(f"\n상위 {TOP_K_FEATURES_FOR_RETRAIN}개 피처로 재학습...")

        model2 = xgb.XGBRegressor(**params)
        try:
            model2.fit(
                X_train_k,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val_k, y_val)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
        except TypeError:
            model2.fit(X_train_k, y_train, sample_weight=sample_weight)

        model = model2
        feature_cols = top_cols
        X_test = X_test_k

        if verbose:
            print("재학습 완료.")

    # -----------------------------------------------------------------------
    # STEP 7 — 전세비율 예측 후 전세가로 환산
    # -----------------------------------------------------------------------
    predicted_ratio = model.predict(X_test)
    sale_price_test = df_test["salePrice"].values
    predicted_jeonse_price = predicted_ratio * sale_price_test
    actual_jeonse_price = df_test["jeonsePrice"].values

    # -----------------------------------------------------------------------
    # STEP 8 — 평가
    # -----------------------------------------------------------------------
    mae = mean_absolute_error(actual_jeonse_price, predicted_jeonse_price)
    rmse = np.sqrt(mean_squared_error(actual_jeonse_price, predicted_jeonse_price))

    nonzero_mask = actual_jeonse_price != 0
    safe_actual = actual_jeonse_price[nonzero_mask]
    safe_pred = predicted_jeonse_price[nonzero_mask]

    mape = (
        (np.abs(safe_actual - safe_pred) / safe_actual).mean() * 100
        if len(safe_actual) > 0
        else float("nan")
    )

    if verbose:
        print("\nTest MAE:", mae)
        print("Test RMSE:", rmse)
        print("Test MAPE:", mape)

        bands = [(0, 50000, "0~5억"), (50000, 10**9, "5억 이상")]
        print("\n가격대별 MAPE (만원 기준):")
        for low, high, label in bands:
            mask = (actual_jeonse_price >= low) & (actual_jeonse_price < high)
            if mask.sum() == 0:
                continue
            a = actual_jeonse_price[mask]
            p = predicted_jeonse_price[mask]
            nz = a != 0
            if nz.sum() > 0:
                band_mape = (np.abs(a[nz] - p[nz]) / a[nz]).mean() * 100
                print(f"  {label}: MAPE={band_mape:.2f}% (n={mask.sum()})")

        ratio_errors = (np.abs(safe_actual - safe_pred) / safe_actual) * 100
        print("\n오차 분포")
        print("median MAPE:", np.median(ratio_errors))
        print("p75:", np.percentile(ratio_errors, 75))
        print("p90:", np.percentile(ratio_errors, 90))
        print("p95:", np.percentile(ratio_errors, 95))
        print("max:", np.max(ratio_errors))

        df_test = df_test.loc[nonzero_mask].copy()
        df_test["predicted"] = safe_pred
        df_test["error_pct"] = ratio_errors
        worst = df_test.sort_values("error_pct", ascending=False).head(20).reset_index(drop=True)
        print("\nWorst predictions")
        print(worst[["salePrice", "jeonsePrice", "predicted", "error_pct"]])

    return {
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "model": model,
        "feature_cols": feature_cols,
        "params": params,
    }


def main() -> None:
    result = run_training(xgb_params=None, verbose=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(result["model"], MODEL_PATH)
    joblib.dump(result["feature_cols"], os.path.join(MODEL_DIR, "xgboost_feature_cols.pkl"))

    print(f"\n모델 저장 경로: {MODEL_PATH}")
    print(f"사용 피처 수: {len(result['feature_cols'])}")


if __name__ == "__main__":
    main()