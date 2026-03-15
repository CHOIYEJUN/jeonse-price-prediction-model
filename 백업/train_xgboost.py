"""
아파트 전세비율(jeonseRatio) 예측용 XGBoost 회귀 모델 학습.
시점 기반 분할, 원핫 인코딩 사용, 모델은 model/xgboost_jeonse_model.pkl 로 저장.
"""

import os
import time
import threading
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

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

# XGBoost 하이퍼파라미터
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

PROGRESS_LOG_INTERVAL_SEC = 10.0


def _progress_logger(interval_sec: float, stop_event: threading.Event) -> None:
    """10초 간격으로 '진행 중' 로그를 출력하는 스레드."""
    start = time.perf_counter()
    while not stop_event.is_set():
        stop_event.wait(interval_sec)
        if stop_event.is_set():
            break
        elapsed = int(time.perf_counter() - start)
        print(f"  진행 중: 학습 {elapsed}초 경과")


def main() -> None:
    # ---------------------------------------------------------------------------
    # STEP 1 — 데이터셋 로드
    # ---------------------------------------------------------------------------
    df = pd.read_csv(DATASET_PATH)

    # ---------------------------------------------------------------------------
    # STEP 2 — 범주형 컬럼 원핫 인코딩
    # ---------------------------------------------------------------------------
    df = pd.get_dummies(df, columns=["apartmentName", "dong"])

    # ---------------------------------------------------------------------------
    # STEP 3 — 피처 선택
    # ---------------------------------------------------------------------------
    numeric_features = ["area", "floor", "buildingAge", "salePrice", "price_per_m2"]
    if "saleYear" in df.columns:
        numeric_features = numeric_features + ["saleYear"]
    encoded_apt = [c for c in df.columns if c.startswith("apartmentName_")]
    encoded_dong = [c for c in df.columns if c.startswith("dong_")]
    feature_cols = numeric_features + encoded_apt + encoded_dong
    target_col = "jeonseRatio"

    X = df[feature_cols]
    y = df[target_col]

    # ---------------------------------------------------------------------------
    # STEP 4 — 연도 기준 학습/검증/테스트 분할
    # ---------------------------------------------------------------------------
    train_mask = df["year"] <= 2023
    val_mask = df["year"] == 2024
    test_mask = df["year"] == 2025

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    df_test = df[test_mask].copy()

    # ---------------------------------------------------------------------------
    # STEP 5 — XGBoost 모델 학습
    # ---------------------------------------------------------------------------
    print("XGBoost 학습 시작...")
    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=_progress_logger,
        args=(PROGRESS_LOG_INTERVAL_SEC, stop_event),
        daemon=True,
    )
    progress_thread.start()
    try:
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
    finally:
        stop_event.set()
        progress_thread.join(timeout=PROGRESS_LOG_INTERVAL_SEC + 1)
    print("학습 완료.")

    # ---------------------------------------------------------------------------
    # STEP 6 — 전세비율 예측 후 전세가로 환산
    # ---------------------------------------------------------------------------
    predicted_ratio = model.predict(X_test)
    sale_price_test = df_test["salePrice"].values
    predicted_jeonse_price = predicted_ratio * sale_price_test
    actual_jeonse_price = df_test["jeonsePrice"].values

    # ---------------------------------------------------------------------------
    # STEP 7 — 평가 (전세가 기준 MAE, MAPE)
    # ---------------------------------------------------------------------------
    mae = mean_absolute_error(actual_jeonse_price, predicted_jeonse_price)
    # MAPE: mean(|실제 - 예측| / 실제) * 100, 실제값 0 제외
    nonzero = actual_jeonse_price != 0
    mape = (
        (abs(actual_jeonse_price[nonzero] - predicted_jeonse_price[nonzero]) / actual_jeonse_price[nonzero]).mean() * 100
        if nonzero.sum() > 0
        else float("nan")
    )

    print("Test MAE:", mae)
    print("Test MAPE:", mape)

    # ---------------------------------------------------------------------------
    # STEP 8 — 피처 중요도 출력
    # ---------------------------------------------------------------------------
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n피처 중요도:")
    #print(importance.to_string())

    # ---------------------------------------------------------------------------
    # STEP 9 — 모델 저장
    # ---------------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n모델 저장 경로: {MODEL_PATH}")


if __name__ == "__main__":
    main()
