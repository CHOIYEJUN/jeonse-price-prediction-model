"""
아파트 전세비율(jeonseRatio) 예측용 XGBoost 회귀 모델 학습.
eval: train ≤ 2025 / val 2026 H1 / test 2026 H2
production: 전 기간 재학습 후 모델 저장 (공식 성능은 eval만 사용)
"""

import argparse
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

TRAIN_YM_MAX = 202512
VAL_YM_MIN = 202601
VAL_YM_MAX = 202606
TEST_YM_MIN = 202607

# eval 학습에 실제로 들어가는 연도 가중치 (2026은 eval에 없음)
EVAL_YEAR_WEIGHTS = {
    2025: 1.0,
    2024: 0.8,
    2023: 0.7,
    2022: 0.6,
    2021: 0.5,
    2020: 0.4,
}

# production은 최신 연도 가중치 최대
PROD_YEAR_WEIGHTS = {
    2026: 1.0,
    2025: 0.9,
    2024: 0.8,
    2023: 0.7,
    2022: 0.6,
    2021: 0.5,
    2020: 0.4,
}
DEFAULT_YEAR_WEIGHT = 0.4

TOP_K_FEATURES_FOR_RETRAIN = 100

OUTLIER_FILTER_MODE = "percentile"  # "bounds" | "percentile"
JEONSE_RATIO_MIN = 0.10
JEONSE_RATIO_MAX = 0.95
OUTLIER_PERCENTILE_LOW = 10
OUTLIER_PERCENTILE_HIGH = 5

DUMMY_PREFIXES = ("apartmentName_", "dong_", "region_")


def _progress_logger(interval_sec: float, stop_event: threading.Event) -> None:
    start = time.perf_counter()
    while not stop_event.is_set():
        stop_event.wait(interval_sec)
        if stop_event.is_set():
            break
        elapsed = int(time.perf_counter() - start)
        print(f"  진행 중: 학습 {elapsed}초 경과")


def _price_mape(actual, predicted):
    nonzero = actual != 0
    if nonzero.sum() == 0:
        return float("nan")
    a = actual[nonzero]
    p = predicted[nonzero]
    return (np.abs(a - p) / a).mean() * 100


def _fit_xgb(params, X_train, y_train, sample_weight, X_val=None, y_val=None):
    model = xgb.XGBRegressor(**params)
    fit_kwargs = {
        "sample_weight": sample_weight,
    }
    try:
        if X_val is not None and y_val is not None and len(X_val) > 0:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
                **fit_kwargs,
            )
        else:
            model.fit(X_train, y_train, verbose=False, **fit_kwargs)
    except TypeError:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def _eval_price_metrics(model, X, df_rows):
    predicted_ratio = model.predict(X)
    sale_price = df_rows["salePrice"].values
    predicted = predicted_ratio * sale_price
    actual = df_rows["jeonsePrice"].values
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = _price_mape(actual, predicted)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "predicted": predicted,
        "actual": actual,
    }


def run_training(xgb_params=None, verbose=True, mode="eval"):
    """
    데이터 로드·전처리·학습·평가까지 수행.
    mode: "eval" | "production"
    """
    if mode not in ("eval", "production"):
        raise ValueError("mode must be 'eval' or 'production'")

    params = dict(XGB_PARAMS) if xgb_params is None else dict(xgb_params)
    params.setdefault("random_state", 42)
    year_weights = EVAL_YEAR_WEIGHTS if mode == "eval" else PROD_YEAR_WEIGHTS

    df = pd.read_csv(DATASET_PATH)

    if "last_jeonse_ratio" in df.columns:
        df = df.dropna(subset=["last_jeonse_ratio"]).copy()

    if "jeonseYm" not in df.columns:
        raise ValueError("dataset에 jeonseYm이 없습니다. build_dataset.py를 다시 실행하세요.")

    df["jeonseYm"] = pd.to_numeric(df["jeonseYm"], errors="coerce")
    df = df.dropna(subset=["jeonseYm", "jeonseRatio", "salePrice", "jeonsePrice"]).copy()
    df["jeonseYm"] = df["jeonseYm"].astype(int)

    if mode == "eval":
        train_mask = df["jeonseYm"] <= TRAIN_YM_MAX
        val_mask = (df["jeonseYm"] >= VAL_YM_MIN) & (df["jeonseYm"] <= VAL_YM_MAX)
        test_mask = df["jeonseYm"] >= TEST_YM_MIN
    else:
        train_mask = pd.Series(True, index=df.index)
        val_mask = pd.Series(False, index=df.index)
        test_mask = pd.Series(False, index=df.index)

    if verbose:
        if mode == "eval":
            print(
                "시점 분할: "
                f"train jeonseYm ≤ {TRAIN_YM_MAX} / "
                f"val {VAL_YM_MIN}–{VAL_YM_MAX} / "
                f"test ≥ {TEST_YM_MIN}"
            )
            print(
                f"분할 행 수(필터 전): train={int(train_mask.sum())} "
                f"val={int(val_mask.sum())} test={int(test_mask.sum())}"
            )
        else:
            print(f"production 모드: 전 기간 학습 n={len(df)} (공식 성능은 eval 모드만 사용)")

    if "jeonseRatio" in df.columns:
        train_ratio = df.loc[train_mask, "jeonseRatio"]
        if OUTLIER_FILTER_MODE == "bounds":
            low_q, high_q = JEONSE_RATIO_MIN, JEONSE_RATIO_MAX
            if verbose:
                print(
                    f"전세비율 이상치 제거 (bounds, train 기준): "
                    f"jeonseRatio [{low_q}, {high_q}] 밖 제거"
                )
        else:
            low_q = np.percentile(train_ratio, OUTLIER_PERCENTILE_LOW)
            high_q = np.percentile(train_ratio, 100 - OUTLIER_PERCENTILE_HIGH)
            if verbose:
                print(
                    f"전세비율 이상치 제거 (percentile, train 기준): "
                    f"하위 {OUTLIER_PERCENTILE_LOW}%·상위 {OUTLIER_PERCENTILE_HIGH}% "
                    f"유효 구간 [{low_q:.3f}, {high_q:.3f}]"
                )
        keep = (df["jeonseRatio"] >= low_q) & (df["jeonseRatio"] <= high_q)
        n_removed = int((~keep).sum())
        if verbose:
            print(f"이상치 제거: {n_removed}건")
        df = df.loc[keep].copy()
        train_mask = train_mask.loc[keep]
        val_mask = val_mask.loc[keep]
        test_mask = test_mask.loc[keep]

    df_original = df.copy()

    dummy_cols = [c for c in ["apartmentName", "dong", "region"] if c in df.columns]
    for col in dummy_cols:
        counts = df.loc[train_mask, col].value_counts()
        keep_vals = set(counts[counts >= MIN_SAMPLES_FOR_CATEGORY].index.tolist())
        df[col] = df[col].where(df[col].isin(keep_vals), "기타")

    df_encoded = pd.get_dummies(df, columns=dummy_cols)
    dummy_feature_cols = [
        c for c in df_encoded.columns if c.startswith(DUMMY_PREFIXES)
    ]
    train_dummy_cols = [
        c for c in dummy_feature_cols if df_encoded.loc[train_mask, c].sum() > 0
    ]

    numeric_features = [
        "area",
        "floor",
        "buildingAge",
        "salePrice",
        "price_per_m2",
        "last_jeonse_ratio",
    ]
    for col in (
        "match_gap_days",
        "match_gap_year",
        "price_percentile_in_dong",
        "last_3_mean_jeonse_ratio",
        "saleYear",
    ):
        if col in df_encoded.columns:
            numeric_features.append(col)

    feature_cols = numeric_features + train_dummy_cols
    target_col = "jeonseRatio"

    X = df_encoded[feature_cols]
    y = df_encoded[target_col]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    df_val = df_original[val_mask].copy()
    df_test = df_original[test_mask].copy()

    if len(X_train) == 0:
        raise ValueError("학습 데이터가 비었습니다.")

    train_years = df_encoded.loc[train_mask, "year"].astype(int)
    sample_weight = np.array(
        [year_weights.get(int(year_value), DEFAULT_YEAR_WEIGHT) for year_value in train_years]
    )

    if verbose:
        print("연도별 학습 가중치:", year_weights, f"(default={DEFAULT_YEAR_WEIGHT})")
        print(f"학습 행 수(필터 후): train={len(X_train)} val={len(X_val)} test={len(X_test)}")

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
        model = _fit_xgb(
            params,
            X_train,
            y_train,
            sample_weight,
            X_val if mode == "eval" and len(X_val) > 0 else None,
            y_val if mode == "eval" and len(X_val) > 0 else None,
        )
    finally:
        stop_event.set()
        progress_thread.join(timeout=PROGRESS_LOG_INTERVAL_SEC + 1)

    if verbose:
        print("1차 학습 완료.")

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    if verbose:
        print("\n피처 중요도 (상위 20):")
        print(importance.head(20).to_string())

    if TOP_K_FEATURES_FOR_RETRAIN > 0 and len(feature_cols) > TOP_K_FEATURES_FOR_RETRAIN:
        top_cols = importance.head(TOP_K_FEATURES_FOR_RETRAIN).index.tolist()
        X_train_k = X_train[top_cols]
        X_val_k = X_val[top_cols] if len(X_val) else X_val
        X_test_k = X_test[top_cols] if len(X_test) else X_test

        if verbose:
            print(f"\n상위 {TOP_K_FEATURES_FOR_RETRAIN}개 피처로 재학습...")

        model = _fit_xgb(
            params,
            X_train_k,
            y_train,
            sample_weight,
            X_val_k if mode == "eval" and len(X_val_k) > 0 else None,
            y_val if mode == "eval" and len(X_val_k) > 0 else None,
        )
        feature_cols = top_cols
        X_val = X_val_k
        X_test = X_test_k
        if verbose:
            print("재학습 완료.")

    val_metrics = {"mape": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    test_metrics = {"mape": float("nan"), "mae": float("nan"), "rmse": float("nan")}

    if mode == "eval":
        if len(X_val) > 0:
            val_metrics = _eval_price_metrics(model, X_val, df_val)
        if len(X_test) > 0:
            test_metrics = _eval_price_metrics(model, X_test, df_test)

        if verbose:
            print("\nVal MAE:", val_metrics["mae"])
            print("Val RMSE:", val_metrics["rmse"])
            print("Val MAPE:", val_metrics["mape"])
            print("\nTest MAE:", test_metrics["mae"])
            print("Test RMSE:", test_metrics["rmse"])
            print("Test MAPE:", test_metrics["mape"])

            if len(X_test) > 0:
                actual = test_metrics["actual"]
                predicted = test_metrics["predicted"]
                bands = [(0, 50000, "0~5억"), (50000, 10**9, "5억 이상")]
                print("\n가격대별 MAPE (만원 기준, Test):")
                for low, high, label in bands:
                    mask = (actual >= low) & (actual < high)
                    if mask.sum() == 0:
                        continue
                    band_mape = _price_mape(actual[mask], predicted[mask])
                    print(f"  {label}: MAPE={band_mape:.2f}% (n={int(mask.sum())})")

                nonzero = actual != 0
                ratio_errors = (np.abs(actual[nonzero] - predicted[nonzero]) / actual[nonzero]) * 100
                print("\n오차 분포 (Test)")
                print("median MAPE:", np.median(ratio_errors))
                print("p75:", np.percentile(ratio_errors, 75))
                print("p90:", np.percentile(ratio_errors, 90))
                print("p95:", np.percentile(ratio_errors, 95))
                print("max:", np.max(ratio_errors))

                df_test_out = df_test.loc[nonzero].copy()
                df_test_out["predicted"] = predicted[nonzero]
                df_test_out["error_pct"] = ratio_errors
                worst = df_test_out.sort_values("error_pct", ascending=False).head(20).reset_index(drop=True)
                print("\nWorst predictions")
                cols = ["salePrice", "jeonsePrice", "predicted", "error_pct"]
                if "jeonseYm" in worst.columns:
                    cols = ["jeonseYm"] + cols
                print(worst[cols])
    elif verbose:
        print("\nproduction 모드는 Test MAPE를 출력하지 않습니다. 공식 성능은 eval 모드를 사용하세요.")

    return {
        "mape": test_metrics["mape"],
        "val_mape": val_metrics["mape"],
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "model": model,
        "feature_cols": feature_cols,
        "params": params,
        "mode": mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["eval", "production"],
        default="eval",
        help="eval: 시점 홀드아웃 평가 / production: 전 기간 재학습 후 저장",
    )
    args = parser.parse_args()

    result = run_training(xgb_params=None, verbose=True, mode=args.mode)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(result["model"], MODEL_PATH)
    joblib.dump(result["feature_cols"], os.path.join(MODEL_DIR, "xgboost_feature_cols.pkl"))

    print(f"\n모델 저장 경로: {MODEL_PATH}")
    print(f"사용 피처 수: {len(result['feature_cols'])}")
    print(f"mode: {args.mode}")


if __name__ == "__main__":
    main()
