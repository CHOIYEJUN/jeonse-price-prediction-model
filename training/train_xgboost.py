"""아파트 전세가율(jeonseRatio)을 XGBoost로 학습한다.

eval
    학습: 2025년 12월까지 / 검증: 2026년 1~6월 / 평가: 2026년 7월 이후
    공식 성능은 이 모드에서만 본다.
production
    전 기간으로 다시 학습해 저장한다. 배포용. 평가 숫자는 출력하지 않는다.
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
# __file__ = 이 파일 경로. 상위 폴더가 training, 그 위가 프로젝트 루트
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "merged_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_jeonse_model.pkl")

# ---------------------------------------------------------------------------
# XGBoost 설정 (하이퍼파라미터: 학습이 아니라 사람이 고르는 값)
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 400,  # 트리 그루 수
    "max_depth": 6,  # 한 트리의 최대 깊이
    "learning_rate": 0.07,  # 한 그루가 잔차를 얼마나 반영할지
    "subsample": 0.8,  # 각 트리에 쓰는 행의 비율
    "colsample_bytree": 0.85,  # 각 트리에 쓰는 열의 비율
    "min_child_weight": 12,  # 잎에 필요한 최소 가중치. 클수록 단순한 트리
    "reg_alpha": 0.2,  # L1 규제
    "reg_lambda": 1.0,  # L2 규제
    "random_state": 42,  # 난수 고정. 같은 결과 재현
}

PROGRESS_LOG_INTERVAL_SEC = 10.0  # 학습 중 몇 초마다 "진행 중"을 찍을지
MIN_SAMPLES_FOR_CATEGORY = 15  # 이보다 적은 단지·동은 "기타"로 묶는다
EARLY_STOPPING_ROUNDS = 50  # 검증 점수가 50라운드 안 좋아지면 멈춘다

# jeonseYm은 202512처럼 연월을 한 정수로 쓴다
TRAIN_YM_MAX = 202512
VAL_YM_MIN = 202601
VAL_YM_MAX = 202606
TEST_YM_MIN = 202607

# eval 학습에 실제로 들어가는 연도 가중치 (2026은 eval 학습에 없음)
EVAL_YEAR_WEIGHTS = {
    2025: 1.0,
    2024: 0.8,
    2023: 0.7,
    2022: 0.6,
    2021: 0.5,
    2020: 0.4,
}

# production은 최신 연도 가중치를 가장 크게
PROD_YEAR_WEIGHTS = {
    2026: 1.0,
    2025: 0.9,
    2024: 0.8,
    2023: 0.7,
    2022: 0.6,
    2021: 0.5,
    2020: 0.4,
}
DEFAULT_YEAR_WEIGHT = 0.4  # 표에 없는 연도에 줄 기본 무게

TOP_K_FEATURES_FOR_RETRAIN = 100  # 1차 학습 후 중요도 상위 몇 개만 다시 학습할지

OUTLIER_FILTER_MODE = "percentile"  # "bounds" | "percentile"
JEONSE_RATIO_MIN = 0.10  # bounds 모드일 때 하한
JEONSE_RATIO_MAX = 0.95  # bounds 모드일 때 상한
OUTLIER_PERCENTILE_LOW = 10  # 학습 분포 하위 이 %를 자른다
OUTLIER_PERCENTILE_HIGH = 5  # 학습 분포 상위 이 %를 자른다

# 원-핫으로 만든 열 이름의 앞부분
DUMMY_PREFIXES = ("apartmentName_", "dong_", "region_")


def _progress_logger(interval_sec: float, stop_event: threading.Event) -> None:
    """다른 스레드에서 일정 간격으로 경과 시간을 출력한다.

    interval_sec: 몇 초마다 한 줄 찍을지
    stop_event: 학습이 끝나면 set()되는 이벤트. 켜지면 이 함수도 끝난다.
    """
    start = time.perf_counter()  # 시작 시각 (초, 고정밀)
    while not stop_event.is_set():
        stop_event.wait(interval_sec)  # interval_sec 동안 자거나, 중간에 set되면 깨어남
        if stop_event.is_set():
            break
        elapsed = int(time.perf_counter() - start)
        print(f"  진행 중: 학습 {elapsed}초 경과")


def _price_mape(actual, predicted):
    """전세보증금 기준 평균절대백분율오차(MAPE, %)를 계산한다.

    actual: 실제 보증금 배열
    predicted: 추정 보증금 배열
    반환: 상대오차의 평균 × 100. 실제값이 모두 0이면 nan.
    """
    nonzero = actual != 0  # True/False 배열. 0으로 나누지 않기 위함
    if nonzero.sum() == 0:
        return float("nan")
    a = actual[nonzero]
    p = predicted[nonzero]
    return (np.abs(a - p) / a).mean() * 100


def _fit_xgb(params, X_train, y_train, sample_weight, X_val=None, y_val=None):
    """XGBoost 회귀 모형을 학습해 돌려준다.

    params: n_estimators 등이 들어 있는 딕셔너리
    X_train: 학습 입력 (행=거래, 열=피처)
    y_train: 학습 정답 (전세가율)
    sample_weight: 행마다 다른 가중치. 최근 연도를 더 크게
    X_val, y_val: 검증 표. 있으면 조기 종료에 쓴다
    반환: 학습된 XGBRegressor
    """
    model = xgb.XGBRegressor(**params)  # **는 딕셔너리를 인자로 펼친다
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
        # 버전마다 early_stopping 인자 이름이 다를 수 있어 최소 인자로 재시도
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def _eval_price_metrics(model, X, df_rows):
    """전세가율 예측을 보증금으로 바꾼 뒤 MAE·RMSE·MAPE를 계산한다.

    model: 학습된 XGBRegressor
    X: 그 구간의 입력 표
    df_rows: 같은 행의 원본 표 (salePrice, jeonsePrice가 필요)
    반환: mae, rmse, mape, predicted, actual 이 있는 사전
    """
    predicted_ratio = model.predict(X)  # 전세가율
    sale_price = df_rows["salePrice"].values
    predicted = predicted_ratio * sale_price  # 추정 보증금
    actual = df_rows["jeonsePrice"].values  # 실제 보증금
    mae = mean_absolute_error(actual, predicted)  # 평균 절대오차
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))  # 큰 오차에 민감
    mape = _price_mape(actual, predicted)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "predicted": predicted,
        "actual": actual,
    }


def run_training(xgb_params=None, verbose=True, mode="eval"):
    """데이터 로드부터 학습·평가까지 한 번에 수행한다.

    xgb_params: None이면 XGB_PARAMS를 쓴다. 튜닝 스크립트가 다른 설정을 넣을 수 있다.
    verbose: True면 진행 로그를 출력한다.
    mode: "eval" 또는 "production"
    반환: mape, val_mape, model, feature_cols 등이 있는 사전
    """
    if mode not in ("eval", "production"):
        raise ValueError("mode must be 'eval' or 'production'")

    params = dict(XGB_PARAMS) if xgb_params is None else dict(xgb_params)
    params.setdefault("random_state", 42)  # 없으면 42를 넣는다
    year_weights = EVAL_YEAR_WEIGHTS if mode == "eval" else PROD_YEAR_WEIGHTS

    df = pd.read_csv(DATASET_PATH)  # 학습용 전체 표

    # 직전 전세가율이 없는 첫 거래는 학습에서 뺀다
    if "last_jeonse_ratio" in df.columns:
        df = df.dropna(subset=["last_jeonse_ratio"]).copy()

    if "jeonseYm" not in df.columns:
        raise ValueError("dataset에 jeonseYm이 없습니다. build_dataset.py를 다시 실행하세요.")

    df["jeonseYm"] = pd.to_numeric(df["jeonseYm"], errors="coerce")
    df = df.dropna(subset=["jeonseYm", "jeonseRatio", "salePrice", "jeonsePrice"]).copy()
    df["jeonseYm"] = df["jeonseYm"].astype(int)

    # mask: 각 행이 학습/검증/평가 중 어디에 속하는지 True/False
    if mode == "eval":
        train_mask = df["jeonseYm"] <= TRAIN_YM_MAX
        val_mask = (df["jeonseYm"] >= VAL_YM_MIN) & (df["jeonseYm"] <= VAL_YM_MAX)
        test_mask = df["jeonseYm"] >= TEST_YM_MIN
    else:
        # production: 모든 행을 학습에 쓰고 검증·평가는 비운다
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

    # 이상치: 자르는 선은 학습 구간 분포로만 정한 뒤, 세 구간에 같은 자를 댄다
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

    df_original = df.copy()  # 원-핫 전의 표. 평가 때 매매가·보증금을 읽는다

    # 학습에서 적게 나온 단지·동은 "기타"로 합친다
    dummy_cols = [c for c in ["apartmentName", "dong", "region"] if c in df.columns]
    for col in dummy_cols:
        counts = df.loc[train_mask, col].value_counts()  # 학습 구간에서만 센다
        keep_vals = set(counts[counts >= MIN_SAMPLES_FOR_CATEGORY].index.tolist())
        df[col] = df[col].where(df[col].isin(keep_vals), "기타")

    # 범주 열을 0/1 열로 펼친다. apartmentName_정자동 같은 이름이 생긴다
    df_encoded = pd.get_dummies(df, columns=dummy_cols)
    dummy_feature_cols = [
        c for c in df_encoded.columns if c.startswith(DUMMY_PREFIXES)
    ]
    # 학습에 한 번도 안 나온 더미 열은 뺀다
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

    feature_cols = numeric_features + train_dummy_cols  # 모형이 보는 열 목록
    target_col = "jeonseRatio"  # 맞힐 값

    X = df_encoded[feature_cols]  # 입력
    y = df_encoded[target_col]  # 정답

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    df_val = df_original[val_mask].copy()
    df_test = df_original[test_mask].copy()

    if len(X_train) == 0:
        raise ValueError("학습 데이터가 비었습니다.")

    train_years = df_encoded.loc[train_mask, "year"].astype(int)
    # 각 학습 행에 연도별 가중치를 붙인다
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
        daemon=True,  # 메인 종료 시 같이 끝난다
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
        # 성공이든 실패든 진행 로그 스레드를 멈춘다
        stop_event.set()
        progress_thread.join(timeout=PROGRESS_LOG_INTERVAL_SEC + 1)

    if verbose:
        print("1차 학습 완료.")

    # feature_importances_: 각 열이 분할에 얼마나 쓰였는지. 합이 1 근처
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    if verbose:
        print("\n피처 중요도 (상위 20):")
        print(importance.head(20).to_string())

    # 열이 너무 많으면 상위 K개만 남기고 다시 학습한다
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
    """명령줄에서 모드를 받아 학습하고 모형을 파일로 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["eval", "production"],
        default="eval",
        help="eval: 시점 홀드아웃 평가 / production: 전 기간 재학습 후 저장",
    )
    args = parser.parse_args()  # 예: --mode eval

    result = run_training(xgb_params=None, verbose=True, mode=args.mode)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(result["model"], MODEL_PATH)
    joblib.dump(result["feature_cols"], os.path.join(MODEL_DIR, "xgboost_feature_cols.pkl"))

    print(f"\n모델 저장 경로: {MODEL_PATH}")
    print(f"사용 피처 수: {len(result['feature_cols'])}")
    print(f"mode: {args.mode}")


if __name__ == "__main__":
    main()
