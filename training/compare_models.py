"""저장해 둔 XGBoost와 Embedding MLP를 같은 평가 구간에서 비교한다.

두 모형이 같은 표·같은 학습/검증/평가 나눔을 쓰는지 확인한 뒤,
전세보증금으로 환산한 MAPE·MAE·RMSE를 나란히 출력한다.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from embedding_mlp_model import JeonseEmbeddingMLP
from train_embedding_mlp import (
    MIN_SAMPLES_FOR_CATEGORY,
    encode_mlp_inputs,
    evaluate_price,
    load_split_frame,
    print_eval_report,
    price_mape,
)
from train_xgboost import MODEL_DIR as XGB_MODEL_DIR
from train_xgboost import MODEL_PATH as XGB_MODEL_PATH

# 프로젝트 루트/model 아래의 MLP 산출물
MLP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "embedding_mlp_model.pt",
)
MLP_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "embedding_category_mapping.pkl",
)
XGB_FEATURE_PATH = os.path.join(XGB_MODEL_DIR, "xgboost_feature_cols.pkl")
# 원-핫으로 바꿀 범주 열 이름
DUMMY_COLS = ["apartmentName", "dong", "region"]


def _xgb_predict(split: dict) -> np.ndarray:
    """평가 구간의 전세가율을 XGBoost로 예측한다.

    split: load_split_frame()이 준 사전
        df: 이상치·희소 범주를 처리한 표
        test_mask: 평가 행이면 True인 불리언 시리즈
    반환: 평가 행의 예측 전세가율 (1차원 배열)
    """
    model = joblib.load(XGB_MODEL_PATH)  # 학습된 XGBRegressor
    feature_cols = joblib.load(XGB_FEATURE_PATH)  # 학습 때 쓴 열 순서
    df = split["df"].copy()
    dummy_cols = [c for c in DUMMY_COLS if c in df.columns]
    # get_dummies: 범주 열을 apartmentName_정자동 같은 0/1 열로 펼친다
    encoded = pd.get_dummies(df, columns=dummy_cols)
    # 학습에 있던 열만 맞추고, 없는 열은 0으로 채운다
    X = encoded.reindex(columns=feature_cols, fill_value=0)
    X_test = X.loc[split["test_mask"]]
    return np.asarray(model.predict(X_test), dtype=float)


def _mlp_predict(split: dict) -> np.ndarray:
    """평가 구간의 전세가율을 Embedding MLP로 예측한다.

    split: _xgb_predict와 같은 분할 사전
    반환: 평가 행의 예측 전세가율
    """
    mapping = joblib.load(MLP_MAPPING_PATH)  # 이름→인덱스, 스케일러
    try:
        ckpt = torch.load(MLP_MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(MLP_MODEL_PATH, map_location="cpu")
    # 저장 당시의 크기와 같은 빈 모형을 만든 뒤 가중치를 넣는다
    model = JeonseEmbeddingMLP(
        n_apt=ckpt["n_apt"],
        n_dong=ckpt["n_dong"],
        n_numeric=ckpt["n_numeric"],
        embedding_dim=ckpt["embedding_dim"],
        dropout=ckpt.get("dropout", 0.25),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    df_test = split["df"].loc[split["test_mask"]]
    apt_idx, dong_idx, numeric = encode_mlp_inputs(df_test, mapping, fit_scaler=False)
    # no_grad: 예측만 하므로 기울기를 계산하지 않아 메모리를 줄인다
    with torch.no_grad():
        pred = model(
            torch.from_numpy(np.ascontiguousarray(apt_idx)),
            torch.from_numpy(np.ascontiguousarray(dong_idx)),
            torch.from_numpy(np.ascontiguousarray(numeric)),
        )
    return pred.numpy()


def _band_mapes(metrics: dict) -> dict:
    """매매가(만원) 구간별로 MAPE를 나눈다.

    metrics: evaluate_price() 결과. actual·predicted는 보증금 배열
    반환: {"0~5억": MAPE, "0~5억 n": 건수, ...}
    """
    actual = metrics["actual"]
    predicted = metrics["predicted"]
    out = {}
    for low, high, label in ((0, 50000, "0~5억"), (50000, 10**9, "5억 이상")):
        mask = (actual >= low) & (actual < high)  # 그 가격대에 속하는 행
        out[label] = price_mape(actual[mask], predicted[mask]) if mask.sum() else float("nan")
        out[f"{label} n"] = int(mask.sum())
    return out


def _error_dist(metrics: dict) -> dict:
    """상대오차(%)의 중앙값·분위수를 계산한다.

    평균만 보면 극단 한 건에 끌리므로, 분포의 중간과 상단을 같이 본다.
    """
    actual = metrics["actual"]
    predicted = metrics["predicted"]
    nonzero = actual != 0  # 0으로 나누기 방지
    err = (np.abs(actual[nonzero] - predicted[nonzero]) / actual[nonzero]) * 100
    return {
        "median MAPE": float(np.median(err)),
        "p75": float(np.percentile(err, 75)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "max": float(np.max(err)),
    }


def main() -> None:
    """두 모형의 평가 지표를 표로 출력한다."""
    missing = []
    for path in (XGB_MODEL_PATH, XGB_FEATURE_PATH, MLP_MODEL_PATH, MLP_MAPPING_PATH):
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("아래 파일이 없습니다. XGBoost와 MLP를 먼저 학습하세요.")
        for path in missing:
            print(" ", path)
        sys.exit(1)

    split = load_split_frame(verbose=True)
    df_test = split["df"].loc[split["test_mask"]]
    print(f"비교 Test n={len(df_test)} (MIN_SAMPLES_FOR_CATEGORY={MIN_SAMPLES_FOR_CATEGORY})")

    xgb_ratio = _xgb_predict(split)  # 예측 전세가율
    mlp_ratio = _mlp_predict(split)
    xgb_metrics = evaluate_price(xgb_ratio, df_test)  # 비율 × 매매가 → 보증금 오차
    mlp_metrics = evaluate_price(mlp_ratio, df_test)

    print_eval_report("XGBoost", xgb_metrics, df_test)
    print_eval_report("EmbeddingMLP", mlp_metrics, df_test)

    xgb_band = _band_mapes(xgb_metrics)
    mlp_band = _band_mapes(mlp_metrics)
    xgb_dist = _error_dist(xgb_metrics)
    mlp_dist = _error_dist(mlp_metrics)

    # 한 열이 한 모형. **dist는 분위수 키를 그대로 펼쳐 넣는다
    table = pd.DataFrame(
        {
            "XGBoost": {
                "Test MAPE": xgb_metrics["mape"],
                "Test MAE": xgb_metrics["mae"],
                "Test RMSE": xgb_metrics["rmse"],
                "0~5억 MAPE": xgb_band["0~5억"],
                "5억 이상 MAPE": xgb_band["5억 이상"],
                **xgb_dist,
            },
            "EmbeddingMLP": {
                "Test MAPE": mlp_metrics["mape"],
                "Test MAE": mlp_metrics["mae"],
                "Test RMSE": mlp_metrics["rmse"],
                "0~5억 MAPE": mlp_band["0~5억"],
                "5억 이상 MAPE": mlp_band["5억 이상"],
                **mlp_dist,
            },
        }
    )
    print("\n=== XGBoost vs Embedding MLP ===")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
