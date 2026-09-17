"""저장된 XGBoost와 Embedding MLP를 같은 eval 분할에서 비교한다."""

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
DUMMY_COLS = ["apartmentName", "dong", "region"]


def _xgb_predict(split: dict) -> np.ndarray:
    model = joblib.load(XGB_MODEL_PATH)
    feature_cols = joblib.load(XGB_FEATURE_PATH)
    df = split["df"].copy()
    dummy_cols = [c for c in DUMMY_COLS if c in df.columns]
    encoded = pd.get_dummies(df, columns=dummy_cols)
    X = encoded.reindex(columns=feature_cols, fill_value=0)
    X_test = X.loc[split["test_mask"]]
    return np.asarray(model.predict(X_test), dtype=float)


def _mlp_predict(split: dict) -> np.ndarray:
    mapping = joblib.load(MLP_MAPPING_PATH)
    try:
        ckpt = torch.load(MLP_MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(MLP_MODEL_PATH, map_location="cpu")
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
    with torch.no_grad():
        pred = model(
            torch.from_numpy(np.ascontiguousarray(apt_idx)),
            torch.from_numpy(np.ascontiguousarray(dong_idx)),
            torch.from_numpy(np.ascontiguousarray(numeric)),
        )
    return pred.numpy()


def _band_mapes(metrics: dict) -> dict:
    actual = metrics["actual"]
    predicted = metrics["predicted"]
    out = {}
    for low, high, label in ((0, 50000, "0~5억"), (50000, 10**9, "5억 이상")):
        mask = (actual >= low) & (actual < high)
        out[label] = price_mape(actual[mask], predicted[mask]) if mask.sum() else float("nan")
        out[f"{label} n"] = int(mask.sum())
    return out


def _error_dist(metrics: dict) -> dict:
    actual = metrics["actual"]
    predicted = metrics["predicted"]
    nonzero = actual != 0
    err = (np.abs(actual[nonzero] - predicted[nonzero]) / actual[nonzero]) * 100
    return {
        "median MAPE": float(np.median(err)),
        "p75": float(np.percentile(err, 75)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "max": float(np.max(err)),
    }


def main() -> None:
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

    xgb_ratio = _xgb_predict(split)
    mlp_ratio = _mlp_predict(split)
    xgb_metrics = evaluate_price(xgb_ratio, df_test)
    mlp_metrics = evaluate_price(mlp_ratio, df_test)

    print_eval_report("XGBoost", xgb_metrics, df_test)
    print_eval_report("EmbeddingMLP", mlp_metrics, df_test)

    xgb_band = _band_mapes(xgb_metrics)
    mlp_band = _band_mapes(mlp_metrics)
    xgb_dist = _error_dist(xgb_metrics)
    mlp_dist = _error_dist(mlp_metrics)

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
