"""
apartmentName / dong Entity Embedding + MLP로 전세가율 예측.
분할·이상치·가중치·수치 피처는 train_xgboost.py의 eval 설정과 동일하게 복제.
"""

import os
import random
import threading
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from embedding_mlp_model import JeonseEmbeddingMLP

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "merged_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "embedding_mlp_model.pt")
MAPPING_PATH = os.path.join(MODEL_DIR, "embedding_category_mapping.pkl")

# ---------------------------------------------------------------------------
# train_xgboost.py eval 설정과 동일
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
MIN_SAMPLES_FOR_CATEGORY = 15
TRAIN_YM_MAX = 202512
VAL_YM_MIN = 202601
VAL_YM_MAX = 202606
TEST_YM_MIN = 202607
EVAL_YEAR_WEIGHTS = {
    2025: 1.0,
    2024: 0.8,
    2023: 0.7,
    2022: 0.6,
    2021: 0.5,
    2020: 0.4,
}
DEFAULT_YEAR_WEIGHT = 0.4
OUTLIER_PERCENTILE_LOW = 10
OUTLIER_PERCENTILE_HIGH = 5
UNKNOWN_IDX = 0
EMBEDDING_DIM = 8
DROPOUT = 0.25
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
PATIENCE = 50
PROGRESS_LOG_INTERVAL_SEC = 10.0


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _progress_logger(interval_sec: float, stop_event: threading.Event) -> None:
    start = time.perf_counter()
    while not stop_event.is_set():
        stop_event.wait(interval_sec)
        if stop_event.is_set():
            break
        elapsed = int(time.perf_counter() - start)
        print(f"  진행 중: 학습 {elapsed}초 경과")


def price_mape(actual, predicted):
    nonzero = actual != 0
    if nonzero.sum() == 0:
        return float("nan")
    a = actual[nonzero]
    p = predicted[nonzero]
    return float((np.abs(a - p) / a).mean() * 100)


def numeric_feature_list(df: pd.DataFrame) -> list:
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
        if col in df.columns:
            numeric_features.append(col)
    return numeric_features


def load_split_frame(verbose: bool = True) -> dict:
    df = pd.read_csv(DATASET_PATH)
    if "last_jeonse_ratio" in df.columns:
        df = df.dropna(subset=["last_jeonse_ratio"]).copy()
    if "jeonseYm" not in df.columns:
        raise ValueError("dataset에 jeonseYm이 없습니다. build_dataset.py를 다시 실행하세요.")

    df["jeonseYm"] = pd.to_numeric(df["jeonseYm"], errors="coerce")
    df = df.dropna(subset=["jeonseYm", "jeonseRatio", "salePrice", "jeonsePrice"]).copy()
    df["jeonseYm"] = df["jeonseYm"].astype(int)

    train_mask = df["jeonseYm"] <= TRAIN_YM_MAX
    val_mask = (df["jeonseYm"] >= VAL_YM_MIN) & (df["jeonseYm"] <= VAL_YM_MAX)
    test_mask = df["jeonseYm"] >= TEST_YM_MIN

    if verbose:
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

    train_ratio = df.loc[train_mask, "jeonseRatio"]
    low_q = np.percentile(train_ratio, OUTLIER_PERCENTILE_LOW)
    high_q = np.percentile(train_ratio, 100 - OUTLIER_PERCENTILE_HIGH)
    keep = (df["jeonseRatio"] >= low_q) & (df["jeonseRatio"] <= high_q)
    if verbose:
        print(
            f"전세비율 이상치 제거 (percentile, train 기준): "
            f"하위 {OUTLIER_PERCENTILE_LOW}%·상위 {OUTLIER_PERCENTILE_HIGH}% "
            f"유효 구간 [{low_q:.3f}, {high_q:.3f}]"
        )
        print(f"이상치 제거: {int((~keep).sum())}건")

    df = df.loc[keep].copy()
    train_mask = train_mask.loc[keep]
    val_mask = val_mask.loc[keep]
    test_mask = test_mask.loc[keep]

    for col in ("apartmentName", "dong"):
        counts = df.loc[train_mask, col].value_counts()
        keep_vals = set(counts[counts >= MIN_SAMPLES_FOR_CATEGORY].index.tolist())
        df[col] = df[col].where(df[col].isin(keep_vals), "기타")

    return {
        "df": df,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


def build_index_map(values: pd.Series) -> dict:
    mapping = {"기타": UNKNOWN_IDX}
    names = sorted({str(v) for v in values.unique() if str(v) != "기타"})
    for i, name in enumerate(names, start=1):
        mapping[name] = i
    return mapping


def to_index(series: pd.Series, mapping: dict) -> np.ndarray:
    return series.astype(str).map(lambda x: mapping.get(x, UNKNOWN_IDX)).to_numpy(dtype=np.int64)


def encode_mlp_inputs(df: pd.DataFrame, mapping: dict, fit_scaler: bool = False):
    numeric_features = mapping["numeric_features"]
    apt_idx = to_index(df["apartmentName"], mapping["apt_to_idx"])
    dong_idx = to_index(df["dong"], mapping["dong_to_idx"])
    numeric = df[numeric_features].to_numpy(dtype=np.float32)
    scaler = mapping["scaler"]
    if fit_scaler:
        numeric = scaler.fit_transform(numeric).astype(np.float32)
    else:
        numeric = scaler.transform(numeric).astype(np.float32)
    return apt_idx, dong_idx, numeric


def evaluate_price(predicted_ratio, df_rows: pd.DataFrame) -> dict:
    sale_price = df_rows["salePrice"].to_numpy(dtype=float)
    predicted = predicted_ratio * sale_price
    actual = df_rows["jeonsePrice"].to_numpy(dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = price_mape(actual, predicted)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "predicted": predicted,
        "actual": actual,
    }


def print_eval_report(name: str, metrics: dict, df_test: pd.DataFrame) -> None:
    print(f"\n[{name}] Test MAE:", metrics["mae"])
    print(f"[{name}] Test RMSE:", metrics["rmse"])
    print(f"[{name}] Test MAPE:", metrics["mape"])

    actual = metrics["actual"]
    predicted = metrics["predicted"]
    bands = [(0, 50000, "0~5억"), (50000, 10**9, "5억 이상")]
    print(f"\n[{name}] 가격대별 MAPE (만원 기준, Test):")
    for low, high, label in bands:
        mask = (actual >= low) & (actual < high)
        if mask.sum() == 0:
            continue
        band_mape = price_mape(actual[mask], predicted[mask])
        print(f"  {label}: MAPE={band_mape:.2f}% (n={int(mask.sum())})")

    nonzero = actual != 0
    ratio_errors = (np.abs(actual[nonzero] - predicted[nonzero]) / actual[nonzero]) * 100
    print(f"\n[{name}] 오차 분포 (Test)")
    print("median MAPE:", np.median(ratio_errors))
    print("p75:", np.percentile(ratio_errors, 75))
    print("p90:", np.percentile(ratio_errors, 90))
    print("p95:", np.percentile(ratio_errors, 95))
    print("max:", np.max(ratio_errors))

    df_test_out = df_test.loc[nonzero].copy()
    df_test_out["predicted"] = predicted[nonzero]
    df_test_out["error_pct"] = ratio_errors
    worst = df_test_out.sort_values("error_pct", ascending=False).head(20).reset_index(drop=True)
    print(f"\n[{name}] Worst predictions")
    cols = ["salePrice", "jeonsePrice", "predicted", "error_pct"]
    if "jeonseYm" in worst.columns:
        cols = ["jeonseYm"] + cols
    print(worst[cols])


def _predict_ratio(model, apt_idx, dong_idx, numeric, device, batch_size=1024):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(apt_idx), batch_size):
            end = start + batch_size
            apt_t = torch.from_numpy(apt_idx[start:end]).to(device)
            dong_t = torch.from_numpy(dong_idx[start:end]).to(device)
            num_t = torch.from_numpy(numeric[start:end]).to(device)
            preds.append(model(apt_t, dong_t, num_t).cpu().numpy())
    return np.concatenate(preds, axis=0)


def run_training(verbose: bool = True, embedding_dim: int = EMBEDDING_DIM):
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"device: {device}")

    split = load_split_frame(verbose=verbose)
    df = split["df"]
    train_mask = split["train_mask"]
    val_mask = split["val_mask"]
    test_mask = split["test_mask"]

    numeric_features = numeric_feature_list(df)
    mapping = {
        "apt_to_idx": build_index_map(df.loc[train_mask, "apartmentName"]),
        "dong_to_idx": build_index_map(df.loc[train_mask, "dong"]),
        "numeric_features": numeric_features,
        "scaler": StandardScaler(),
        "embedding_dim": embedding_dim,
    }
    mapping["n_apt"] = max(mapping["apt_to_idx"].values()) + 1
    mapping["n_dong"] = max(mapping["dong_to_idx"].values()) + 1

    apt_tr, dong_tr, num_tr = encode_mlp_inputs(df.loc[train_mask], mapping, fit_scaler=True)
    apt_va, dong_va, num_va = encode_mlp_inputs(df.loc[val_mask], mapping, fit_scaler=False)
    apt_te, dong_te, num_te = encode_mlp_inputs(df.loc[test_mask], mapping, fit_scaler=False)

    y_tr = df.loc[train_mask, "jeonseRatio"].to_numpy(dtype=np.float32)
    y_va = df.loc[val_mask, "jeonseRatio"].to_numpy(dtype=np.float32)
    train_years = df.loc[train_mask, "year"].astype(int)
    sample_weight = np.array(
        [EVAL_YEAR_WEIGHTS.get(int(y), DEFAULT_YEAR_WEIGHT) for y in train_years],
        dtype=np.float32,
    )

    if verbose:
        print("연도별 학습 가중치:", EVAL_YEAR_WEIGHTS, f"(default={DEFAULT_YEAR_WEIGHT})")
        print(
            f"학습 행 수(필터 후): train={int(train_mask.sum())} "
            f"val={int(val_mask.sum())} test={int(test_mask.sum())}"
        )
        print(
            f"임베딩: apt={mapping['n_apt']} dong={mapping['n_dong']} "
            f"dim={embedding_dim} numeric={len(numeric_features)}"
        )

    train_ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(apt_tr)),
        torch.from_numpy(np.ascontiguousarray(dong_tr)),
        torch.from_numpy(np.ascontiguousarray(num_tr)),
        torch.from_numpy(np.ascontiguousarray(y_tr)),
        torch.from_numpy(np.ascontiguousarray(sample_weight)),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = JeonseEmbeddingMLP(
        n_apt=mapping["n_apt"],
        n_dong=mapping["n_dong"],
        n_numeric=len(numeric_features),
        embedding_dim=embedding_dim,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_none = nn.MSELoss(reduction="none")

    if verbose:
        print("Embedding MLP 학습 시작...")
    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=_progress_logger,
        args=(PROGRESS_LOG_INTERVAL_SEC, stop_event),
        daemon=True,
    )
    progress_thread.start()

    best_val = float("inf")
    best_state = None
    wait = 0
    best_epoch = 0

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            running = 0.0
            n_seen = 0
            for apt_b, dong_b, num_b, y_b, w_b in train_loader:
                apt_b = apt_b.to(device)
                dong_b = dong_b.to(device)
                num_b = num_b.to(device)
                y_b = y_b.to(device)
                w_b = w_b.to(device)
                pred = model(apt_b, dong_b, num_b)
                loss = (mse_none(pred, y_b) * w_b).sum() / w_b.sum().clamp_min(1e-8)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * len(y_b)
                n_seen += len(y_b)

            model.eval()
            with torch.no_grad():
                val_pred = model(
                    torch.from_numpy(apt_va).to(device),
                    torch.from_numpy(dong_va).to(device),
                    torch.from_numpy(num_va).to(device),
                )
                val_loss = float(
                    nn.functional.mse_loss(
                        val_pred,
                        torch.from_numpy(y_va).to(device),
                    ).item()
                )

            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
                best_epoch = epoch
            else:
                wait += 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"  epoch {epoch:3d}  train_mse={running / max(n_seen, 1):.6f}  "
                    f"val_mse={val_loss:.6f}  best={best_val:.6f}  wait={wait}"
                )

            if wait >= PATIENCE:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break
    finally:
        stop_event.set()
        progress_thread.join(timeout=PROGRESS_LOG_INTERVAL_SEC + 1)

    if best_state is None:
        raise RuntimeError("학습이 완료되지 않았습니다.")
    model.load_state_dict(best_state)
    model.to(device)

    if verbose:
        print("학습 완료.")

    pred_va = _predict_ratio(model, apt_va, dong_va, num_va, device)
    pred_te = _predict_ratio(model, apt_te, dong_te, num_te, device)
    val_metrics = evaluate_price(pred_va, df.loc[val_mask])
    test_metrics = evaluate_price(pred_te, df.loc[test_mask])

    if verbose:
        print("\nVal MAE:", val_metrics["mae"])
        print("Val RMSE:", val_metrics["rmse"])
        print("Val MAPE:", val_metrics["mape"])
        print_eval_report("EmbeddingMLP", test_metrics, df.loc[test_mask])

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_apt": mapping["n_apt"],
            "n_dong": mapping["n_dong"],
            "n_numeric": len(numeric_features),
            "embedding_dim": embedding_dim,
            "dropout": DROPOUT,
        },
        MODEL_PATH,
    )
    joblib.dump(mapping, MAPPING_PATH)
    if verbose:
        print(f"\n모델 저장 경로: {MODEL_PATH}")
        print(f"매핑 저장 경로: {MAPPING_PATH}")

    return {
        "mape": test_metrics["mape"],
        "val_mape": val_metrics["mape"],
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "model": model,
        "mapping": mapping,
        "test_metrics": test_metrics,
    }


def main() -> None:
    run_training(verbose=True, embedding_dim=EMBEDDING_DIM)


if __name__ == "__main__":
    main()
