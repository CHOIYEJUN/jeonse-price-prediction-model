"""단지명·동을 임베딩으로 넣는 MLP로 전세가율을 학습한다.

분할·이상치·가중치·숫자 피처는 train_xgboost.py의 eval 설정과 같다.
다른 점은 범주를 원-핫이 아니라 정수 인덱스 → 임베딩으로 넣는 것이다.
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
# train_xgboost.py eval 과 같은 숫자
# ---------------------------------------------------------------------------
RANDOM_SEED = 42  # 난수를 고정해 다시 돌려도 비슷하게
MIN_SAMPLES_FOR_CATEGORY = 15  # 이보다 적은 단지·동은 "기타"
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

# 이상치 제거
OUTLIER_PERCENTILE_LOW = 10 # 학습 분포 하위 이 %를 자른다
OUTLIER_PERCENTILE_HIGH = 5 # 학습 분포 상위 이 %를 자른다

UNKNOWN_IDX = 0  # 학습에 없던 이름, "기타"가 쓰는 인덱스
EMBEDDING_DIM = 8  # 단지·동을 몇 칸 벡터로 표현할지
DROPOUT = 0.25  # 학습 때 뉴런을 끄는 비율
BATCH_SIZE = 256  # 한 번에 모형에 넣는 행 수
LEARNING_RATE = 1e-3  # Adam이 가중치를 한 번에 얼마나 옮길지
MAX_EPOCHS = 200  # 표를 최대 몇 바퀴 볼지
PATIENCE = 50  # 검증 손실이 이 횟수 동안 안 줄면 멈춘다
PROGRESS_LOG_INTERVAL_SEC = 10.0


def set_seed(seed: int = RANDOM_SEED) -> None:
    """파이썬·NumPy·PyTorch 난수를 같은 씨앗으로 맞춘다.

    seed: 정수. 같으면 드롭아웃·셔플 순서가 재현된다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _progress_logger(interval_sec: float, stop_event: threading.Event) -> None:
    """학습이 돌아가는 동안 몇 초마다 경과를 출력한다.

    interval_sec: 출력 간격(초)
    stop_event: 학습이 끝나면 set()되어 이 루프가 끝난다
    """
    start = time.perf_counter()
    while not stop_event.is_set():
        stop_event.wait(interval_sec)
        if stop_event.is_set():
            break
        elapsed = int(time.perf_counter() - start)
        print(f"  진행 중: 학습 {elapsed}초 경과")


def price_mape(actual, predicted):
    """보증금 기준 평균절대백분율오차(MAPE, %)를 계산한다.

    actual: 실제 전세보증금
    predicted: 추정 전세보증금
    반환: 상대오차 평균 × 100. 실제값이 모두 0이면 nan
    """
    nonzero = actual != 0
    if nonzero.sum() == 0:
        return float("nan")
    a = actual[nonzero]
    p = predicted[nonzero]
    return float((np.abs(a - p) / a).mean() * 100)


def numeric_feature_list(df: pd.DataFrame) -> list:
    """표에 실제로 있는 숫자 피처 이름만 골라 리스트로 돌려준다.

    df: 학습용 표
    반환: ["area", "floor", ...] 같은 열 이름 목록
    """
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
    """CSV를 읽고, 이상치를 자른 뒤 학습/검증/평가 마스크를 만든다.

    verbose: True면 분할·필터 로그를 출력한다
    반환: {df, train_mask, val_mask, test_mask}
        df: 필터 후 표
        *_mask: 각 행이 어느 구간인지 True/False
    """
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

    # 자르는 선은 학습 전세가율 분포로만 계산한다
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

    # 학습에서 드문 단지·동은 "기타" (인덱스 0)
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
    """이름 문자열을 정수 인덱스로 바꾸는 사전을 만든다.

    values: 학습 구간의 단지명 또는 동 시리즈
    반환: {"기타": 0, "정자동": 1, ...}
    """
    mapping = {"기타": UNKNOWN_IDX}
    names = sorted({str(v) for v in values.unique() if str(v) != "기타"})
    for i, name in enumerate(names, start=1):  # 0은 기타가 쓰므로 1부터
        mapping[name] = i
    return mapping


def to_index(series: pd.Series, mapping: dict) -> np.ndarray:
    """이름 열을 인덱스 배열로 바꾼다. 사전에 없으면 0(기타).

    series: apartmentName 또는 dong 열
    mapping: build_index_map 결과
    반환: int64 배열
    """
    return series.astype(str).map(lambda x: mapping.get(x, UNKNOWN_IDX)).to_numpy(dtype=np.int64)


def encode_mlp_inputs(df: pd.DataFrame, mapping: dict, fit_scaler: bool = False):
    """한 구간 표를 모형이 받는 세 배열로 바꾼다.

    df: 그 구간의 행만 있는 표
    mapping: apt_to_idx, dong_to_idx, numeric_features, scaler가 들어 있는 사전
    fit_scaler: True면 이 표로 평균·표준편차를 새로 맞춘다 (학습만 True)
    반환: (단지 인덱스, 동 인덱스, 표준화된 숫자 배열)
    """
    numeric_features = mapping["numeric_features"]
    apt_idx = to_index(df["apartmentName"], mapping["apt_to_idx"])
    dong_idx = to_index(df["dong"], mapping["dong_to_idx"])
    numeric = df[numeric_features].to_numpy(dtype=np.float32)
    scaler = mapping["scaler"]  # StandardScaler: (값-평균)/표준편차
    if fit_scaler:
        numeric = scaler.fit_transform(numeric).astype(np.float32)
    else:
        numeric = scaler.transform(numeric).astype(np.float32)
    return apt_idx, dong_idx, numeric


def evaluate_price(predicted_ratio, df_rows: pd.DataFrame) -> dict:
    """예측 전세가율에 매매가를 곱해 보증금 오차를 계산한다.

    predicted_ratio: 모형이 낸 비율 배열
    df_rows: 같은 행의 원본 (salePrice, jeonsePrice)
    반환: mae, rmse, mape, predicted, actual
    """
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
    """평가 구간의 MAE·RMSE·MAPE, 가격대별 오차, 최악 20건을 출력한다.

    name: 로그에 찍을 모형 이름
    metrics: evaluate_price 결과
    df_test: 평가 행의 원본 표
    """
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
    """평가용으로 전세가율을 배치 단위로 예측한다.

    model: JeonseEmbeddingMLP
    apt_idx, dong_idx, numeric: encode_mlp_inputs 결과
    device: cpu 또는 cuda
    batch_size: 한 번에 넣을 행 수. 메모리가 부족하면 줄인다
    반환: 예측 전세가율 1차원 배열
    """
    model.eval()
    preds = []
    with torch.no_grad():  # 기울기 그래프를 만들지 않음
        for start in range(0, len(apt_idx), batch_size):
            end = start + batch_size
            apt_t = torch.from_numpy(apt_idx[start:end]).to(device)
            dong_t = torch.from_numpy(dong_idx[start:end]).to(device)
            num_t = torch.from_numpy(numeric[start:end]).to(device)
            preds.append(model(apt_t, dong_t, num_t).cpu().numpy())
    return np.concatenate(preds, axis=0)


def run_training(verbose: bool = True, embedding_dim: int = EMBEDDING_DIM):
    """분할·인코딩·학습·조기종료·저장까지 수행한다.

    verbose: 로그 출력 여부
    embedding_dim: 단지·동 벡터 칸 수
    반환: mape, val_mape, model, mapping 등이 있는 사전
    """
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
    # mapping: 나중에 예측할 때도 같은 인덱스·스케일러를 쓰기 위해 저장한다
    mapping = {
        "apt_to_idx": build_index_map(df.loc[train_mask, "apartmentName"]),
        "dong_to_idx": build_index_map(df.loc[train_mask, "dong"]),
        "numeric_features": numeric_features,
        "scaler": StandardScaler(),
        "embedding_dim": embedding_dim,
    }
    mapping["n_apt"] = max(mapping["apt_to_idx"].values()) + 1  # 임베딩 표의 행 수
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

    # TensorDataset: 같은 위치의 텐서를 한 묶음으로 꺼내게 한다
    train_ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(apt_tr)),
        torch.from_numpy(np.ascontiguousarray(dong_tr)),
        torch.from_numpy(np.ascontiguousarray(num_tr)),
        torch.from_numpy(np.ascontiguousarray(y_tr)),
        torch.from_numpy(np.ascontiguousarray(sample_weight)),
    )
    # shuffle=True: 매 epoch마다 행 순서를 섞어 과적합을 줄인다
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = JeonseEmbeddingMLP(
        n_apt=mapping["n_apt"],
        n_dong=mapping["n_dong"],
        n_numeric=len(numeric_features),
        embedding_dim=embedding_dim,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_none = nn.MSELoss(reduction="none")  # 행마다 제곱오차. 가중치를 곱하기 위함

    if verbose:
        print("Embedding MLP 학습 시작...")
    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=_progress_logger,
        args=(PROGRESS_LOG_INTERVAL_SEC, stop_event),
        daemon=True,
    )
    progress_thread.start()

    best_val = float("inf")  # 지금까지 가장 좋은 검증 손실
    best_state = None  # 그때의 가중치 복사본
    wait = 0  # 개선이 없었던 epoch 수
    best_epoch = 0

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()  # Dropout·BatchNorm을 학습 모드로
            running = 0.0  # 이 epoch 손실의 가중 합
            n_seen = 0
            for apt_b, dong_b, num_b, y_b, w_b in train_loader:
                apt_b = apt_b.to(device)
                dong_b = dong_b.to(device)
                num_b = num_b.to(device)
                y_b = y_b.to(device)
                w_b = w_b.to(device)
                pred = model(apt_b, dong_b, num_b)
                # 행별 MSE × 연도 가중치의 가중 평균
                loss = (mse_none(pred, y_b) * w_b).sum() / w_b.sum().clamp_min(1e-8)
                optimizer.zero_grad()  # 이전 배치 기울기를 지운다
                loss.backward()  # 기울기 계산
                optimizer.step()  # 가중치 한 걸음 이동
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
                # cpu로 복사해 두어야 이후 device가 바뀌어도 안전하다
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
    model.load_state_dict(best_state)  # 가장 좋았던 가중치로 되돌린다
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
    """기본 임베딩 크기로 학습을 한 번 돌린다."""
    run_training(verbose=True, embedding_dim=EMBEDDING_DIM)


if __name__ == "__main__":
    main()
