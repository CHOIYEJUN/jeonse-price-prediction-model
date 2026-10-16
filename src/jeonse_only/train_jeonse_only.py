"""전세 거래만으로 다음 보증금을 학습하고 세 모형을 비교한다.

비교 대상
    Persistence: 예측 = 직전 전세보증금
    Lag Embedding MLP: 직전 가격·평균·경과 일수를 피처로 넣은 다층퍼셉트론
    Embedding LSTM: 최근 최대 8건의 거래 순서를 LSTM이 요약

타깃은 log(이번 보증금)이다. 지표는 exp로 되돌린 만원 단위로 계산한다.
분할은 기존 전세가율 모형과 같이 jeonseYm 기준 시점 홀드아웃이다.
매매가 열은 읽지 않는다.
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

from jeonse_only_model import JeonseHistoryLSTM, LagEmbeddingMLP


# ---------------------------------------------------------------------------
# 경로·분할·학습 설정
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "jeonse_only_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MLP_PATH = os.path.join(MODEL_DIR, "jeonse_only_mlp.pt")
LSTM_PATH = os.path.join(MODEL_DIR, "jeonse_only_lstm.pt")
MAPPING_PATH = os.path.join(MODEL_DIR, "jeonse_only_mapping.pkl")

RANDOM_SEED = 42
MIN_SAMPLES_FOR_CATEGORY = 15  # 이보다 적은 단지·동은 임베딩을 공유하는 "기타"
TRAIN_YM_MAX = 202512
VAL_YM_MIN = 202601
VAL_YM_MAX = 202606
TEST_YM_MIN = 202607

UNKNOWN_IDX = 0
EMBEDDING_DIM = 8
DROPOUT = 0.25
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # Adam의 L2 규제. 가중치가 커져 단지 이름을 외우는 것을 줄인다
MAX_EPOCHS = 80
PATIENCE = 15  # 검증 Huber가 이 횟수만큼 개선되지 않으면 멈춘다
HUBER_DELTA = 1.0  # 로그 오차가 이 값을 넘으면 제곱 대신 선형으로 벌점을 준다
GRAD_CLIP = 1.0  # LSTM 기울기 폭주 방지. 이 길이로 자른다
MAX_SEQ_LEN = 8
LSTM_HIDDEN = 64
N_SEQ = 4  # log 보증금, 면적, 층, 다음 거래까지 일수

# 가격 계열 래그는 로그로 바꿔 타깃과 같은 스케일로 맞춘 뒤 표준화한다.
MLP_FEATURES = [
    "area",
    "floor",
    "buildingAge",
    "log_last_jeonse_price",
    "log_last_3_mean_jeonse_price",
    "days_since_last",
    "log_last_price_per_m2",
]
STATIC_FEATURES = ["area", "floor", "buildingAge"]
SEQ_FEATURES = ["log_price", "area", "floor", "gap_days"]


def set_seed(seed: int = RANDOM_SEED) -> None:
    """파이썬·NumPy·PyTorch 난수를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    """CUDA가 있으면 쓰고, 없으면 CPU를 쓴다.

    pack_padded_sequence는 MPS에서 환경에 따라 실패하므로 Mac GPU는 쓰지 않는다.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_frame() -> pd.DataFrame:
    """전세 전용 CSV를 읽고 날짜·숫자를 학습에 맞게 정리한다."""
    df = pd.read_csv(DATASET_PATH)
    df["jeonseDate"] = pd.to_datetime(df["jeonseDate"])
    df["jeonseYm"] = pd.to_numeric(df["jeonseYm"], errors="coerce").astype(int)
    for col in (
        "area",
        "floor",
        "buildingAge",
        "jeonsePrice",
        "last_jeonse_price",
        "last_3_mean_jeonse_price",
        "days_since_last",
        "last_price_per_m2",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["apartmentName"] = df["apartmentName"].astype(str)
    df["dong"] = df["dong"].astype(str)
    df = df.sort_values(
        ["apt_id", "area_group", "jeonseDate", "jeonsePrice"]
    ).reset_index(drop=True)
    return df


def build_history_sequences(df: pd.DataFrame, max_len: int = MAX_SEQ_LEN):
    """단지·면적대마다 이번 행 이전의 최근 거래를 오른쪽 패딩 시퀀스로 만든다.

    각 스텝의 일수는 '그 거래부터 다음 거래까지'이다.
    그래서 시퀀스 마지막 스텝의 일수는 직전 거래가 이번 계약보다 얼마나 오래됐는지와 같다.
    이번 행의 보증금은 넣지 않는다.

    반환: seq (행, max_len, 4), lengths (행,). 첫 거래는 길이 0.
    """
    n = len(df)
    seq = np.zeros((n, max_len, N_SEQ), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    dates = df["jeonseDate"].to_numpy()
    prices = df["jeonsePrice"].to_numpy(dtype=np.float64)
    areas = df["area"].to_numpy(dtype=np.float64)
    floors = df["floor"].to_numpy(dtype=np.float64)

    grouped = df.groupby(["apt_id", "area_group"], sort=False).indices
    for positions in grouped.values():
        idx = np.asarray(list(positions), dtype=np.int64)
        order = np.argsort(dates[idx], kind="mergesort")
        idx = idx[order]
        g_dates = dates[idx]
        g_price = prices[idx]
        g_area = areas[idx]
        g_floor = floors[idx]
        # gap_to_next[i] = 거래 i가 그 다음 거래까지 걸린 일수. 그룹의 마지막 거래는 쓰지 않는다.
        gap_to_next = np.zeros(len(idx), dtype=np.float64)
        if len(idx) > 1:
            gap_to_next[:-1] = (g_dates[1:] - g_dates[:-1]) / np.timedelta64(1, "D")

        for i in range(1, len(idx)):
            start = max(0, i - max_len)
            steps = i - start
            block = np.column_stack(
                [
                    np.log(g_price[start:i]),
                    g_area[start:i],
                    g_floor[start:i],
                    gap_to_next[start:i],
                ]
            ).astype(np.float32)
            row = int(idx[i])
            seq[row, :steps] = block
            lengths[row] = steps
    return seq, lengths


def assert_sequence_matches_lags(df: pd.DataFrame, seq: np.ndarray, lengths: np.ndarray, mask: np.ndarray) -> None:
    """시퀀스 마지막 스텝이 직전 보증금·경과 일수와 같은지 확인한다.

    여기서 어긋나면 이번 보증금이 과거 칸에 들어간 것이므로 학습하면 안 된다.
    """
    rows = np.flatnonzero(mask)
    if len(rows) == 0:
        raise RuntimeError("직전 거래가 있는 행이 없습니다.")
    if np.any(lengths[rows] < 1):
        raise RuntimeError("직전 거래가 있는데 시퀀스 길이가 0인 행이 있습니다.")
    last = lengths[rows] - 1
    log_price = seq[rows, last, 0]
    gap = seq[rows, last, 3]
    expected_log = np.log(df.loc[rows, "last_jeonse_price"].to_numpy(dtype=np.float64))
    expected_gap = df.loc[rows, "days_since_last"].to_numpy(dtype=np.float64)
    if not np.allclose(log_price, expected_log, rtol=1e-4, atol=1e-4):
        raise RuntimeError("시퀀스의 직전 로그 보증금이 last_jeonse_price와 다릅니다.")
    if not np.allclose(gap, expected_gap, rtol=1e-4, atol=1e-3):
        raise RuntimeError("시퀀스 마지막 일수가 days_since_last와 다릅니다.")


def time_masks(df: pd.DataFrame, supervised: np.ndarray) -> dict:
    """직전 거래가 있는 행만 학습·검증·평가로 나눈다."""
    ym = df["jeonseYm"].to_numpy()
    return {
        "train": supervised & (ym <= TRAIN_YM_MAX),
        "val": supervised & (ym >= VAL_YM_MIN) & (ym <= VAL_YM_MAX),
        "test": supervised & (ym >= TEST_YM_MIN),
    }


def collapse_rare(names: pd.Series, train_mask: np.ndarray) -> pd.Series:
    """학습 구간에 드문 이름을 '기타'로 바꾼다. 검증·평가에도 같은 규칙을 적용한다."""
    counts = names[train_mask].value_counts()
    keep = set(counts[counts >= MIN_SAMPLES_FOR_CATEGORY].index.tolist())
    return names.where(names.isin(keep), "기타")


def build_index_map(names: pd.Series) -> dict:
    """이름 → 정수. 0은 기타."""
    mapping = {"기타": UNKNOWN_IDX}
    unique = sorted({str(v) for v in names.unique() if str(v) != "기타"})
    for i, name in enumerate(unique, start=1):
        mapping[name] = i
    return mapping


def to_index(names: pd.Series, mapping: dict) -> np.ndarray:
    """사전에 없는 이름은 0(기타)으로 둔다."""
    return names.map(lambda x: mapping.get(str(x), UNKNOWN_IDX)).to_numpy(dtype=np.int64)


def add_log_lag_columns(df: pd.DataFrame, supervised: np.ndarray) -> pd.DataFrame:
    """MLP가 볼 가격 래그를 자연로그로 바꾼 열을 추가한다."""
    out = df.copy()
    out["log_last_jeonse_price"] = np.nan
    out["log_last_3_mean_jeonse_price"] = np.nan
    out["log_last_price_per_m2"] = np.nan
    rows = np.flatnonzero(supervised)
    out.loc[rows, "log_last_jeonse_price"] = np.log(out.loc[rows, "last_jeonse_price"].to_numpy())
    out.loc[rows, "log_last_3_mean_jeonse_price"] = np.log(
        out.loc[rows, "last_3_mean_jeonse_price"].to_numpy()
    )
    out.loc[rows, "log_last_price_per_m2"] = np.log(out.loc[rows, "last_price_per_m2"].to_numpy())
    return out


def sequence_mean_std(seq: np.ndarray, lengths: np.ndarray):
    """패딩을 제외한 스텝만으로 평균·표준편차를 구한다. 학습 구간에서만 호출한다."""
    steps = np.arange(seq.shape[1])[None, :]
    valid = seq[steps < lengths[:, None]]
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, np.float32(1.0), std)
    return mean, std


def apply_sequence_scale(seq: np.ndarray, lengths: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """실제 스텝만 표준화한다. 패딩은 0으로 둔다."""
    out = np.zeros_like(seq)
    steps = np.arange(seq.shape[1])[None, :]
    mask = steps < lengths[:, None]
    out[mask] = (seq[mask] - mean) / std
    return out


def price_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """만원 단위 보증금의 MAE, RMSE, MAPE와 오차 분위를 계산한다."""
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    error_pct = np.abs(actual - predicted) / actual * 100.0
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": float(error_pct.mean()),
        "median_mape": float(np.median(error_pct)),
        "p75": float(np.percentile(error_pct, 75)),
        "p90": float(np.percentile(error_pct, 90)),
        "p95": float(np.percentile(error_pct, 95)),
        "max": float(error_pct.max()),
        "predicted": predicted,
        "actual": actual,
        "error_pct": error_pct,
    }


def print_metrics(name: str, metrics: dict, df_rows: pd.DataFrame) -> None:
    """한 모형의 평가 구간 지표와 오차가 큰 거래 5건을 출력한다."""
    print(f"\n[{name}]")
    print(f"  MAE {metrics['mae']:.1f}  RMSE {metrics['rmse']:.1f}")
    print(
        f"  MAPE {metrics['mape']:.2f}%  median {metrics['median_mape']:.2f}%  "
        f"p75 {metrics['p75']:.2f}%  p90 {metrics['p90']:.2f}%  "
        f"p95 {metrics['p95']:.2f}%  max {metrics['max']:.2f}%"
    )
    view = df_rows.copy()
    view["predicted"] = metrics["predicted"]
    view["error_pct"] = metrics["error_pct"]
    worst = view.sort_values("error_pct", ascending=False).head(5)
    cols = ["jeonseYm", "dong", "apartmentName", "jeonsePrice", "predicted", "error_pct"]
    print(worst[cols].to_string(index=False))


def _progress_logger(name: str, interval_sec: float, stop_event: threading.Event) -> None:
    """학습이 길 때 몇 초마다 경과 시간을 찍는다."""
    start = time.perf_counter()
    while not stop_event.is_set():
        stop_event.wait(interval_sec)
        if stop_event.is_set():
            break
        elapsed = int(time.perf_counter() - start)
        print(f"  [{name}] 진행 중: {elapsed}초 경과")


def fit_model(model, train_loader, unpack_batch, val_loss_fn, device, clip_grad, name: str):
    """Huber 손실로 학습하고, 검증 손실이 가장 낮았던 가중치를 되돌린다.

    unpack_batch(model, batch) -> 예측 (배치,)
    val_loss_fn(model) -> 검증 Huber (float)
    clip_grad: None이면 기울기를 자르지 않는다. LSTM만 값을 넘긴다.
    """
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    stop_event = threading.Event()
    progress = threading.Thread(
        target=_progress_logger,
        args=(name, 10.0, stop_event),
        daemon=True,
    )
    progress.start()
    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            running = 0.0
            n_seen = 0
            for batch in train_loader:
                batch = tuple(t.to(device) for t in batch)
                pred = unpack_batch(model, batch)
                y = batch[-1]
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                running += float(loss.item()) * len(y)
                n_seen += len(y)

            model.eval()
            val_loss = val_loss_fn(model)
            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                wait = 0
            else:
                wait += 1

            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"  [{name}] epoch {epoch:3d}  "
                    f"train_huber={running / max(n_seen, 1):.5f}  "
                    f"val_huber={val_loss:.5f}  best={best_val:.5f}  wait={wait}"
                )
            if wait >= PATIENCE:
                print(f"  [{name}] early stopping at epoch {epoch} (best {best_epoch})")
                break
    finally:
        stop_event.set()
        progress.join(timeout=2)

    if best_state is None:
        raise RuntimeError(f"{name} 학습이 끝나지 않았습니다.")
    model.load_state_dict(best_state)
    model.to(device)
    return best_epoch, best_val


def predict_mlp(model, apt, dong, numeric, device, batch_size=1024) -> np.ndarray:
    """MLP 로그 보증금 예측을 배치로 모아 1차원 배열로 돌려준다."""
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(apt), batch_size):
            end = start + batch_size
            pred = model(
                torch.from_numpy(apt[start:end]).to(device),
                torch.from_numpy(dong[start:end]).to(device),
                torch.from_numpy(numeric[start:end]).to(device),
            )
            pieces.append(pred.cpu().numpy())
    return np.concatenate(pieces)


def predict_lstm(model, apt, dong, static, seq, lengths, device, batch_size=1024) -> np.ndarray:
    """LSTM 로그 보증금 예측을 배치로 모아 1차원 배열로 돌려준다."""
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(apt), batch_size):
            end = start + batch_size
            pred = model(
                torch.from_numpy(apt[start:end]).to(device),
                torch.from_numpy(dong[start:end]).to(device),
                torch.from_numpy(static[start:end]).to(device),
                torch.from_numpy(seq[start:end]).to(device),
                torch.from_numpy(lengths[start:end]).to(device),
            )
            pieces.append(pred.cpu().numpy())
    return np.concatenate(pieces)


def huber_from_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """이미 계산된 로그 예측의 Huber 손실."""
    return float(
        nn.functional.huber_loss(
            torch.from_numpy(pred.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
            delta=HUBER_DELTA,
        ).item()
    )


def run_training(verbose: bool = True) -> dict:
    """데이터 분할, 두 신경망 학습, 지속 예측과의 비교까지 수행한다."""
    set_seed(RANDOM_SEED)
    device = pick_device()
    if verbose:
        print(f"device: {device}")

    df = load_frame()
    if verbose:
        print("시퀀스 구성 중...")
    seq, lengths = build_history_sequences(df, MAX_SEQ_LEN)
    supervised = (
        df["last_jeonse_price"].notna()
        & df["last_3_mean_jeonse_price"].notna()
        & df["days_since_last"].notna()
        & df["last_price_per_m2"].notna()
        & (df["last_jeonse_price"] > 0)
        & (df["last_price_per_m2"] > 0)
    ).to_numpy()
    assert_sequence_matches_lags(df, seq, lengths, supervised)

    masks = time_masks(df, supervised)
    for key in ("train", "val", "test"):
        if int(masks[key].sum()) == 0:
            raise RuntimeError(f"{key} 구간 행이 없습니다. jeonseYm 분할을 확인하세요.")
    if verbose:
        print(
            "시점 분할 (직전 거래 있는 행): "
            f"train≤{TRAIN_YM_MAX} {int(masks['train'].sum())} / "
            f"val {VAL_YM_MIN}–{VAL_YM_MAX} {int(masks['val'].sum())} / "
            f"test≥{TEST_YM_MIN} {int(masks['test'].sum())}"
        )

    df = add_log_lag_columns(df, supervised)
    apt_names = collapse_rare(df["apartmentName"], masks["train"])
    dong_names = collapse_rare(df["dong"], masks["train"])
    apt_map = build_index_map(apt_names[masks["train"]])
    dong_map = build_index_map(dong_names[masks["train"]])
    apt_idx = to_index(apt_names, apt_map)
    dong_idx = to_index(dong_names, dong_map)
    n_apt = max(apt_map.values()) + 1
    n_dong = max(dong_map.values()) + 1

    mlp_scaler = StandardScaler()
    mlp_numeric = df[MLP_FEATURES].to_numpy(dtype=np.float32)
    mlp_numeric[masks["train"]] = mlp_scaler.fit_transform(mlp_numeric[masks["train"]]).astype(np.float32)
    mlp_numeric[masks["val"]] = mlp_scaler.transform(mlp_numeric[masks["val"]]).astype(np.float32)
    mlp_numeric[masks["test"]] = mlp_scaler.transform(mlp_numeric[masks["test"]]).astype(np.float32)

    static_scaler = StandardScaler()
    static = df[STATIC_FEATURES].to_numpy(dtype=np.float32)
    static[masks["train"]] = static_scaler.fit_transform(static[masks["train"]]).astype(np.float32)
    static[masks["val"]] = static_scaler.transform(static[masks["val"]]).astype(np.float32)
    static[masks["test"]] = static_scaler.transform(static[masks["test"]]).astype(np.float32)

    seq_mean, seq_std = sequence_mean_std(seq[masks["train"]], lengths[masks["train"]])
    seq_scaled = apply_sequence_scale(seq, lengths, seq_mean, seq_std)

    y_log = np.log(df["jeonsePrice"].to_numpy(dtype=np.float64)).astype(np.float32)

    def take(mask):
        return {
            "apt": np.ascontiguousarray(apt_idx[mask]),
            "dong": np.ascontiguousarray(dong_idx[mask]),
            "mlp": np.ascontiguousarray(mlp_numeric[mask]),
            "static": np.ascontiguousarray(static[mask]),
            "seq": np.ascontiguousarray(seq_scaled[mask]),
            "lengths": np.ascontiguousarray(lengths[mask]),
            "y": np.ascontiguousarray(y_log[mask]),
            "frame": df.loc[mask].reset_index(drop=True),
        }

    train = take(masks["train"])
    val = take(masks["val"])
    test = take(masks["test"])

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    mlp = LagEmbeddingMLP(
        n_apt=n_apt,
        n_dong=n_dong,
        n_numeric=len(MLP_FEATURES),
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT,
    )
    mlp_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["apt"]),
            torch.from_numpy(train["dong"]),
            torch.from_numpy(train["mlp"]),
            torch.from_numpy(train["y"]),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )

    def mlp_forward(model, batch):
        apt_b, dong_b, num_b, _y = batch
        return model(apt_b, dong_b, num_b)

    def mlp_val_loss(model):
        pred = predict_mlp(model, val["apt"], val["dong"], val["mlp"], device)
        return huber_from_numpy(pred, val["y"])

    if verbose:
        print(
            f"MLP 학습 시작 (apt={n_apt}, dong={n_dong}, numeric={len(MLP_FEATURES)})"
        )
    mlp_epoch, mlp_val = fit_model(
        mlp, mlp_loader, mlp_forward, mlp_val_loss, device, clip_grad=None, name="MLP"
    )

    lstm = JeonseHistoryLSTM(
        n_apt=n_apt,
        n_dong=n_dong,
        n_seq=N_SEQ,
        n_static=len(STATIC_FEATURES),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=LSTM_HIDDEN,
        dropout=DROPOUT,
    )
    lstm_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["apt"]),
            torch.from_numpy(train["dong"]),
            torch.from_numpy(train["static"]),
            torch.from_numpy(train["seq"]),
            torch.from_numpy(train["lengths"]),
            torch.from_numpy(train["y"]),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )

    def lstm_forward(model, batch):
        apt_b, dong_b, static_b, seq_b, len_b, _y = batch
        return model(apt_b, dong_b, static_b, seq_b, len_b)

    def lstm_val_loss(model):
        pred = predict_lstm(
            model,
            val["apt"],
            val["dong"],
            val["static"],
            val["seq"],
            val["lengths"],
            device,
        )
        return huber_from_numpy(pred, val["y"])

    if verbose:
        print(f"LSTM 학습 시작 (hidden={LSTM_HIDDEN}, max_len={MAX_SEQ_LEN})")
    lstm_epoch, lstm_val = fit_model(
        lstm,
        lstm_loader,
        lstm_forward,
        lstm_val_loss,
        device,
        clip_grad=GRAD_CLIP,
        name="LSTM",
    )

    results = {}
    results["persistence"] = price_metrics(
        test["frame"]["jeonsePrice"].to_numpy(dtype=np.float64),
        test["frame"]["last_jeonse_price"].to_numpy(dtype=np.float64),
    )
    mlp_log = predict_mlp(mlp, test["apt"], test["dong"], test["mlp"], device)
    results["mlp"] = price_metrics(
        test["frame"]["jeonsePrice"].to_numpy(dtype=np.float64),
        np.exp(mlp_log),
    )
    lstm_log = predict_lstm(
        lstm,
        test["apt"],
        test["dong"],
        test["static"],
        test["seq"],
        test["lengths"],
        device,
    )
    results["lstm"] = price_metrics(
        test["frame"]["jeonsePrice"].to_numpy(dtype=np.float64),
        np.exp(lstm_log),
    )

    if verbose:
        print("\n=== Test (만원, 직전 거래가 있는 행) ===")
        print_metrics("Persistence", results["persistence"], test["frame"])
        print_metrics("LagEmbeddingMLP", results["mlp"], test["frame"])
        print_metrics("EmbeddingLSTM", results["lstm"], test["frame"])
        print(f"\nMLP best epoch {mlp_epoch}, val huber {mlp_val:.5f}")
        print(f"LSTM best epoch {lstm_epoch}, val huber {lstm_val:.5f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(
        {
            "state_dict": mlp.state_dict(),
            "n_apt": n_apt,
            "n_dong": n_dong,
            "n_numeric": len(MLP_FEATURES),
            "embedding_dim": EMBEDDING_DIM,
            "dropout": DROPOUT,
        },
        MLP_PATH,
    )
    torch.save(
        {
            "state_dict": lstm.state_dict(),
            "n_apt": n_apt,
            "n_dong": n_dong,
            "n_seq": N_SEQ,
            "n_static": len(STATIC_FEATURES),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_size": LSTM_HIDDEN,
            "dropout": DROPOUT,
        },
        LSTM_PATH,
    )
    mapping = {
        "apt_to_idx": apt_map,
        "dong_to_idx": dong_map,
        "n_apt": n_apt,
        "n_dong": n_dong,
        "mlp_features": MLP_FEATURES,
        "static_features": STATIC_FEATURES,
        "seq_features": SEQ_FEATURES,
        "mlp_scaler": mlp_scaler,
        "static_scaler": static_scaler,
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "max_seq_len": MAX_SEQ_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "dropout": DROPOUT,
        "lstm_hidden": LSTM_HIDDEN,
        "huber_delta": HUBER_DELTA,
    }
    joblib.dump(mapping, MAPPING_PATH)
    if verbose:
        print(f"\n저장: {MLP_PATH}")
        print(f"저장: {LSTM_PATH}")
        print(f"저장: {MAPPING_PATH}")

    return {
        "test": results,
        "mlp_best_epoch": mlp_epoch,
        "lstm_best_epoch": lstm_epoch,
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
    }


def main() -> None:
    """전세 전용 데이터로 세 모형을 비교 학습한다."""
    run_training(verbose=True)


if __name__ == "__main__":
    main()
