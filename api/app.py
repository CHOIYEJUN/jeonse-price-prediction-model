"""학습된 임베딩 MLP로 전세보증금을 추정하는 FastAPI 서버.

계산기(Next)에서 매매가·면적·단지·동 등을 보내면
    예측 전세가율 × 매매가 → 추정 전세보증금
을 JSON으로 돌려준다. 응답 키는 예전 XGBoost API와 같다.
"""

import sys
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _ROOT / "model"
# training/ 안의 JeonseEmbeddingMLP 클래스를 그대로 쓴다
sys.path.insert(0, str(_ROOT / "training"))
from embedding_mlp_model import JeonseEmbeddingMLP  # noqa: E402

UNKNOWN_IDX = 0
# last_jeonse_ratio가 없을 때 쓰는 값 (학습 분포 근처의 중간)
DEFAULT_LAST_RATIO = 0.50
DEFAULT_PERCENTILE = 0.50

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MAPPING_PATH = _MODEL_DIR / "embedding_category_mapping.pkl"
_CKPT_PATH = _MODEL_DIR / "embedding_mlp_model.pt"

if not _MAPPING_PATH.exists() or not _CKPT_PATH.exists():
    raise RuntimeError(
        "임베딩 모형 파일이 없습니다. "
        "training/train_embedding_mlp.py 를 먼저 실행하세요."
    )

# mapping: 단지/동 이름 → 인덱스, 숫자 열 목록, StandardScaler
mapping = joblib.load(_MAPPING_PATH)
try:
    ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(_CKPT_PATH, map_location="cpu")

model = JeonseEmbeddingMLP(
    n_apt=ckpt["n_apt"],
    n_dong=ckpt["n_dong"],
    n_numeric=ckpt["n_numeric"],
    embedding_dim=ckpt["embedding_dim"],
    dropout=ckpt.get("dropout", 0.25),
)
model.load_state_dict(ckpt["state_dict"])
model.eval()


class PredictRequest(BaseModel):
    """POST /predict 본문.

    salePrice: 매매가 (만원)
    area: 전용면적 (㎡). 평이 아님
    floor: 층
    buildYear: 준공연도
    saleYear: 예측 기준 연도
    apartmentName, dong: 학습에 없으면 '기타' 인덱스로 간다
    lastJeonsePrice: 최근 전세보증금 (만원). 있으면 last_jeonse_ratio를 여기서 계산
    lastSaleDate, lastJeonseDate: 'YYYY-MM'. 매칭 시차 계산용
    """

    salePrice: float
    area: float
    floor: int
    buildYear: int
    saleYear: int
    apartmentName: str = "기타"
    dong: str = "기타"
    lastJeonsePrice: float | None = None
    lastSaleDate: str | None = None
    lastJeonseDate: str | None = None
    last_jeonse_ratio: float | None = Field(default=None, ge=0, le=2)
    match_gap_days: int | None = None
    price_percentile_in_dong: float | None = Field(default=None, ge=0, le=1)


def _parse_ym(value: str | None) -> date | None:
    """'2026-05' 또는 '2026-05-01'을 date로 바꾼다. 실패하면 None."""
    if not value:
        return None
    text = str(value).strip()
    for fmt, size in (("%Y-%m-%d", 10), ("%Y-%m", 7)):
        try:
            return datetime.strptime(text[:size], fmt).date()
        except ValueError:
            continue
    return None


def _name_to_idx(name: str, table: dict) -> int:
    """학습 사전에 있는 이름이면 그 번호, 없으면 0(기타)."""
    key = str(name or "").strip() or "기타"
    return int(table.get(key, table.get("기타", UNKNOWN_IDX)))


def _last_ratio(req: PredictRequest) -> float:
    if req.last_jeonse_ratio is not None:
        return float(req.last_jeonse_ratio)
    if req.lastJeonsePrice and req.salePrice > 0:
        return float(req.lastJeonsePrice / req.salePrice)
    return DEFAULT_LAST_RATIO


def _match_gap(req: PredictRequest) -> tuple[int, int]:
    """전세–매매 시차 (일, 연). 요청에 있으면 그대로, 없으면 날짜로 계산."""
    if req.match_gap_days is not None:
        days = int(req.match_gap_days)
        years = int(round(days / 365))
        return days, years

    sale_d = _parse_ym(req.lastSaleDate)
    jeonse_d = _parse_ym(req.lastJeonseDate) or date.today()
    if sale_d is None:
        return 30, 0
    delta = (jeonse_d - sale_d).days
    # 학습은 전세 이전 매매만 썼다. 미래 매매가 오면 0으로 둔다
    days = max(0, min(delta, 365))
    years = int(sale_d.year - jeonse_d.year)
    return days, years


def _feature_row(req: PredictRequest) -> np.ndarray:
    """mapping['numeric_features'] 순서대로 한 행을 만든다."""
    if req.area <= 0:
        raise HTTPException(status_code=400, detail="area(전용면적 ㎡)는 0보다 커야 합니다.")
    last_ratio = _last_ratio(req)
    gap_days, gap_year = _match_gap(req)
    percentile = (
        DEFAULT_PERCENTILE
        if req.price_percentile_in_dong is None
        else float(req.price_percentile_in_dong)
    )
    values = {
        "area": float(req.area),
        "floor": float(req.floor),
        "buildingAge": float(req.saleYear - req.buildYear),
        "salePrice": float(req.salePrice),
        "price_per_m2": float(req.salePrice / req.area),
        "last_jeonse_ratio": last_ratio,
        "match_gap_days": float(gap_days),
        "match_gap_year": float(gap_year),
        "price_percentile_in_dong": percentile,
        "last_3_mean_jeonse_ratio": last_ratio,
        "saleYear": float(req.saleYear),
    }
    cols = mapping["numeric_features"]
    row = np.array([[values[c] for c in cols]], dtype=np.float32)
    return mapping["scaler"].transform(row).astype(np.float32)


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "embedding_mlp",
        "n_apt": int(mapping["n_apt"]),
        "n_dong": int(mapping["n_dong"]),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """전세가율을 추정한 뒤 매매가(만원)를 곱해 보증금을 돌려준다."""
    apt_idx = np.array([_name_to_idx(req.apartmentName, mapping["apt_to_idx"])], dtype=np.int64)
    dong_idx = np.array([_name_to_idx(req.dong, mapping["dong_to_idx"])], dtype=np.int64)
    numeric = _feature_row(req)

    with torch.no_grad():
        pred = model(
            torch.from_numpy(apt_idx),
            torch.from_numpy(dong_idx),
            torch.from_numpy(numeric),
        )
    pred_ratio = float(pred.numpy().reshape(-1)[0])
    predicted_jeonse_price = pred_ratio * req.salePrice

    return {
        "predicted_jeonse_price": float(predicted_jeonse_price),
        "predicted_jeonse_ratio": pred_ratio,
    }
