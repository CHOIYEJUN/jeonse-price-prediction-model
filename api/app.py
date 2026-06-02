import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS: 프론트에서 API 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용: 모든 출처 허용. 배포 시 특정 도메인만 지정 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드 (app.py 위치 기준 경로 → 프로젝트 루트/model)
_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
model = pickle.load(open(_MODEL_DIR / "xgboost_jeonse_model.pkl", "rb"))
feature_cols = pickle.load(open(_MODEL_DIR / "xgboost_feature_cols.pkl", "rb"))


class PredictRequest(BaseModel):
    salePrice: float
    area: float
    floor: int
    buildYear: int
    saleYear: int


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):

    data = pd.DataFrame([{
        "salePrice": req.salePrice,
        "area": req.area,
        "floor": req.floor,
        "buildYear": req.buildYear,
        "saleYear": req.saleYear
    }])

    # feature 맞추기
    data = data.reindex(columns=feature_cols, fill_value=0)

    pred_ratio = model.predict(data)[0]  # 전세비율 (0~1)
    predicted_jeonse_price = pred_ratio * req.salePrice  # 실제 예측 전세가 (매매가와 동일 단위)

    return {
        "predicted_jeonse_price": float(predicted_jeonse_price),
        "predicted_jeonse_ratio": float(pred_ratio),  # 필요하면 % 표시용 (0.55 → 55%)
    }