# Entity Embedding MLP

아파트명·동을 원핫 대신 임베딩으로 넣는 MLP입니다. XGBoost 학습 스크립트(`training/train_xgboost.py`)는 수정하지 않습니다.

분할은 XGBoost eval과 같습니다. `jeonseYm` 기준 train ≤ 2025.12 / val 2026.01–06 / test ≥ 2026.07.

## 실행 순서

프로젝트 루트에서:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe training\train_embedding_mlp.py
.\.venv\Scripts\python.exe training\visualize_dong_embedding.py
.\.venv\Scripts\python.exe training\compare_models.py
```

비교 전에 XGBoost 모델(`model/xgboost_jeonse_model.pkl`)이 있어야 합니다. 없으면 먼저:

```powershell
.\.venv\Scripts\python.exe training\train_xgboost.py --mode eval
```

## 산출물

- `model/embedding_mlp_model.pt`
- `model/embedding_category_mapping.pkl`
- `model/dong_embedding_visualization.png`

`embedding_dim` 기본값은 8입니다. `training/train_embedding_mlp.py`의 `EMBEDDING_DIM`을 바꿔 튜닝할 수 있습니다.
