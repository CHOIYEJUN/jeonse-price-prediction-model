# Entity Embedding MLP

전세가율 방식의 신경망이다. 아파트명·동을 원핫 대신 임베딩으로 넣는 MLP이다. 전체 비교는 [방식 비교](../comparison.md)에 있다.

XGBoost 학습 스크립트(`src/jeonse_ratio/train_xgboost.py`)는 수정하지 않는다.

분할은 XGBoost eval과 같습니다. `jeonseYm` 기준 train ≤ 2025.12 / val 2026.01–06 / test ≥ 2026.07.

## 실행 순서

프로젝트 루트에서:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\jeonse_ratio\train_embedding_mlp.py
.\.venv\Scripts\python.exe src\jeonse_ratio\visualize_dong_embedding.py
.\.venv\Scripts\python.exe src\jeonse_ratio\compare_models.py
```

비교 전에 XGBoost 모델(`model/xgboost_jeonse_model.pkl`)이 있어야 합니다. 없으면 먼저:

```powershell
.\.venv\Scripts\python.exe src\jeonse_ratio\train_xgboost.py --mode eval
```

## 산출물

- `model/embedding_mlp_model.pt`
- `model/embedding_category_mapping.pkl`
- `model/dong_embedding_visualization.png`

`embedding_dim` 기본값은 8입니다. `src/jeonse_ratio/train_embedding_mlp.py`의 `EMBEDDING_DIM`을 바꿔 튜닝할 수 있습니다.
