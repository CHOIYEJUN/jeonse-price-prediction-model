# 전세가율

매매가를 분모로 전세가율을 학습하고, 추론 때 그 비율에 매매가를 다시 곱하는 방식이다.

- [문제 정의, 개선 과정, 평가](문제정리.md)
- [Embedding MLP 실행 방법](EMBEDDING_MLP.md)
- [전세만 방식과의 차이](../comparison.md)

코드는 `src/jeonse_ratio/`에 있다. `build_dataset.py`, `train_xgboost.py`, `train_embedding_mlp.py`, `app.py`.
