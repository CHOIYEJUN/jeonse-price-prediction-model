# 성남시 아파트 전세가 예측

코드는 `src` 아래에 방식별로 나뉘어 있다. `src/jeonse_ratio`는 매매가 기준, `src/jeonse_only`는 전세 거래만 쓴다. 차이는 [docs/comparison.md](docs/comparison.md)에 있다.

- [전세가율](docs/jeonse_ratio/문제정리.md): 최근 매매가에 전세가율을 곱해 보증금을 낸다.
- [전세만](docs/jeonse_only/실험정리.md): 매매가 없이, 같은 단지·면적의 과거 전세로 다음 보증금을 낸다.
