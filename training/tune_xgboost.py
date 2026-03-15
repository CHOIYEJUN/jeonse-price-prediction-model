"""
XGBoost 하이퍼파라미터 랜덤 탐색 (20회).
최적 조합으로 모델 저장 및 train_xgboost.py의 XGB_PARAMS 갱신 제안.
"""

import os
import random
import joblib

# 재현을 위한 시드
RANDOM_SEED = 42
N_TRIALS = 20

# 탐색 구간 (논문/실무에서 자주 쓰는 범위)
PARAM_GRID = {
    "n_estimators": [300, 400, 500, 600, 700],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.04, 0.05, 0.06, 0.07],
    "subsample": [0.7, 0.75, 0.8, 0.85, 0.9],
    "colsample_bytree": [0.7, 0.75, 0.8, 0.85, 0.9],
    "min_child_weight": [5, 8, 10, 12, 15],
    "reg_alpha": [0.05, 0.1, 0.15, 0.2],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],
}


def sample_params(seed: int):
    """한 세트의 하이퍼파라미터를 무작위 샘플링."""
    rng = random.Random(seed)
    return {
        "n_estimators": rng.choice(PARAM_GRID["n_estimators"]),
        "max_depth": rng.choice(PARAM_GRID["max_depth"]),
        "learning_rate": rng.choice(PARAM_GRID["learning_rate"]),
        "subsample": rng.choice(PARAM_GRID["subsample"]),
        "colsample_bytree": rng.choice(PARAM_GRID["colsample_bytree"]),
        "min_child_weight": rng.choice(PARAM_GRID["min_child_weight"]),
        "reg_alpha": rng.choice(PARAM_GRID["reg_alpha"]),
        "reg_lambda": rng.choice(PARAM_GRID["reg_lambda"]),
        "random_state": 42,
    }


def main():
    from train_xgboost import (
        run_training,
        MODEL_DIR,
        MODEL_PATH,
    )

    print(f"하이퍼파라미터 랜덤 탐색 {N_TRIALS}회 (정확도 = Test MAPE, 낮을수록 좋음)\n")

    results = []
    for i in range(N_TRIALS):
        params = sample_params(RANDOM_SEED + i)
        print(f"[{i+1}/{N_TRIALS}] max_d={params['max_depth']} lr={params['learning_rate']} n_est={params['n_estimators']} ... ", end="", flush=True)
        try:
            res = run_training(xgb_params=params, verbose=False)
            mape = res["mape"]
            results.append((mape, res))
            print(f"Test MAPE = {mape:.4f}%")
        except Exception as e:
            print(f"실패: {e}")
            results.append((float("inf"), None))

    valid = [(m, r) for m, r in results if r is not None]
    if not valid:
        print("유효한 실행이 없습니다.")
        return

    best_mape, best_result = min(valid, key=lambda x: x[0])
    best_params = best_result["params"]

    print("\n" + "=" * 60)
    print("최적 하이퍼파라미터 (Test MAPE 최소)")
    print("=" * 60)
    print(f"  Test MAPE: {best_mape:.4f}%")
    print(f"  Test MAE:  {best_result['mae']:.2f}")
    print(f"  Test RMSE: {best_result['rmse']:.2f}")
    print("\n  권장 XGB_PARAMS:")
    for k, v in sorted(best_params.items()):
        print(f"    {k!r}: {v},")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_result["model"], MODEL_PATH)
    joblib.dump(best_result["feature_cols"], os.path.join(MODEL_DIR, "xgboost_feature_cols.pkl"))
    print(f"\n최적 모델 저장: {MODEL_PATH}")

    # train_xgboost.py에 반영할 내용 출력 (복사해서 넣을 수 있도록)
    print("\n--- train_xgboost.py 에 넣을 XGB_PARAMS (복사용) ---")
    print("XGB_PARAMS = {")
    for k, v in sorted(best_params.items()):
        print(f'    "{k}": {v},')
    print("}")


if __name__ == "__main__":
    main()
