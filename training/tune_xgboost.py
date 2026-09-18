"""XGBoost 설정을 무작위로 20번 뽑아 검증 점수가 가장 좋은 조합을 고른다.

하이퍼파라미터
    학습이 정하는 가중치가 아니라, 사람이 미리 고르는 설정
    (나무 깊이, 학습률 등).
목적함수
    검증 구간 평균절대백분율오차(Val MAPE). 작을수록 좋다.
주의
    평가(Test) MAPE는 고른 뒤에 한 번만 본다. 튜닝에 쓰지 않는다.
"""

import os
import random
import joblib

RANDOM_SEED = 42  # 난수 시작값. 같으면 같은 순서로 조합이 뽑힌다
N_TRIALS = 20  # 몇 번 다른 설정을 시험할지

# 각 키의 리스트에서 하나씩 고른다. 가능한 모든 조합을 다 쓰는 것은 아니다.
PARAM_GRID = {
    "n_estimators": [300, 400, 500, 600, 700],  # 나무 그루 수
    "max_depth": [3, 4, 5, 6],  # 한 나무가 몇 단까지 자를지
    "learning_rate": [0.03, 0.04, 0.05, 0.06, 0.07],  # 한 그루가 잔차를 얼마나 반영할지
    "subsample": [0.7, 0.75, 0.8, 0.85, 0.9],  # 각 나무에 쓸 행의 비율
    "colsample_bytree": [0.7, 0.75, 0.8, 0.85, 0.9],  # 각 나무에 쓸 열의 비율
    "min_child_weight": [5, 8, 10, 12, 15],  # 잎에 필요한 최소 가중치. 크면 단순한 나무
    "reg_alpha": [0.05, 0.1, 0.15, 0.2],  # L1 규제. 불필요한 분할을 줄인다
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],  # L2 규제
}


def sample_params(seed: int):
    """seed를 고정한 난수로 PARAM_GRID에서 설정 하나를 뽑는다.

    seed: 난수 씨앗. 시행마다 다르게 주면 다른 조합이 나온다.
    반환: XGBRegressor에 넣을 딕셔너리
    """
    rng = random.Random(seed)  # 전역 random과 섞이지 않는 독립 난수기
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
    """20회 학습 후 Val MAPE가 가장 낮은 모형을 저장한다."""
    from train_xgboost import (
        run_training,
        MODEL_DIR,
        MODEL_PATH,
    )

    print(f"하이퍼파라미터 랜덤 탐색 {N_TRIALS}회 (정확도 = Val MAPE, 낮을수록 좋음)\n")

    # results: (검증 MAPE, run_training이 돌려준 결과 사전) 목록
    results = []
    for i in range(N_TRIALS):
        params = sample_params(RANDOM_SEED + i)
        print(
            f"[{i+1}/{N_TRIALS}] max_d={params['max_depth']} "
            f"lr={params['learning_rate']} n_est={params['n_estimators']} ... ",
            end="",
            flush=True,
        )
        try:
            # verbose=False: 매 시행마다 긴 로그를 찍지 않는다
            res = run_training(xgb_params=params, verbose=False, mode="eval")
            val_mape = res["val_mape"]  # 검증 구간 평균절대백분율오차
            results.append((val_mape, res))
            print(f"Val MAPE = {val_mape:.4f}%")
        except Exception as e:
            print(f"실패: {e}")
            # 실패한 시행은 무한대로 두어 최소값 비교에서 빠지게 한다
            results.append((float("inf"), None))

    # 학습이 성공한 시행만
    valid = [(m, r) for m, r in results if r is not None]
    if not valid:
        print("유효한 실행이 없습니다.")
        return

    # key=lambda x: x[0] → 튜플의 첫 값(Val MAPE)이 가장 작은 것
    best_val, best_result = min(valid, key=lambda x: x[0])
    best_params = best_result["params"]

    print("\n" + "=" * 60)
    print("최적 하이퍼파라미터 (Val MAPE 최소)")
    print("=" * 60)
    print(f"  Val MAPE:  {best_val:.4f}%")
    print(f"  Test MAPE: {best_result['mape']:.4f}%  (최종 1회, 튜닝에 사용하지 않음)")
    print(f"  Test MAE:  {best_result['mae']:.2f}")
    print(f"  Test RMSE: {best_result['rmse']:.2f}")
    print("\n  권장 XGB_PARAMS:")
    for k, v in sorted(best_params.items()):
        print(f"    {k!r}: {v},")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_result["model"], MODEL_PATH)
    joblib.dump(best_result["feature_cols"], os.path.join(MODEL_DIR, "xgboost_feature_cols.pkl"))
    print(f"\n최적(eval) 모델 저장: {MODEL_PATH}")
    print("배포 모델은 training/train_xgboost.py --mode production 으로 다시 학습하세요.")

    print("\n--- train_xgboost.py 에 넣을 XGB_PARAMS (복사용) ---")
    print("XGB_PARAMS = {")
    for k, v in sorted(best_params.items()):
        print(f'    "{k}": {v},')
    print("}")


if __name__ == "__main__":
    main()
