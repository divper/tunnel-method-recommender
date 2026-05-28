from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.normalizer import (
    load_feature_schema,
    normalize_training_data,
)
from src.inference.model_trainer import train_method_score_models
from src.inference.scorer import score_methods


def is_acceptable_prediction(
    actual_method: str,
    score_df: pd.DataFrame,
) -> tuple[bool, str]:
    top1 = score_df.iloc[0]
    top2 = score_df.iloc[1]
    top3 = score_df.iloc[2]

    top1_method = str(top1["공법"])
    top2_method = str(top2["공법"])
    top3_method = str(top3["공법"])

    top1_score = float(top1["점수(100점 만점)"])
    top2_score = float(top2["점수(100점 만점)"])

    top1_judgement = str(top1["판정"])
    top2_judgement = str(top2["판정"])
    top3_judgement = str(top3["판정"])

    if actual_method == top1_method:
        return True, "1순위 정답"

    same_score = top1_score == top2_score
    both_suitable = top1_judgement == "적합" and top2_judgement == "적합"

    if actual_method == top2_method and (same_score or both_suitable):
        return True, "2순위 인정"

    possible_judgements = ["적합", "가능"]

    if actual_method == top3_method and top3_judgement in possible_judgements:
        return True, "3순위 가능 인정"

    return False, "오답"


def main() -> None:
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "조사 자료 정규화.xlsx"
    schema_path = PROJECT_ROOT / "config" / "feature_schema.yaml"
    output_path = PROJECT_ROOT / "data" / "outputs" / "backtest_result.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = load_feature_schema(schema_path)

    normalized_df, _, _ = normalize_training_data(
        input_path=raw_data_path,
        schema_path=schema_path,
    )

    results: list[dict] = []
    total_count = len(normalized_df)

    print("===== Leave-One-Out Backtest 시작 =====")
    print(f"전체 데이터 개수: {total_count}")
    print()

    for test_idx in range(total_count):
        test_row = normalized_df.iloc[test_idx]
        train_df = normalized_df.drop(index=test_idx).reset_index(drop=True)

        actual_method = str(test_row[schema.target])

        bundle = train_method_score_models(
            normalized_df=train_df,
            target_col=schema.target,
            methods=schema.methods,
            numeric_cols=schema.numeric_features,
            categorical_cols=schema.categorical_features,
        )

        input_data = {}
        for col in schema.numeric_features + schema.categorical_features:
            input_data[col] = test_row[col]

        score_df = score_methods(
            input_data=input_data,
            bundle=bundle,
        )

        top1 = score_df.iloc[0]
        top2 = score_df.iloc[1]
        top3 = score_df.iloc[2]

        correct, reason = is_acceptable_prediction(
            actual_method=actual_method,
            score_df=score_df,
        )

        results.append(
            {
                "index": test_idx,
                "actual_method": actual_method,

                "predicted_1st_method": str(top1["공법"]),
                "predicted_1st_score": top1["점수(100점 만점)"],
                "predicted_1st_raw_score": top1["raw_score"],
                "predicted_1st_judgement": top1["판정"],

                "predicted_2nd_method": str(top2["공법"]),
                "predicted_2nd_score": top2["점수(100점 만점)"],
                "predicted_2nd_raw_score": top2["raw_score"],
                "predicted_2nd_judgement": top2["판정"],

                "predicted_3rd_method": str(top3["공법"]),
                "predicted_3rd_score": top3["점수(100점 만점)"],
                "predicted_3rd_raw_score": top3["raw_score"],
                "predicted_3rd_judgement": top3["판정"],

                "correct": correct,
                "reason": reason,
            }
        )

        print(
            f"[{test_idx + 1}/{total_count}] "
            f"실제={actual_method} | "
            f"1순위={top1['공법']}({top1['점수(100점 만점)']}점, {top1['판정']}) | "
            f"2순위={top2['공법']}({top2['점수(100점 만점)']}점, {top2['판정']}) | "
            f"3순위={top3['공법']}({top3['점수(100점 만점)']}점, {top3['판정']}) | "
            f"결과={reason}"
        )

    result_df = pd.DataFrame(results)

    correct_count = int(result_df["correct"].sum())
    wrong_count = total_count - correct_count
    accuracy_percent = result_df["correct"].mean() * 100

    first_rank_correct = (result_df["reason"] == "1순위 정답").sum()
    second_rank_accepted = (result_df["reason"] == "2순위 인정").sum()
    third_rank_accepted = (result_df["reason"] == "3순위 가능 인정").sum()

    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print()
    print("===== BACKTEST RESULT =====")
    print(f"전체 개수: {total_count}")
    print(f"정답 개수: {correct_count}")
    print(f"오답 개수: {wrong_count}")
    print(f"정답률: {accuracy_percent:.2f}%")
    print()
    print("===== 인정 기준별 개수 =====")
    print(f"1순위 정답: {first_rank_correct}")
    print(f"2순위 인정: {second_rank_accepted}")
    print(f"3순위 가능 인정: {third_rank_accepted}")
    print()
    print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()