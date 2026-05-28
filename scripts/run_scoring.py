from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.normalizer import (
    load_feature_schema,
    normalize_training_data,
)
from src.inference.model_trainer import train_method_score_models
from src.inference.scorer import score_methods
from src.reporting.export_score_excel import export_score_result_excel


def load_input_case(input_excel_path: Path) -> dict:
    if not input_excel_path.exists():
        raise FileNotFoundError(f"입력 엑셀 파일이 없습니다: {input_excel_path}")

    df = pd.read_excel(input_excel_path)

    if df.empty:
        raise ValueError("입력 엑셀에 데이터가 없습니다.")

    input_data = df.iloc[0].to_dict()
    return input_data


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    training_input_path = project_root / "data" / "raw" / "조사 자료 정규화.xlsx"
    schema_path = project_root / "config" / "feature_schema.yaml"
    input_case_path = project_root / "data" / "input" / "input_case.xlsx"
    output_result_path = project_root / "data" / "outputs" / "result.xlsx"

    schema = load_feature_schema(schema_path)

    normalized_df, encoded_df, category_maps = normalize_training_data(
        input_path=training_input_path,
        schema_path=schema_path,
    )

    bundle = train_method_score_models(
        normalized_df=normalized_df,
        target_col=schema.target,
        methods=schema.methods,
        numeric_cols=schema.numeric_features,
        categorical_cols=schema.categorical_features,
    )

    input_data = load_input_case(input_case_path)

    result_df = score_methods(
        input_data=input_data,
        bundle=bundle,
    )

    export_score_result_excel(
        output_path=output_result_path,
        input_data=input_data,
        result_df=result_df,
    )

    print("공법 점수 result 엑셀 생성 완료")
    print(f"- input: {input_case_path}")
    print(f"- output: {output_result_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()