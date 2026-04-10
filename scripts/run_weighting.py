from __future__ import annotations

import json
from pathlib import Path

from src.preprocessing.normalizer import (
    load_feature_schema,
    normalize_training_data,
)
from src.weighting.weight_calculator import run_weight_analysis
from src.reporting.export_excel import export_weight_analysis_excel


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    input_path = project_root / "data" / "raw" / "조사 자료 정규화.xlsx"
    schema_path = project_root / "config" / "feature_schema.yaml"

    interim_dir = project_root / "data" / "interim"
    output_dir = project_root / "data" / "outputs"

    interim_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_output_path = interim_dir / "normalized_training_data.xlsx"
    encoded_output_path = interim_dir / "encoded_training_data.xlsx"
    category_map_output_path = interim_dir / "category_maps.json"
    weight_analysis_output_path = output_dir / "weight_analysis.xlsx"

    schema = load_feature_schema(schema_path)

    normalized_df, encoded_df, category_maps = normalize_training_data(
        input_path=input_path,
        schema_path=schema_path,
    )

    normalized_df.to_excel(normalized_output_path, index=False)
    encoded_df.to_excel(encoded_output_path, index=False)

    with open(category_map_output_path, "w", encoding="utf-8") as f:
        json.dump(category_maps, f, ensure_ascii=False, indent=2)

    mi_df, xgb_df, comparison_df = run_weight_analysis(
        normalized_df=normalized_df,
        encoded_df=encoded_df,
        target_col=schema.target,
        numeric_cols=schema.numeric_features,
        categorical_cols=schema.categorical_features,
    )

    export_weight_analysis_excel(
        output_path=weight_analysis_output_path,
        normalized_df=normalized_df,
        mi_df=mi_df,
        xgb_df=xgb_df,
        comparison_df=comparison_df,
    )

    print("정규화 및 가중치 분석 완료")
    print(f"- normalized: {normalized_output_path}")
    print(f"- encoded: {encoded_output_path}")
    print(f"- category map: {category_map_output_path}")
    print(f"- weight analysis: {weight_analysis_output_path}")
    print(f"- row count: {len(normalized_df)}")
    print(f"- MI top 3:\n{mi_df.head(3)}")
    print(f"- XGB top 3:\n{xgb_df.head(3)}")


if __name__ == "__main__":
    main()