from __future__ import annotations

from pathlib import Path

from src.preprocessing.normalizer import (
    load_feature_schema,
    normalize_training_data,
)
from src.inference.model_trainer import train_method_score_models
from src.inference.scorer import score_methods


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    input_path = project_root / "data" / "raw" / "조사 자료 정규화.xlsx"
    schema_path = project_root / "config" / "feature_schema.yaml"

    schema = load_feature_schema(schema_path)

    normalized_df, encoded_df, category_maps = normalize_training_data(
        input_path=input_path,
        schema_path=schema_path,
    )

    bundle = train_method_score_models(
        normalized_df=normalized_df,
        target_col=schema.target,
        methods=schema.methods,
        numeric_cols=schema.numeric_features,
        categorical_cols=schema.categorical_features,
    )

    # 테스트 입력값 예시
    test_input = {
        "심도(m)_정규화": 25,
        "구경(m)_정규화": 6.5,
        "터널 길이(km)_정규화": 3.2,
        "지반 조건_정규화": "연약지반",
        "시공 위치_정규화": "도심지",
        "주변 민감도_정규화": "높음",
        "요구사항_주요정규화": "침하·변형관리",
        "요구사항_보조정규화": "고지압",
    }

    result_df = score_methods(
        input_data=test_input,
        bundle=bundle,
    )

    print("공법별 점수 결과")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()