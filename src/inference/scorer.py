from __future__ import annotations

from typing import Dict

import pandas as pd

from src.inference.rule_filter import apply_feasibility_rules
from src.inference.calibration import calibrate_score

from src.inference.model_trainer import ScoreModelBundle


def _prepare_single_input_frame(
    input_data: Dict[str, object],
    bundle: ScoreModelBundle,
) -> pd.DataFrame:
    row = {col: input_data.get(col) for col in bundle.feature_columns}
    input_df = pd.DataFrame([row])

    # 숫자형 강제 변환
    for col in bundle.feature_columns:
        if col in input_df.columns and col not in bundle.categorical_columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    input_encoded = pd.get_dummies(
        input_df,
        columns=[col for col in bundle.categorical_columns if col in input_df.columns],
        dummy_na=False,
    )

    input_encoded = input_encoded.reindex(columns=bundle.dummy_columns, fill_value=0)

    return input_encoded


def score_methods(
    input_data: Dict[str, object],
    bundle: ScoreModelBundle,
) -> pd.DataFrame:
    input_encoded = _prepare_single_input_frame(
        input_data=input_data,
        bundle=bundle,
    )

    rows = []

    for method in bundle.methods:
        model = bundle.models[method]
        raw_score = float(model.predict_proba(input_encoded)[0][1])

        calibrated = calibrate_score(
            method,
            raw_score,
            bundle.calibration
        )

        rows.append({
            "공법": method,
            "raw_score": round(raw_score, 4),
            "점수(100점 만점)": calibrated,
        })

    result_df = pd.DataFrame(rows).sort_values(
        by="점수(100점 만점)",
        ascending=False
    ).reset_index(drop=True)

    result_df = apply_feasibility_rules(
        input_data=input_data,
        result_df=result_df,
    )

    return result_df