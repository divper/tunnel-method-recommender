from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_weight_analysis_excel(
    output_path: str | Path,
    normalized_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        normalized_df.to_excel(writer, sheet_name="normalized_data", index=False)
        mi_df.to_excel(writer, sheet_name="mi_weights", index=False)
        xgb_df.to_excel(writer, sheet_name="xgb_importance", index=False)
        comparison_df.to_excel(writer, sheet_name="weight_comparison", index=False)