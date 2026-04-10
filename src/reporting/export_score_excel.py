from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def export_score_result_excel(
    output_path: str | Path,
    input_data: Dict[str, object],
    result_df: pd.DataFrame,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_df = pd.DataFrame([input_data])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        input_df.to_excel(writer, sheet_name="input_data", index=False)
        result_df.to_excel(writer, sheet_name="result", index=False)