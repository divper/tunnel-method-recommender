from __future__ import annotations

from typing import Dict

import pandas as pd


def apply_feasibility_rules(
    input_data: Dict[str, object],
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    df = result_df.copy()

    location = str(input_data.get("시공 위치_정규화", "")).strip()

    if "판정" not in df.columns:
        df["판정"] = ""
    if "비고" not in df.columns:
        df["비고"] = ""

    # 1) 기본 판정: 최종 점수 기준
    df.loc[df["점수(100점 만점)"] >= 85, "판정"] = "적합"
    df.loc[(df["점수(100점 만점)"] >= 60) & (df["점수(100점 만점)"] < 85), "판정"] = "가능"
    df.loc[df["점수(100점 만점)"] < 60, "판정"] = "추천하지 않음"

    # 2) 절대적 적용 불가 조건만 별도 반영
    # 침매식은 수중 조건이 아닐 경우 적용 곤란 처리
    if location not in ["수중", "해저", "하천", "호수", "수중구간"]:
        mask = df["공법"] == "침매식"
        df.loc[mask, "점수(100점 만점)"] = 0
        df.loc[mask, "판정"] = "적용 곤란"
        df.loc[mask, "비고"] = "수중 조건이 아니므로 침매식 적용 곤란"

    df = df.sort_values(by="점수(100점 만점)", ascending=False).reset_index(drop=True)

    return df