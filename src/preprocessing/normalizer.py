from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


@dataclass
class FeatureSchema:
    target: str
    methods: List[str]
    exclude_features: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    required_columns: List[str]


def load_feature_schema(schema_path: str | Path) -> FeatureSchema:
    schema_path = Path(schema_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return FeatureSchema(
        target=raw["target"],
        methods=raw["methods"],
        exclude_features=raw.get("exclude_features", []),
        numeric_features=raw.get("numeric_features", []),
        categorical_features=raw.get("categorical_features", []),
        required_columns=raw.get("required_columns", []),
    )


def load_raw_data(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    df = pd.read_excel(input_path)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def validate_required_columns(df: pd.DataFrame, schema: FeatureSchema) -> None:
    missing = [col for col in schema.required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            cleaned[col] = cleaned[col].astype(str).str.strip()

    cleaned = cleaned.replace({
        "": pd.NA,
        " ": pd.NA,
        "-": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
    })

    return cleaned


def filter_valid_methods(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    filtered = df[df[schema.target].isin(schema.methods)].copy()
    return filtered


def coerce_numeric_features(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    converted = df.copy()

    for col in schema.numeric_features:
        if col in converted.columns:
            converted[col] = pd.to_numeric(converted[col], errors="coerce")

    return converted


def fill_missing_values(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    filled = df.copy()

    for col in schema.numeric_features:
        if col in filled.columns:
            median_value = filled[col].median()
            filled[col] = filled[col].fillna(median_value)

    for col in schema.categorical_features:
        if col in filled.columns:
            mode_series = filled[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if not mode_series.empty else "미상"
            filled[col] = filled[col].fillna(fill_value)

    filled = filled.dropna(subset=[schema.target]).copy()

    return filled


def build_model_input(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    keep_columns = (
        ["사례명"]
        + schema.numeric_features
        + schema.categorical_features
        + [schema.target]
    )

    keep_columns = [col for col in keep_columns if col in df.columns]
    model_df = df[keep_columns].copy()

    return model_df


def encode_categorical_features(
    df: pd.DataFrame,
    schema: FeatureSchema
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    encoded = df.copy()
    category_maps: Dict[str, Dict[str, int]] = {}

    for col in schema.categorical_features + [schema.target]:
        if col in encoded.columns:
            categories = sorted(encoded[col].dropna().astype(str).unique().tolist())
            mapping = {value: idx for idx, value in enumerate(categories)}
            encoded[col] = encoded[col].astype(str).map(mapping)
            category_maps[col] = mapping

    return encoded, category_maps


def normalize_training_data(
    input_path: str | Path,
    schema_path: str | Path
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]]]:
    schema = load_feature_schema(schema_path)

    df = load_raw_data(input_path)
    validate_required_columns(df, schema)
    df = clean_dataframe(df)
    df = filter_valid_methods(df, schema)
    df = coerce_numeric_features(df, schema)
    df = fill_missing_values(df, schema)

    normalized_df = build_model_input(df, schema)
    encoded_df, category_maps = encode_categorical_features(normalized_df, schema)

    return normalized_df, encoded_df, category_maps