from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


def _normalize_scores(score_dict: Dict[str, float]) -> Dict[str, float]:
    total = sum(score_dict.values())
    if total <= 0:
        n = len(score_dict)
        if n == 0:
            return {}
        return {k: 1.0 / n for k in score_dict}
    return {k: v / total for k, v in score_dict.items()}


def calculate_mi_weights(
    encoded_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    X = encoded_df[feature_cols].copy()
    y = encoded_df[target_col].copy()

    discrete_mask = [col in categorical_cols for col in feature_cols]

    mi_scores = mutual_info_classif(
        X=X,
        y=y,
        discrete_features=discrete_mask,
        random_state=42,
    )

    raw_scores = {col: float(score) for col, score in zip(feature_cols, mi_scores)}
    normalized_scores = _normalize_scores(raw_scores)

    result_df = pd.DataFrame({
        "변수명": feature_cols,
        "MI_raw_score": [raw_scores[col] for col in feature_cols],
        "MI_normalized_weight": [normalized_scores[col] for col in feature_cols],
    })

    result_df = result_df.sort_values(
        by="MI_normalized_weight",
        ascending=False
    ).reset_index(drop=True)

    return result_df


def calculate_xgb_importance(
    normalized_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    X = normalized_df[feature_cols].copy()
    y = normalized_df[target_col].copy()

    # 타깃 인코딩
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y.astype(str))

    # 범주형 원핫 인코딩
    X_encoded = pd.get_dummies(
        X,
        columns=[col for col in categorical_cols if col in X.columns],
        dummy_na=False,
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )

    model.fit(X_encoded, y_encoded)

    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")

    # 원핫 컬럼별 gain을 원래 변수 단위로 다시 합산
    aggregated_scores: Dict[str, float] = {col: 0.0 for col in feature_cols}

    for encoded_col, gain in gain_scores.items():
        matched = False

        # 숫자형은 컬럼명이 그대로 들어옴
        if encoded_col in aggregated_scores:
            aggregated_scores[encoded_col] += float(gain)
            matched = True

        # 범주형은 get_dummies 후 "원래컬럼명_범주값" 형태라 prefix로 합산
        if not matched:
            for original_col in categorical_cols:
                prefix = f"{original_col}_"
                if encoded_col.startswith(prefix):
                    aggregated_scores[original_col] += float(gain)
                    matched = True
                    break

    normalized_scores = _normalize_scores(aggregated_scores)

    result_df = pd.DataFrame({
        "변수명": feature_cols,
        "XGB_gain_raw_score": [aggregated_scores[col] for col in feature_cols],
        "XGB_normalized_importance": [normalized_scores[col] for col in feature_cols],
    })

    result_df = result_df.sort_values(
        by="XGB_normalized_importance",
        ascending=False
    ).reset_index(drop=True)

    return result_df


def compare_weight_results(
    mi_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.merge(
        mi_df,
        xgb_df,
        on="변수명",
        how="outer",
    )

    merged["차이(MI-XGB)"] = (
        merged["MI_normalized_weight"] - merged["XGB_normalized_importance"]
    ).abs()

    merged = merged.sort_values(
        by="차이(MI-XGB)",
        ascending=False
    ).reset_index(drop=True)

    return merged


def run_weight_analysis(
    normalized_df: pd.DataFrame,
    encoded_df: pd.DataFrame,
    target_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = numeric_cols + categorical_cols

    mi_df = calculate_mi_weights(
        encoded_df=encoded_df,
        target_col=target_col,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
    )

    xgb_df = calculate_xgb_importance(
        normalized_df=normalized_df,
        target_col=target_col,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
    )

    comparison_df = compare_weight_results(mi_df, xgb_df)

    return mi_df, xgb_df, comparison_df