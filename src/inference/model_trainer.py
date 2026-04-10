from __future__ import annotations

from sklearn.model_selection import KFold
from src.inference.calibration import build_calibration_table

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from xgboost import XGBClassifier


@dataclass
class ScoreModelBundle:
    methods: List[str]
    feature_columns: List[str]
    categorical_columns: List[str]
    models: Dict[str, XGBClassifier]
    dummy_columns: List[str]
    calibration: Dict[str, Dict]


def build_training_matrix(
    normalized_df: pd.DataFrame,
    feature_columns: List[str],
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    X = normalized_df[feature_columns].copy()

    X_encoded = pd.get_dummies(
        X,
        columns=[col for col in categorical_columns if col in X.columns],
        dummy_na=False,
    )

    return X_encoded, X_encoded.columns.tolist()


def train_method_score_models(
    normalized_df: pd.DataFrame,
    target_col: str,
    methods: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ScoreModelBundle:
    feature_columns = numeric_cols + categorical_cols

    X_encoded, dummy_columns = build_training_matrix(
        normalized_df=normalized_df,
        feature_columns=feature_columns,
        categorical_columns=categorical_cols,
    )

    y = normalized_df[target_col].astype(str)
    models: Dict[str, XGBClassifier] = {}

    for method in methods:
        y_binary = (y == method).astype(int)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )

        model.fit(X_encoded, y_binary)
        models[method] = model

    # === OOF raw score 생성 ===
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    raw_scores = {method: [] for method in methods}

    for train_idx, val_idx in kf.split(X_encoded):
        X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for method in methods:
            y_binary = (y_train == method).astype(int)

            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )

            model.fit(X_train, y_binary)

            probs = model.predict_proba(X_val)[:, 1]
            raw_scores[method].extend(probs.tolist())

    calibration = build_calibration_table(raw_scores)

    return ScoreModelBundle(
        methods=methods,
        feature_columns=feature_columns,
        categorical_columns=categorical_cols,
        models=models,
        dummy_columns=dummy_columns,
        calibration=calibration,   # 추가
    )