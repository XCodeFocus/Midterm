from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class EvalConfig:
    label_column: str
    positive_label: str
    categorical_columns: Sequence[str]
    numeric_columns: Sequence[str]
    test_size: float = 0.25
    random_state: int = 42


def _make_preprocessor(cfg: EvalConfig) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                list(cfg.categorical_columns),
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(cfg.numeric_columns),
            ),
        ],
        remainder="drop",
    )


def _binarize_y(y: pd.Series, positive_label: str) -> np.ndarray:
    y = y.astype(str).str.strip()
    return (y == positive_label).astype(int).to_numpy()


def build_feature_columns(*, numeric_qi: Sequence[str], categorical_qi: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (categorical_columns, numeric_columns) for generalized datasets."""

    cat_cols = list(categorical_qi)
    num_cols: List[str] = []
    for col in numeric_qi:
        num_cols.append(f"{col}_mid")
        num_cols.append(f"{col}_width")
    return cat_cols, num_cols


def evaluate_dataset(
    df: pd.DataFrame,
    *,
    cfg: EvalConfig,
    train_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate SVM on a single dataset."""

    df = df.reset_index(drop=True)

    X = df[list(cfg.categorical_columns) + list(cfg.numeric_columns)]
    y = _binarize_y(df[cfg.label_column], cfg.positive_label)

    if train_idx is None or test_idx is None:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y,
        )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pre = _make_preprocessor(cfg)

    models = {
        "svm_linear": Pipeline(
            steps=[
                ("pre", pre),
                ("clf", LinearSVC(C=1.0)),
            ]
        ),
    }

    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # AUC uses decision function if available; otherwise probabilities.
        y_score = None
        if hasattr(model, "decision_function"):
            try:
                y_score = model.decision_function(X_test)
            except Exception:
                y_score = None
        if y_score is None and hasattr(model, "predict_proba"):
            try:
                y_score = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_score = None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_score)) if y_score is not None else float("nan")

        results[name] = {
            "misclassification": 1.0 - acc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "auc": auc,
        }

    return results


def make_split_indices(df: pd.DataFrame, *, cfg: EvalConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Create a single, reusable split for fair comparisons."""

    df = df.reset_index(drop=True)
    y = _binarize_y(df[cfg.label_column], cfg.positive_label)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    return train_idx, test_idx
