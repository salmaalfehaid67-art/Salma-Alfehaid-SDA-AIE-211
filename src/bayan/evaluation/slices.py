"""Lab 6: slice evaluation."""

from typing import Callable, Iterable

import numpy as np
import pandas as pd


def _length_bucket(text: str) -> str:
    n = len(str(text).split())

    if n <= 5:
        return "short"
    if n <= 15:
        return "medium"
    return "long"


def evaluate_slices(
    df: pd.DataFrame,
    *,
    y_true_col="gold",
    y_pred_col="pred",
    text_col="text",
    language_col="lang",
    dialect_col="dialect_region",
    class_col=None,
    metric_fn: Callable[[Iterable, Iterable], float] | None = None,
    min_slice_size=20,
):
    if metric_fn is None:
        def metric_fn(y_true, y_pred):
            y_true = np.asarray(list(y_true))
            y_pred = np.asarray(list(y_pred))
            return float(np.mean(y_true == y_pred))

    work = df.copy()
    results = []

    def add_group(slice_type, series):
        for value, idx in series.groupby(series).groups.items():
            subset = work.loc[idx]
            n = len(subset)

            score = metric_fn(
                subset[y_true_col],
                subset[y_pred_col]
            )

            results.append({
                "slice_type": slice_type,
                "slice_value": str(value),
                "n": int(n),
                "score": float(score),
                "small_slice": bool(n < min_slice_size),
            })

    if language_col in work.columns:
        add_group(
            "language",
            work[language_col].fillna("UNKNOWN")
        )

    if dialect_col in work.columns:
        add_group(
            "dialect",
            work[dialect_col].fillna("UNKNOWN")
        )

    if class_col and class_col in work.columns:
        add_group(
            "class",
            work[class_col].fillna("UNKNOWN")
        )
    else:
        add_group(
            "class",
            work[y_true_col].fillna("UNKNOWN")
        )

    if text_col in work.columns:
        add_group(
            "length",
            work[text_col]
            .fillna("")
            .astype(str)
            .map(_length_bucket)
        )

    return pd.DataFrame(results)
