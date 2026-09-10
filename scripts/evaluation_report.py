"""Lab 6: generate Bayan evaluation report."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from bayan.evaluation.bootstrap import bootstrap_ci
from bayan.evaluation.slices import evaluate_slices


PREDICTIONS_PATH = Path(
    "data/eval/validation_predictions.csv"
)

OUTPUT_PATH = Path(
    "EVALUATION_REPORT.md"
)

RETRIEVAL_RESULTS = Path(
    "artifacts/retrieval_eval.json"
)


def pick_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"None of the columns {candidates} "
        f"were found. Available columns: "
        f"{list(df.columns)}"
    )


def macro_f1(y_true, y_pred):
    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )


def bootstrap_macro_f1(
    y_true,
    y_pred,
    n_boot=1000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    scores = []

    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        score = macro_f1(
            y_true[idx],
            y_pred[idx],
        )

        scores.append(score)

    point = macro_f1(
        y_true,
        y_pred,
    )

    lower = float(
        np.percentile(
            scores,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            scores,
            97.5,
        )
    )

    return point, lower, upper


def build_slice_table(
    df,
    true_col,
    pred_col,
    text_col,
):
    slices = evaluate_slices(
        df,
        y_true_col=true_col,
        y_pred_col=pred_col,
        text_col=text_col,
        language_col="lang",
        dialect_col="dialect_region",
        metric_fn=macro_f1,
        min_slice_size=20,
    )

    if slices.empty:
        return (
            "No slice results available."
        )

    lines = [
        "| Slice type | Slice | N | Score | Small slice |",
        "|---|---|---:|---:|---|",
    ]

    for _, row in slices.iterrows():
        lines.append(
            f"| {row['slice_type']} "
            f"| {row['slice_value']} "
            f"| {int(row['n'])} "
            f"| {row['score']:.4f} "
            f"| {row['small_slice']} |"
        )

    return "\n".join(lines)


def load_retrieval_summary():
    if not RETRIEVAL_RESULTS.exists():
        return (
            "Retrieval evaluation results "
            "have not been generated yet."
        )

    results = json.loads(
        RETRIEVAL_RESULTS.read_text(
            encoding="utf-8"
        )
    )

    overall = results.get(
        "overall",
        {}
    )

    return (
        f"- Recall@10: "
        f"{overall.get('recall@10', 0):.4f}\n"
        f"- MRR@10: "
        f"{overall.get('mrr@10', 0):.4f}\n"
        f"- p50 latency: "
        f"{overall.get('p50_latency_ms', 0):.2f} ms\n"
        f"- p99 latency: "
        f"{overall.get('p99_latency_ms', 0):.2f} ms\n"
        f"- No-answer correct: "
        f"{overall.get('no_answer_correct', 0)}/"
        f"{overall.get('no_answer_total', 0)}"
    )


def error_taxonomy(
    df,
    true_col,
    pred_col,
):
    errors = df[
        df[true_col]
        != df[pred_col]
    ].copy()

    if errors.empty:
        return (
            "No classification errors "
            "were found in this evaluation set."
        )

    pairs = (
        errors
        .groupby(
            [
                true_col,
                pred_col,
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    lines = [
        "| Gold label | Predicted label | Count |",
        "|---|---|---:|",
    ]

    for (
        gold,
        pred,
    ), count in pairs.items():
        lines.append(
            f"| {gold} | {pred} | {count} |"
        )

    return "\n".join(lines)


def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing evaluation file: "
            f"{PREDICTIONS_PATH}"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    true_col = pick_column(
        df,
        [
            "gold",
            "label",
            "true_label",
            "y_true",
        ],
    )

    pred_col = pick_column(
        df,
        [
            "pred",
            "prediction",
            "predicted_label",
            "y_pred",
        ],
    )

    text_col = pick_column(
        df,
        [
            "text",
            "feedback_text",
            "case_text",
        ],
    )

    point, low, high = (
        bootstrap_macro_f1(
            df[true_col],
            df[pred_col],
        )
    )

    accuracy = float(
        np.mean(
            df[true_col]
            == df[pred_col]
        )
    )

    slice_table = build_slice_table(
        df,
        true_col,
        pred_col,
        text_col,
    )

    taxonomy = error_taxonomy(
        df,
        true_col,
        pred_col,
    )

    retrieval = (
        load_retrieval_summary()
    )

    report = f"""# EVALUATION REPORT — Bayan

## Executive headline

The Bayan topic classifier achieved a macro-F1 of **{point:.4f}**
with a 95% bootstrap confidence interval of
**[{low:.4f}, {high:.4f}]**.

Overall validation accuracy was **{accuracy:.4f}**.
Slice-level results are reported below to highlight performance differences across language, dialect, class, and input length.

## Aggregate metrics

- Macro-F1: **{point:.4f}**
- 95% bootstrap CI: **[{low:.4f}, {high:.4f}]**
- Accuracy: **{accuracy:.4f}**
- Evaluation examples: **{len(df)}**

## Sliced metrics

{slice_table}

## Behavioural suite

Behavioural evaluation utilities are implemented for:

- invariance tests
- directional expectation tests
- minimum functionality tests

These checks complement aggregate metrics by testing expected model behaviour under controlled input transformations.

## Error taxonomy

The most common gold-to-predicted label confusions are:

{taxonomy}

### Prioritised fixes

1. Review the most frequent confused label pairs and add targeted training examples.
2. Improve under-performing Arabic or dialect-specific slices with balanced data.
3. Inspect long and ambiguous feedback samples and improve preprocessing or classification context.

## Retrieval quality

{retrieval}

## Known limitations

- Model performance depends on the quality and balance of the supplied training data.
- Dialect-specific slices may contain fewer samples than the aggregate evaluation set.
- Confidence intervals describe uncertainty on the available evaluation sample and do not guarantee production behaviour.
- Manual qualitative error analysis should be used alongside automated metrics before production deployment.
"""

    OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"Evaluation report written to "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Macro-F1: {point:.4f} "
        f"[{low:.4f}, {high:.4f}]"
    )


if __name__ == "__main__":
    main()
