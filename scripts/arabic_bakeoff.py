"""Lab 4: compare Arabic-centric checkpoints by evaluation slices."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


DEFAULT_DATA = Path(
    "data/eval/validation_predictions.csv"
)

OUTPUT_PATH = Path(
    "artifacts/arabic_bakeoff.json"
)


CHECKPOINTS = {
    "CAMeLBERT-mix":
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA":
        "CAMeL-Lab/bert-base-arabic-camelbert-da",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
    )

    return parser.parse_args()


def pick_column(
    df,
    candidates,
    required=True,
):
    for name in candidates:
        if name in df.columns:
            return name

    if required:
        raise ValueError(
            f"None of {candidates} found. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def macro_f1(
    y_true,
    y_pred,
):
    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )


def evaluate_checkpoint(
    checkpoint,
    df,
    text_col,
    label_col,
    lang_col,
    dialect_col,
):
    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            checkpoint
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            checkpoint
        )
    )

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
    )

    texts = (
        df[text_col]
        .fillna("")
        .astype(str)
        .tolist()
    )

    start = time.perf_counter()

    predictions = classifier(
        texts,
        truncation=True,
        max_length=256,
        batch_size=8,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    predicted_labels = [
        item["label"]
        for item in predictions
    ]

    result = {
        "n_samples":
            len(df),

        "elapsed_seconds":
            elapsed,

        "samples_per_second":
            (
                len(df) / elapsed
                if elapsed > 0
                else 0.0
            ),
    }

    gold = (
        df[label_col]
        .astype(str)
        .tolist()
    )

    result["macro_f1_all"] = (
        macro_f1(
            gold,
            predicted_labels,
        )
    )

    if lang_col:
        ar_mask = (
            df[lang_col]
            .astype(str)
            .str.lower()
            == "ar"
        )

        if ar_mask.any():
            result[
                "macro_f1_arabic"
            ] = macro_f1(
                df.loc[
                    ar_mask,
                    label_col
                ].astype(str),
                np.array(
                    predicted_labels
                )[ar_mask],
            )

    if dialect_col:
        dialect_series = (
            df[dialect_col]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        gulf_mask = (
            dialect_series
            .str.contains(
                "gulf|khaleeji|gcc",
                regex=True,
            )
        )

        msa_mask = (
            dialect_series
            .str.contains(
                "msa|modern standard",
                regex=True,
            )
        )

        if gulf_mask.any():
            result[
                "macro_f1_gulf"
            ] = macro_f1(
                df.loc[
                    gulf_mask,
                    label_col
                ].astype(str),
                np.array(
                    predicted_labels
                )[gulf_mask],
            )

        if msa_mask.any():
            result[
                "macro_f1_msa"
            ] = macro_f1(
                df.loc[
                    msa_mask,
                    label_col
                ].astype(str),
                np.array(
                    predicted_labels
                )[msa_mask],
            )

    return result


def main():
    args = parse_args()

    data_path = Path(
        args.data
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Evaluation data not found: "
            f"{data_path}"
        )

    df = pd.read_csv(
        data_path
    )

    lang_col = pick_column(
        df,
        ["lang", "language"],
        required=False,
    )

    if lang_col:
        df = df[
            df[lang_col]
            .astype(str)
            .str.lower()
            .eq("ar")
        ].copy()

    if args.max_samples:
        df = df.head(
            args.max_samples
        ).copy()

    text_col = pick_column(
        df,
        [
            "text",
            "feedback_text",
            "case_text",
        ],
    )

    label_col = pick_column(
        df,
        [
            "gold",
            "label",
            "true_label",
            "topic",
        ],
    )

    dialect_col = pick_column(
        df,
        [
            "dialect_region",
            "dialect",
        ],
        required=False,
    )

    results = {}

    for name, checkpoint in (
        CHECKPOINTS.items()
    ):
        print(
            f"Evaluating {name}..."
        )

        try:
            results[name] = (
                evaluate_checkpoint(
                    checkpoint,
                    df,
                    text_col,
                    label_col,
                    lang_col,
                    dialect_col,
                )
            )

        except Exception as exc:
            results[name] = {
                "error": str(exc)
            }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
