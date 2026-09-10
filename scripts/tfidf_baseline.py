"""Lab 3A: TF-IDF + LinearSVC baseline for Bayan topic classification."""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bayan.models.data import build_topic_dataset
from bayan.preprocessing.core import preprocess


def _pick_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"None of the expected columns {candidates} were found. "
        f"Available columns: {list(df.columns)}"
    )


def main():
    splits = build_topic_dataset()

    text_col = _pick_column(
        splits["train"],
        ["text", "feedback_text", "case_text", "comment"],
    )

    label_col = _pick_column(
        splits["train"],
        ["topic", "label", "category"],
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced"
                ),
            ),
        ]
    )

    model.fit(
        splits["train"][text_col].fillna(""),
        splits["train"][label_col],
    )

    metrics = {}

    for split_name in ["validation", "test"]:
        df = splits[split_name]

        y_true = df[label_col]
        y_pred = model.predict(
            df[text_col].fillna("")
        )

        metrics[split_name] = {
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                )
            ),
            "accuracy": float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "classification_report":
                classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                ),
        }

        print(
            f"{split_name}: "
            f"macro-F1="
            f"{metrics[split_name]['macro_f1']:.4f} "
            f"accuracy="
            f"{metrics[split_name]['accuracy']:.4f}"
        )

    out_dir = Path("artifacts")
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        out_dir /
        "tfidf_baseline_metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
