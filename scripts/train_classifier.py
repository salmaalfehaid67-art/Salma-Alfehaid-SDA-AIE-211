"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    return parser.parse_args()


def pick_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Expected one of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        ),
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = build_topic_dataset()

    text_col = pick_column(
        splits["train"],
        [
            "text",
            "feedback_text",
            "case_text",
            "comment",
        ],
    )

    label_col = pick_column(
        splits["train"],
        [
            "topic",
            "label",
            "category",
        ],
    )

    labels = sorted(
        splits["train"][label_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    label2id = {
        label: index
        for index, label in enumerate(labels)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    datasets = {}

    for split_name, df in splits.items():
        clean = df[
            [text_col, label_col]
        ].copy()

        clean[text_col] = (
            clean[text_col]
            .fillna("")
            .astype(str)
        )

        clean[label_col] = (
            clean[label_col]
            .astype(str)
        )

        clean["labels"] = (
            clean[label_col]
            .map(label2id)
        )

        clean = clean.rename(
            columns={
                text_col: "text"
            }
        )

        datasets[split_name] = Dataset.from_pandas(
            clean[
                ["text", "labels"]
            ],
            preserve_index=False,
        )

    dataset = DatasetDict(datasets)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
        )
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer
        ),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    validation_metrics = trainer.evaluate(
        tokenized["validation"]
    )

    test_metrics = trainer.evaluate(
        tokenized["test"]
    )

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    metrics = {
        "model": MODEL_NAME,
        "validation": validation_metrics,
        "test": test_metrics,
        "labels": labels,
    }

    (
        output_dir / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Classifier training complete.")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
