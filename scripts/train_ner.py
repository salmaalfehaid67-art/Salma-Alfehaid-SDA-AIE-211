"""Lab 3B: fine-tune Bayan NER model with correct label alignment."""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from bayan.models.ner import align_labels


MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
DATA_PATH = "data/models/bayan_ner.conll"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Directory used to save the trained NER model.",
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


def read_conll(path):
    sentences = []
    labels = []

    current_tokens = []
    current_labels = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)

                    current_tokens = []
                    current_labels = []

                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid CoNLL line: {line}"
                )

            token, label = parts

            current_tokens.append(token)
            current_labels.append(label)

    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)

    return sentences, labels


def split_dataset(sentences, labels):
    n = len(sentences)

    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    return {
        "train": (
            sentences[:train_end],
            labels[:train_end],
        ),
        "validation": (
            sentences[train_end:val_end],
            labels[train_end:val_end],
        ),
        "test": (
            sentences[val_end:],
            labels[val_end:],
        ),
    }


def compute_metrics_factory(id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1,
        )

        true_predictions = []
        true_labels = []

        for prediction, label in zip(
            predictions,
            labels
        ):
            pred_tags = []
            gold_tags = []

            for pred_id, gold_id in zip(
                prediction,
                label
            ):
                if gold_id == -100:
                    continue

                pred_tags.append(
                    id2label[int(pred_id)]
                )

                gold_tags.append(
                    id2label[int(gold_id)]
                )

            true_predictions.append(
                pred_tags
            )

            true_labels.append(
                gold_tags
            )

        return {
            "precision":
                precision_score(
                    true_labels,
                    true_predictions,
                ),
            "recall":
                recall_score(
                    true_labels,
                    true_predictions,
                ),
            "f1":
                seqeval_f1(
                    true_labels,
                    true_predictions,
                ),
        }

    return compute_metrics


def main():
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sentences, tag_sequences = read_conll(
        DATA_PATH
    )

    all_labels = sorted({
        tag
        for sequence in tag_sequences
        for tag in sequence
    })

    label2id = {
        label: index
        for index, label
        in enumerate(all_labels)
    }

    id2label = {
        index: label
        for label, index
        in label2id.items()
    }

    splits = split_dataset(
        sentences,
        tag_sequences
    )

    dataset_dict = {}

    for split_name, (
        split_tokens,
        split_labels,
    ) in splits.items():

        numeric_labels = [
            [
                label2id[label]
                for label in sentence_labels
            ]
            for sentence_labels
            in split_labels
        ]

        dataset_dict[
            split_name
        ] = Dataset.from_dict(
            {
                "tokens": split_tokens,
                "ner_tags": numeric_labels,
            }
        )

    dataset = DatasetDict(
        dataset_dict
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=256,
        )

        aligned = []

        for i, word_labels in enumerate(
            examples["ner_tags"]
        ):
            word_ids = tokenized.word_ids(
                batch_index=i
            )

            aligned.append(
                align_labels(
                    word_ids,
                    word_labels,
                )
            )

        tokenized["labels"] = aligned

        return tokenized

    tokenized = dataset.map(
        tokenize_and_align,
        batched=True,
    )

    model = (
        AutoModelForTokenClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(all_labels),
            label2id=label2id,
            id2label=id2label,
        )
    )

    training_args = TrainingArguments(
        output_dir=str(
            output_dir
        ),
        learning_rate=2e-5,
        per_device_train_batch_size=
            args.batch_size,
        per_device_eval_batch_size=
            args.batch_size,
        num_train_epochs=
            args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=
            tokenized["train"],
        eval_dataset=
            tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=
            DataCollatorForTokenClassification(
                tokenizer=tokenizer
            ),
        compute_metrics=
            compute_metrics_factory(
                id2label
            ),
    )

    trainer.train()

    validation_metrics = (
        trainer.evaluate(
            tokenized["validation"]
        )
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
        "labels": all_labels,
        "validation":
            validation_metrics,
        "test":
            test_metrics,
    }

    (
        output_dir /
        "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "NER training complete."
    )

    print(
        f"Saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
