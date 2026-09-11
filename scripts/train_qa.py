"""Train Bayan extractive QA model."""

import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "bert-base-multilingual-cased"
DATA_PATH = Path("data/models/bayan_qa.json")
OUTPUT_DIR = Path("artifacts/qa")


def load_examples():
    raw = json.loads(
        DATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    rows = []

    for item in raw["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                answers = qa.get(
                    "answers",
                    []
                )

                rows.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answers": {
                        "text": [
                            a["text"]
                            for a in answers
                        ],
                        "answer_start": [
                            a["answer_start"]
                            for a in answers
                        ],
                    },
                    "is_impossible": qa.get(
                        "is_impossible",
                        False
                    ),
                })

    return rows


def prepare_features(examples, tokenizer):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=256,
        stride=64,
        return_offsets_mapping=True,
        padding=False,
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(
        tokenized["offset_mapping"]
    ):
        sequence_ids = tokenized.sequence_ids(i)

        answers = examples["answers"][i]

        if len(answers["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answers["answer_start"][0]
        end_char = (
            start_char
            + len(
                answers["text"][0]
            )
        )

        context_start = 0

        while (
            sequence_ids[context_start] != 1
        ):
            context_start += 1

        context_end = (
            len(sequence_ids) - 1
        )

        while (
            sequence_ids[context_end] != 1
        ):
            context_end -= 1

        if (
            offsets[context_start][0] > start_char
            or offsets[context_end][1] < end_char
        ):
            start_positions.append(0)
            end_positions.append(0)
            continue

        token_start = context_start

        while (
            token_start <= context_end
            and offsets[token_start][0]
            <= start_char
        ):
            token_start += 1

        token_end = context_end

        while (
            token_end >= context_start
            and offsets[token_end][1]
            >= end_char
        ):
            token_end -= 1

        start_positions.append(
            token_start - 1
        )

        end_positions.append(
            token_end + 1
        )

    tokenized["start_positions"] = (
        start_positions
    )
    tokenized["end_positions"] = (
        end_positions
    )

    tokenized.pop(
        "offset_mapping"
    )

    return tokenized


def main():
    rows = load_examples()

    rng = np.random.default_rng(42)
    indices = np.arange(len(rows))
    rng.shuffle(indices)

    split = int(
        0.9 * len(rows)
    )

    train_rows = [
        rows[i]
        for i in indices[:split]
    ]

    val_rows = [
        rows[i]
        for i in indices[split:]
    ]

    train_ds = Dataset.from_list(
        train_rows
    )

    val_ds = Dataset.from_list(
        val_rows
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True,
        )
    )

    train_tok = train_ds.map(
        lambda x: prepare_features(
            x,
            tokenizer,
        ),
        batched=True,
        remove_columns=train_ds.column_names,
    )

    val_tok = val_ds.map(
        lambda x: prepare_features(
            x,
            tokenizer,
        ),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    model = (
        AutoModelForQuestionAnswering
        .from_pretrained(
            MODEL_NAME
        )
    )

    args = TrainingArguments(
        output_dir=str(
            OUTPUT_DIR
        ),
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer
        ),
    )

    trainer.train()

    metrics = trainer.evaluate()

    trainer.save_model(
        str(
            OUTPUT_DIR
        )
    )

    tokenizer.save_pretrained(
        str(
            OUTPUT_DIR
        )
    )

    (
        OUTPUT_DIR
        / "train_metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "QA training complete."
    )
    print(
        f"Saved to: {OUTPUT_DIR}"
    )
    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
