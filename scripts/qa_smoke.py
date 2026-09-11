"""Lab 3B: run the Bayan QA smoke set."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span


MODEL_DIR = Path("artifacts/qa")
SMOKE_SET = Path("data/eval/qa_smoke_set.json")


def load_smoke_set():
    raw = json.loads(
        SMOKE_SET.read_text(
            encoding="utf-8"
        )
    )

    examples = []

    for item in raw["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                answers = qa.get("answers", [])

                if (
                    qa.get("is_impossible", False)
                    or not answers
                ):
                    expected_answer = None
                else:
                    expected_answer = answers[0]["text"]

                examples.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answer": expected_answer,
                    "is_impossible": qa.get(
                        "is_impossible",
                        False
                    ),
                })

    return examples


def extract_answer(
    tokenizer,
    model,
    question,
    context,
    null_threshold=0.0,
):
    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
        stride=128,
        return_offsets_mapping=True,
    )

    offsets = encoded.pop(
        "offset_mapping"
    )[0].tolist()

    sequence_ids = encoded.sequence_ids(
        0
    )

    context_offsets = []

    for seq_id, offset in zip(
        sequence_ids,
        offsets
    ):
        if seq_id == 1:
            context_offsets.append(
                tuple(offset)
            )
        else:
            context_offsets.append(
                None
            )

    with torch.no_grad():
        outputs = model(
            **encoded
        )

    start_logits = (
        outputs.start_logits[0]
        .cpu()
        .numpy()
    )

    end_logits = (
        outputs.end_logits[0]
        .cpu()
        .numpy()
    )

    cls_index = 0

    null_score = float(
        start_logits[cls_index]
        + end_logits[cls_index]
    )

    result = best_span(
        start_logits,
        end_logits,
        context_offsets,
        null_score=null_score,
        null_threshold=
            null_threshold,
        max_answer_len=30,
        top_k=20,
    )

    if result["answer"] is None:
        return None

    start_char, end_char = (
        result["answer"]
    )

    return context[
        start_char:end_char
    ]


def normalize_text(text):
    if text is None:
        return None

    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            "QA model not found at "
            f"{MODEL_DIR}. "
            "Place the provided QA "
            "checkpoint in artifacts/qa."
        )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_DIR
        )
    )

    model = (
        AutoModelForQuestionAnswering
        .from_pretrained(
            MODEL_DIR
        )
    )

    model.eval()

    examples = load_smoke_set()

    total = 0
    passed = 0
    answerable_total = 0
    answerable_passed = 0
    null_total = 0
    null_passed = 0

    details = []

    for example in examples:
        question = example[
            "question"
        ]

        context = example[
            "context"
        ]

        expected = example.get(
            "answer"
        )

        predicted = extract_answer(
            tokenizer,
            model,
            question,
            context,
        )

        expected_norm = (
            normalize_text(
                expected
            )
        )

        predicted_norm = (
            normalize_text(
                predicted
            )
        )

        is_null = (
            expected is None
            or expected == ""
        )

        if is_null:
            null_total += 1

            ok = (
                predicted is None
                or predicted == ""
            )

            if ok:
                null_passed += 1

        else:
            answerable_total += 1

            ok = (
                expected_norm
                == predicted_norm
            )

            if ok:
                answerable_passed += 1

        total += 1

        if ok:
            passed += 1

        details.append(
            {
                "question":
                    question,
                "expected":
                    expected,
                "predicted":
                    predicted,
                "passed":
                    ok,
            }
        )

    results = {
        "total":
            total,
        "passed":
            passed,
        "answerable_total":
            answerable_total,
        "answerable_passed":
            answerable_passed,
        "null_total":
            null_total,
        "null_passed":
            null_passed,
        "details":
            details,
    }

    out_path = Path(
        "artifacts/"
        "qa_smoke_results.json"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"QA smoke: "
        f"{passed}/{total} passed"
    )

    print(
        "Answerable: "
        f"{answerable_passed}/"
        f"{answerable_total}"
    )

    print(
        "Null cases: "
        f"{null_passed}/"
        f"{null_total}"
    )


if __name__ == "__main__":
    main()
