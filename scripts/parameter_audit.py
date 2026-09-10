"""Lab 2: parameter accounting for mBERT and CAMeLBERT."""

from transformers import AutoModel


def classify_parameter(name: str) -> str:
    name = name.lower()

    if "embeddings" in name:
        return "embeddings"

    if (
        "attention" in name
        or "query" in name
        or "key" in name
        or "value" in name
    ):
        return "attention"

    if (
        "intermediate" in name
        or "output.dense" in name
    ):
        return "ffn"

    if (
        "layernorm" in name
        or "layer_norm" in name
    ):
        return "norms"

    if "pooler" in name:
        return "pooler"

    return "other"


def audit(checkpoint: str) -> dict:
    model = AutoModel.from_pretrained(
        checkpoint
    )

    buckets = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0,
    }

    total = 0

    for name, parameter in model.named_parameters():
        count = parameter.numel()

        total += count

        bucket = classify_parameter(
            name
        )

        buckets[bucket] += count

    result = {
        "checkpoint": checkpoint,
        "total_parameters": total,
        "buckets": {},
    }

    for bucket, count in buckets.items():
        percentage = (
            100.0 * count / total
            if total
            else 0.0
        )

        result["buckets"][bucket] = {
            "parameters": count,
            "percentage": round(
                percentage,
                2,
            ),
        }

    return result


def print_report(result):
    print(
        "\nCheckpoint:",
        result["checkpoint"],
    )

    print(
        "Total parameters:",
        f"{result['total_parameters']:,}",
    )

    print(
        "-" * 55
    )

    for bucket, values in result[
        "buckets"
    ].items():

        print(
            f"{bucket:12s}"
            f"{values['parameters']:>15,}"
            f"   "
            f"{values['percentage']:>6.2f}%"
        )


if __name__ == "__main__":
    checkpoints = [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]

    for checkpoint in checkpoints:
        result = audit(
            checkpoint
        )

        print_report(
            result
        )
