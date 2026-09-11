%%writefile notebooks/05_retrieval_eval.py
"""Lab 5: labelled-query retrieval evaluation."""

import json
from pathlib import Path

import numpy as np

from bayan.preprocessing.core import preprocess
from bayan.search.service import CaseSearch


QUERY_FILE = Path("data/search/bayan_queries.jsonl")
INDEX_PREFIX = "artifacts/case_index_v1"
OUTPUT_FILE = Path("artifacts/retrieval_eval.json")

K = 10
CANDIDATES = 50


def load_queries():
    rows = []

    with QUERY_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def reciprocal_rank(case_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)

    for rank, case_id in enumerate(case_ids[:k], start=1):
        if case_id in relevant:
            return 1.0 / rank

    return 0.0


def recall_at_k(case_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)

    if not relevant:
        return None

    retrieved = set(case_ids[:k])

    return len(retrieved & relevant) / len(relevant)


def retrieve(searcher, query):
    cleaned = preprocess(query)

    q_emb = searcher.encoder.encode(
        [cleaned],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    n_candidates = min(
        CANDIDATES,
        searcher.index.ntotal,
    )

    bi_scores, indices = searcher.index.search(
        q_emb,
        n_candidates,
    )

    candidates = []

    for score, idx in zip(
        bi_scores[0],
        indices[0],
    ):
        if idx < 0:
            continue

        row = searcher.metadata.iloc[int(idx)]

        candidates.append({
            "case_id": str(row["case_id"]),
            "text": str(row["case_text"]),
            "bi_score": float(score),
        })

    bi_ids = [
        item["case_id"]
        for item in candidates[:K]
    ]

    if not candidates:
        return bi_ids, [], None

    pairs = [
        [cleaned, item["text"]]
        for item in candidates
    ]

    rerank_scores = searcher.reranker.predict(
        pairs
    )

    for item, score in zip(
        candidates,
        rerank_scores,
    ):
        item["rerank_score"] = float(score)

    candidates.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    reranked_ids = [
        item["case_id"]
        for item in candidates[:K]
    ]

    best_score = float(
        candidates[0]["rerank_score"]
    )

    return bi_ids, reranked_ids, best_score


def summarize(rows, key):
    valid = [
        row
        for row in rows
        if not row["no_answer"]
    ]

    recalls = [
        row[key]["recall@10"]
        for row in valid
    ]

    mrrs = [
        row[key]["mrr@10"]
        for row in valid
    ]

    return {
        "n_queries": len(valid),
        "recall@10": float(np.mean(recalls))
        if recalls else 0.0,
        "mrr@10": float(np.mean(mrrs))
        if mrrs else 0.0,
    }


def tune_no_answer_threshold(rows):
    scores = [
        row["best_rerank_score"]
        for row in rows
        if row["best_rerank_score"] is not None
    ]

    if not scores:
        return 0.0, 0.0

    low = min(scores)
    high = max(scores)

    thresholds = np.linspace(
        low,
        high,
        200,
    )

    best_threshold = thresholds[0]
    best_accuracy = -1.0

    for threshold in thresholds:
        correct = 0

        for row in rows:
            score = row["best_rerank_score"]

            predicted_no_answer = (
                score is None
                or score < threshold
            )

            if predicted_no_answer == row["no_answer"]:
                correct += 1

        accuracy = correct / len(rows)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_accuracy),
    )


def main():
    queries = load_queries()

    searcher = CaseSearch(
        INDEX_PREFIX
    )

    results = []

    for i, item in enumerate(
        queries,
        start=1,
    ):
        print(
            f"[{i}/{len(queries)}] "
            f"{item['query_id']}"
        )

        bi_ids, reranked_ids, best_score = retrieve(
            searcher,
            item["query"],
        )

        relevant = item[
            "relevant_case_ids"
        ]

        no_answer = bool(
            item.get(
                "no_answer",
                False,
            )
        )

        result = {
            "query_id": item["query_id"],
            "lang": item.get("lang", "unknown"),
            "topic": item.get("topic"),
            "no_answer": no_answer,
            "best_rerank_score": best_score,
            "bi_encoder": {
                "recall@10": (
                    None
                    if no_answer
                    else recall_at_k(
                        bi_ids,
                        relevant,
                        K,
                    )
                ),
                "mrr@10": (
                    None
                    if no_answer
                    else reciprocal_rank(
                        bi_ids,
                        relevant,
                        K,
                    )
                ),
            },
            "reranked": {
                "recall@10": (
                    None
                    if no_answer
                    else recall_at_k(
                        reranked_ids,
                        relevant,
                        K,
                    )
                ),
                "mrr@10": (
                    None
                    if no_answer
                    else reciprocal_rank(
                        reranked_ids,
                        relevant,
                        K,
                    )
                ),
            },
        }

        results.append(result)

    overall = {
        "bi_encoder": summarize(
            results,
            "bi_encoder",
        ),
        "reranked": summarize(
            results,
            "reranked",
        ),
    }

    languages = sorted(
        set(
            row["lang"]
            for row in results
        )
    )

    by_language = {}

    for lang in languages:
        subset = [
            row
            for row in results
            if row["lang"] == lang
        ]

        by_language[lang] = {
            "bi_encoder": summarize(
                subset,
                "bi_encoder",
            ),
            "reranked": summarize(
                subset,
                "reranked",
            ),
        }

    threshold, threshold_accuracy = (
        tune_no_answer_threshold(
            results
        )
    )

    output = {
        "overall": overall,
        "by_language": by_language,
        "no_answer": {
            "best_min_score": threshold,
            "classification_accuracy": threshold_accuracy,
            "n_no_answer": sum(
                row["no_answer"]
                for row in results
            ),
        },
        "queries": results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== Overall ===")
    print(
        json.dumps(
            overall,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n=== By language ===")
    print(
        json.dumps(
            by_language,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n=== No-answer threshold ===")
    print(
        f"best_min_score = "
        f"{threshold:.4f}"
    )
    print(
        f"accuracy = "
        f"{threshold_accuracy:.4f}"
    )

    print(
        "\nSaved to "
        "artifacts/retrieval_eval.json"
    )


if __name__ == "__main__":
    main()
