"""Lab 3B: extractive QA post-processing."""

import numpy as np


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)

    start_candidates = np.argsort(start_logits)[-top_k:][::-1]
    end_candidates = np.argsort(end_logits)[-top_k:][::-1]

    best_score = float("-inf")
    best = None

    for start_idx in start_candidates:
        if offsets[start_idx] is None:
            continue

        for end_idx in end_candidates:
            if offsets[end_idx] is None:
                continue

            if end_idx < start_idx:
                continue

            if end_idx - start_idx + 1 > max_answer_len:
                continue

            start_char = offsets[start_idx][0]
            end_char = offsets[end_idx][1]

            if end_char <= start_char:
                continue

            score = float(
                start_logits[start_idx]
                + end_logits[end_idx]
            )

            if score > best_score:
                best_score = score
                best = {
                    "answer": (start_char, end_char),
                    "score": score,
                    "start_index": int(start_idx),
                    "end_index": int(end_idx),
                }

    if best is None:
        return {
            "answer": None,
            "score": float(null_score),
            "start_index": None,
            "end_index": None,
        }

    if float(null_score) - best_score >= float(null_threshold):
        return {
            "answer": None,
            "score": float(null_score),
            "start_index": None,
            "end_index": None,
        }

    return best
