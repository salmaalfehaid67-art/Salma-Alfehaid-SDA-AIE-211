"""Lab 5: two-stage semantic search."""

import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

from bayan.preprocessing.core import preprocess, PREPROC_VERSION


RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CaseSearch:
    def __init__(self, prefix: str):
        self.prefix = Path(prefix)

        manifest_path = Path(f"{prefix}_manifest.json")
        index_path = Path(f"{prefix}.faiss")
        metadata_path = Path(f"{prefix}_metadata.csv")

        self.manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if self.manifest["preproc_version"] != PREPROC_VERSION:
            raise ValueError("Preprocessing version mismatch")

        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_csv(metadata_path)

        self.encoder = SentenceTransformer(
            self.manifest["model"]
        )

        self.reranker = CrossEncoder(
            RERANKER_NAME
        )

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
    ):
        cleaned = preprocess(query)

        if not cleaned.strip():
            return []

        q_emb = self.encoder.encode(
            [cleaned],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        n_candidates = min(
            candidates,
            self.index.ntotal
        )

        scores, indices = self.index.search(
            q_emb,
            n_candidates
        )

        rows = []

        for bi_score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            row = self.metadata.iloc[int(idx)].to_dict()

            rows.append({
                "index": int(idx),
                "case_id": row.get("case_id"),
                "text": str(row["case_text"]),
                "bi_score": float(bi_score),
                "metadata": row,
            })

        if not rows:
            return []

        pairs = [
            [cleaned, item["text"]]
            for item in rows
        ]

        ce_scores = self.reranker.predict(pairs)

        for item, score in zip(rows, ce_scores):
            item["rerank_score"] = float(score)

        rows.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        rows = [
            item
            for item in rows
            if item["rerank_score"] >= min_score
        ]

        return rows[:k]
