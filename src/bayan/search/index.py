
"""Lab 5: FAISS index build."""

import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from bayan.preprocessing.core import preprocess, PREPROC_VERSION


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = "data/search/bayan_cases.csv"


def build_index(prefix="artifacts/case_index_v1", limit=None):
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    if limit is not None:
        df = df.head(limit).copy()

    texts = (
        df["case_text"]
        .fillna("")
        .astype(str)
        .map(preprocess)
        .tolist()
    )

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, f"{prefix}.faiss")

    df.to_csv(
        f"{prefix}_metadata.csv",
        index=False
    )

    manifest = {
        "model": MODEL_NAME,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": int(dim),
    }

    Path(f"{prefix}_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )

    return index
