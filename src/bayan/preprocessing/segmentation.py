"""Lab 1: sentence segmentation."""

import spacy

from bayan.preprocessing.core import preprocess


def build_pipeline():
    """Build a lightweight multilingual sentence segmentation pipeline."""
    nlp = spacy.blank("xx")

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    """Preprocess text, split into sentences, and return non-empty strings."""
    cleaned = preprocess(raw)
    doc = nlp(cleaned)

    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]
