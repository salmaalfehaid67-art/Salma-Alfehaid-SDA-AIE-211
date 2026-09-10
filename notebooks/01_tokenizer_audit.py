"""Lab 1: tokenizer audit."""

from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    """Average number of subword pieces per whitespace word."""
    total_pieces = 0
    total_words = 0

    for text in texts:
        text = str(text)
        words = text.split()

        if not words:
            continue

        pieces = tokenizer.tokenize(text)

        total_pieces += len(pieces)
        total_words += len(words)

    return total_pieces / total_words if total_words else 0.0


def main():
    df = pd.read_csv(DATA)

    text_col = (
        "text"
        if "text" in df.columns
        else df.select_dtypes(include="object").columns[0]
    )

    lang_col = next(
        (
            col
            for col in ["language", "lang", "locale"]
            if col in df.columns
        ),
        None,
    )

    if lang_col is None:
        raise ValueError(
            f"Could not find a language column. "
            f"Available columns: {list(df.columns)}"
        )

    results = []

    for model_name, display_name in CANDIDATES.items():
        print(f"Loading {display_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for lang in df[lang_col].dropna().unique():
            texts = (
                df.loc[df[lang_col] == lang, text_col]
                .dropna()
                .astype(str)
                .tolist()
            )

            fert = fertility(tokenizer, texts)

            lengths = [
                len(
                    tokenizer.encode(
                        text,
                        add_special_tokens=True,
                    )
                )
                for text in texts
            ]

            results.append(
                {
                    "tokenizer": display_name,
                    "language": lang,
                    "fertility": round(fert, 3),
                    "avg_seq_len": (
                        round(float(np.mean(lengths)), 2)
                        if lengths
                        else 0.0
                    ),
                    "p95_seq_len": (
                        round(float(np.percentile(lengths, 95)), 2)
                        if lengths
                        else 0.0
                    ),
                }
            )

    result_df = pd.DataFrame(results)

    print("\n=== TOKENIZER AUDIT ===")
    print(result_df.to_string(index=False))

    result_df.to_csv(
        "tokenizer_audit_results.csv",
        index=False,
    )

    print("\nSaved: tokenizer_audit_results.csv")


if __name__ == "__main__":
    main()
