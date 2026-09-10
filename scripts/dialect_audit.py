"""Lab 4 starter: audit dialect mix and record the implication in NOTES.md."""


def main():
    # TODO(Lab 4): run the dialect audit over the Arabic slice and print region distribution.
    raise NotImplementedError("Complete the dialect audit")


if __name__ == "__main__":
    main()
"""Lab 4: dialect distribution audit."""

from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    ar_df = df[df["lang"] == "ar"].copy()

    counts = (
        ar_df["dialect_region"]
        .fillna("UNKNOWN")
        .value_counts()
    )

    percentages = (
        ar_df["dialect_region"]
        .fillna("UNKNOWN")
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages
    })

    print("=== LAB 4 DIALECT AUDIT ===")
    print(result)

    print("\nImplication:")
    print(
        "Arabic evaluation should include dialect-aware slices "
        "rather than relying only on one overall Arabic score."
    )


if __name__ == "__main__":
    main()
