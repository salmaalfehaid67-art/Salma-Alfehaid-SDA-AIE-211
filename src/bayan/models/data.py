"""Lab 3A: leakage-safe grouped dataset splits."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def build_topic_dataset(
    csv_path="data/raw/bayan_feedback.csv",
    group_col="citizen_group_id",
    test_size=0.20,
    val_size=0.10,
    random_state=42,
):
    df = pd.read_csv(csv_path)

    if group_col not in df.columns:
        raise ValueError(
            f"Missing group column: {group_col}. "
            f"Available columns: {list(df.columns)}"
        )

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_val_idx, test_idx = next(
        first_split.split(
            df,
            groups=df[group_col]
        )
    )

    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    relative_val_size = val_size / (1.0 - test_size)

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_val_size,
        random_state=random_state,
    )

    train_idx, val_idx = next(
        second_split.split(
            train_val,
            groups=train_val[group_col]
        )
    )

    train = train_val.iloc[train_idx].copy()
    validation = train_val.iloc[val_idx].copy()

    train_groups = set(train[group_col])
    validation_groups = set(validation[group_col])
    test_groups = set(test[group_col])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)

    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
