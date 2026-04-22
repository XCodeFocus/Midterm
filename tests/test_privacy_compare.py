import pandas as pd

from privacy_compare import compare_datasets, normalize_label_column


def test_normalize_label_column():
    df = pd.DataFrame({"income>50K": ["<=50K", ">50K"]})
    out = normalize_label_column(df)

    assert "income" in out.columns
    assert "income>50K" not in out.columns


def test_compare_datasets_shared_columns():
    left = pd.DataFrame(
        {
            "age": ["[20,21]", "[22,23]"],
            "sex": ["{Male}", "{Female}"],
            "income": ["<=50K", ">50K"],
        }
    )
    right = pd.DataFrame(
        {
            "age": [5, 6],
            "sex": [0, 1],
            "income>50K": [0, 1],
        }
    )

    result = compare_datasets(left, right)

    assert set(result.keys()) == {"shared_columns", "label_balance"}
    assert list(result["shared_columns"]["column"]) == ["age", "sex"]
    assert len(result["label_balance"]) == 2
    assert result["label_balance"].loc[result["label_balance"]["dataset"] == "left", "positive_rate"].iloc[0] == 0.5
    assert result["label_balance"].loc[result["label_balance"]["dataset"] == "right", "positive_rate"].iloc[0] == 0.5