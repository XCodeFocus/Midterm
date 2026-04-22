import pandas as pd

from k_anonymity.mondrian import (
    assert_k_anonymous,
    encode_raw_as_generalized,
    mondrian_anonymize,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20, 21, 22, 40, 41, 42],
            "hours-per-week": [40, 41, 42, 50, 51, 52],
            "sex": ["Male", "Male", "Female", "Male", "Female", "Female"],
            "workclass": ["Private", "Private", "Private", "Self-emp", "Self-emp", "Self-emp"],
            "income": ["<=50K", "<=50K", "<=50K", ">50K", ">50K", ">50K"],
        }
    )


def test_encode_raw_schema():
    df = _toy_df()
    out = encode_raw_as_generalized(
        df,
        qi_columns=["age", "hours-per-week", "sex", "workclass"],
        numeric_qi=["age", "hours-per-week"],
        categorical_qi=["sex", "workclass"],
        label_column="income",
    )

    assert "age_mid" in out.columns
    assert "age_width" in out.columns
    assert out["age_width"].eq(0).all()
    assert out["sex"].str.startswith("{").all()


def test_mondrian_k_anonymous():
    df = _toy_df()
    res = mondrian_anonymize(
        df,
        k=2,
        qi_columns=["age", "hours-per-week", "sex", "workclass"],
        numeric_qi=["age", "hours-per-week"],
        categorical_qi=["sex", "workclass"],
        label_column="income",
    )

    assert_k_anonymous(res.df_anonymized, k=2)
    assert (res.df_anonymized["eq_class_size"] >= 2).all()

    # generalized formats
    assert res.df_anonymized["age"].astype(str).str.startswith("[").all()
    assert res.df_anonymized["sex"].astype(str).str.startswith("{").all()
