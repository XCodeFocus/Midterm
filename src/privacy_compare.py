from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "income" not in out.columns and "income>50K" in out.columns:
        out = out.rename(columns={"income>50K": "income"})
    return out


def _binary_label_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0).astype(int) > 0).astype(int)

    values = series.astype(str).str.strip()
    if values.isin({"0", "1"}).all():
        return values.astype(int)

    return (values == ">50K").astype(int)


def shared_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    exclude: Iterable[str] = (),
) -> list[str]:
    excluded = set(exclude)
    return [col for col in left.columns if col in right.columns and col not in excluded]


def _top_share(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    values = series.astype(str).str.strip().replace({"nan": np.nan})
    counts = values.value_counts(dropna=False)
    return float(counts.iloc[0] / len(values)) if not counts.empty else float("nan")


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_shared_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if columns is None:
        columns = shared_columns(left, right)

    rows: list[dict[str, object]] = []
    for col in columns:
        left_s = left[col]
        right_s = right[col]
        row: dict[str, object] = {
            "column": col,
            "left_unique": int(left_s.astype(str).nunique(dropna=False)),
            "right_unique": int(right_s.astype(str).nunique(dropna=False)),
            "left_missing_rate": float(left_s.isna().mean()),
            "right_missing_rate": float(right_s.isna().mean()),
            "left_top_share": _top_share(left_s),
            "right_top_share": _top_share(right_s),
        }

        if pd.api.types.is_numeric_dtype(left_s) and pd.api.types.is_numeric_dtype(right_s):
            left_num = _numeric_summary(left_s)
            right_num = _numeric_summary(right_s)
            row.update(
                {
                    "left_mean": left_num["mean"],
                    "right_mean": right_num["mean"],
                    "left_std": left_num["std"],
                    "right_std": right_num["std"],
                    "mean_abs_diff": abs(left_num["mean"] - right_num["mean"]),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_label_balance(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    label_column: str = "income",
) -> pd.DataFrame:
    if label_column not in left.columns or label_column not in right.columns:
        raise KeyError(f"label column not found in both dataframes: {label_column}")

    rows = []
    for name, df in [("left", left), ("right", right)]:
        s = _binary_label_series(df[label_column])
        rows.append(
            {
                "dataset": name,
                "label_column": label_column,
                "rows": int(len(df)),
                "positive_rate": float(s.mean()),
                "unique_labels": int(s.nunique(dropna=False)),
            }
        )
    return pd.DataFrame(rows)


def compare_datasets(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    exclude: Iterable[str] = (),
    label_column: str = "income",
) -> dict[str, pd.DataFrame]:
    left = normalize_label_column(left)
    right = normalize_label_column(right)

    base_exclude = set(exclude) | {"eq_class_id", "eq_class_size"}
    if label_column in left.columns and label_column in right.columns:
        label_stats = summarize_label_balance(left, right, label_column=label_column)
        base_exclude.add(label_column)
    else:
        label_stats = pd.DataFrame()

    shared = shared_columns(left, right, exclude=base_exclude)
    shared_stats = summarize_shared_columns(left, right, columns=shared)
    return {
        "shared_columns": shared_stats,
        "label_balance": label_stats,
    }


def write_comparison_report(result: dict[str, pd.DataFrame], output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name, frame in result.items():
        path = output_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare K-anonymity and DP outputs")
    parser.add_argument("--left", required=True, help="Path to the left CSV, usually the K-anonymity output")
    parser.add_argument("--right", required=True, help="Path to the right CSV, usually the DP synthetic output")
    parser.add_argument("--output-dir", default="outputs/comparison", help="Directory to write comparison CSVs")
    args = parser.parse_args(argv)

    left = load_csv(args.left)
    right = load_csv(args.right)
    result = compare_datasets(left, right)
    written = write_comparison_report(result, args.output_dir)

    for name, path in written.items():
        print(f"saved {name}: {path}")
    print(result["shared_columns"])
    if not result["label_balance"].empty:
        print(result["label_balance"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())