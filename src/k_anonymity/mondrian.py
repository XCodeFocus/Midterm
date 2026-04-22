from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MondrianResult:
    df_anonymized: pd.DataFrame
    partitions: List[List[int]]
    eq_class_id: pd.Series


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip() in {"", "?"}:
        return True
    return bool(pd.isna(x))


def _format_interval(lo: float, hi: float) -> str:
    # Prefer integer-like formatting when safe.
    def fmt(v: float) -> str:
        if float(v).is_integer():
            return str(int(v))
        return f"{v:.6g}"

    return f"[{fmt(lo)},{fmt(hi)}]"


def _format_set(values: Iterable[str]) -> str:
    vals = sorted({str(v) for v in values})
    return "{" + "|".join(vals) + "}"


def mondrian_anonymize(
    df: pd.DataFrame,
    *,
    k: int,
    qi_columns: Sequence[str],
    categorical_qi: Sequence[str],
    numeric_qi: Sequence[str],
    label_column: Optional[str] = None,
) -> MondrianResult:
    """
    Output:
    - QI values are generalized:
      - numeric -> interval string in the original column, plus `<col>_mid` and `<col>_width` numeric features
      - categorical -> set string like `{a|b|c}` in the original column
    - label column (if provided) is copied without modification.
    """

    if k <= 1:
        raise ValueError("k must be >= 2")

    qi_columns = list(qi_columns)
    categorical_qi = list(categorical_qi)
    numeric_qi = list(numeric_qi)

    for col in qi_columns:
        if col not in df.columns:
            raise KeyError(f"QI column not found: {col}")
    if label_column is not None and label_column not in df.columns:
        raise KeyError(f"label column not found: {label_column}")

    if set(categorical_qi) | set(numeric_qi) != set(qi_columns):
        raise ValueError("categorical_qi + numeric_qi must exactly match qi_columns")

    clean_df = df.copy()
    for col in qi_columns:
        if clean_df[col].dtype == object:
            clean_df[col] = clean_df[col].astype(str).str.strip()

    mask_missing = pd.Series(False, index=clean_df.index)
    for col in categorical_qi:
        mask_missing |= clean_df[col].map(_is_missing)
    for col in numeric_qi:
        s_num = pd.to_numeric(clean_df[col], errors="coerce")
        mask_missing |= s_num.isna()
        clean_df[col] = s_num
    if label_column is not None:
        mask_missing |= clean_df[label_column].map(_is_missing)
    clean_df = clean_df.loc[~mask_missing].copy().reset_index(drop=True)

    numeric_arr = {c: clean_df[c].to_numpy(dtype=float) for c in numeric_qi}
    cat_arr = {c: clean_df[c].astype(str).to_numpy() for c in categorical_qi}
    global_numeric = {c: (float(numeric_arr[c].min()), float(numeric_arr[c].max())) for c in numeric_qi}
    global_cat = {c: int(np.unique(cat_arr[c]).size) for c in categorical_qi}

    def _split_numeric_fast(idx: List[int], col: str) -> Optional[Tuple[List[int], List[int]]]:
        if len(idx) < 2:
            return None
        a = numeric_arr[col][idx]
        order = np.argsort(a, kind="mergesort")
        ordered_idx = [idx[int(i)] for i in order]
        mid = len(ordered_idx) // 2
        left = ordered_idx[:mid]
        right = ordered_idx[mid:]
        if not left or not right:
            return None
        return left, right

    def _split_categorical_fast(idx: List[int], col: str) -> Optional[Tuple[List[int], List[int]]]:
        if len(idx) < 2:
            return None

        vals = cat_arr[col][idx]
        cats, counts = np.unique(vals, return_counts=True)
        total = int(counts.sum())
        if total < 2:
            return None

        # Deterministic ordering: by freq desc, then value asc
        pairs = list(zip(cats.tolist(), counts.tolist()))
        pairs.sort(key=lambda t: (-int(t[1]), str(t[0])))

        left_set: Set[str] = set()
        left_count = 0
        half = total / 2
        for c, cnt in pairs:
            if left_count >= half:
                break
            left_set.add(str(c))
            left_count += int(cnt)

        if not left_set or len(left_set) == len(pairs):
            return None

        mask_left = np.isin(vals.astype(str), list(left_set))
        left = [idx[i] for i, m in enumerate(mask_left.tolist()) if m]
        right = [idx[i] for i, m in enumerate(mask_left.tolist()) if not m]
        if not left or not right:
            return None
        return left, right

    idx0 = list(range(len(clean_df)))
    stack: List[List[int]] = [idx0]
    finals: List[List[int]] = []

    while stack:
        idx = stack.pop()
        if len(idx) < 2 * k:
            finals.append(idx)
            continue

        spans: List[Tuple[float, str, bool]] = []
        for col in numeric_qi:
            a = numeric_arr[col][idx]
            lo = float(a.min())
            hi = float(a.max())
            g_lo, g_hi = global_numeric[col]
            denom = g_hi - g_lo
            spans.append(((hi - lo) / denom if denom > 0 else 0.0, col, True))

        for col in categorical_qi:
            vals = cat_arr[col][idx]
            g = global_cat[col]
            spans.append(((float(np.unique(vals).size) / g) if g > 0 else 0.0, col, False))
        spans.sort(reverse=True, key=lambda t: (t[0], t[1]))

        did_split = False
        for span, col, is_numeric in spans:
            if span <= 0:
                continue

            split = _split_numeric_fast(idx, col) if is_numeric else _split_categorical_fast(idx, col)
            if split is None:
                continue
            left, right = split
            if len(left) < k or len(right) < k:
                continue

            stack.append(left)
            stack.append(right)
            did_split = True
            break

        if not did_split:
            finals.append(idx)

    # Build eq class id mapping
    eq_id_arr = np.empty(len(clean_df), dtype=np.int64)
    for i, idx in enumerate(finals):
        eq_id_arr[np.asarray(idx, dtype=int)] = int(i)
    eq_id = pd.Series(eq_id_arr, index=clean_df.index, dtype="int64")

    # Materialize anonymized dataframe
    cols_out: List[str] = list(qi_columns)
    if label_column is not None:
        cols_out.append(label_column)

    out = clean_df[cols_out].copy()

    for col in numeric_qi:
        out[col] = out[col].astype(object)

    for col in categorical_qi:
        out[col] = out[col].astype(object)

    # Pre-create numeric feature columns
    for col in numeric_qi:
        out[f"{col}_mid"] = np.nan
        out[f"{col}_width"] = np.nan

    for idx in finals:
        for col in numeric_qi:
            a = numeric_arr[col][idx]
            lo = float(a.min())
            hi = float(a.max())
            out.iloc[idx, out.columns.get_loc(col)] = _format_interval(lo, hi)
            out.iloc[idx, out.columns.get_loc(f"{col}_mid")] = (lo + hi) / 2.0
            out.iloc[idx, out.columns.get_loc(f"{col}_width")] = (hi - lo)

        for col in categorical_qi:
            vals = np.unique(cat_arr[col][idx]).tolist()
            out.iloc[idx, out.columns.get_loc(col)] = _format_set(vals)

    out["eq_class_id"] = eq_id

    # Faster class sizes than value_counts+map for large data.
    counts = np.bincount(eq_id_arr)
    out["eq_class_size"] = counts[eq_id_arr].astype(int)

    return MondrianResult(df_anonymized=out, partitions=finals, eq_class_id=eq_id)


def encode_raw_as_generalized(
    df: pd.DataFrame,
    *,
    qi_columns: Sequence[str],
    categorical_qi: Sequence[str],
    numeric_qi: Sequence[str],
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """Encode the *raw* dataset into the same schema as `mondrian_anonymize` output.

    - numeric QI: interval `[v,v]` plus `<col>_mid=v`, `<col>_width=0`
    - categorical QI: singleton set `{v}`
    """

    qi_columns = list(qi_columns)
    categorical_qi = list(categorical_qi)
    numeric_qi = list(numeric_qi)

    cols_out: List[str] = list(qi_columns)
    if label_column is not None:
        cols_out.append(label_column)

    out = df[cols_out].copy()

    for col in numeric_qi:
        s = pd.to_numeric(out[col], errors="coerce")
        out[col] = s.map(lambda v: _format_interval(float(v), float(v)))
        out[f"{col}_mid"] = s.astype(float)
        out[f"{col}_width"] = 0.0

    for col in categorical_qi:
        s = out[col].astype(str).str.strip()
        out[col] = s.map(lambda v: _format_set([v]))

    return out


def assert_k_anonymous(df_anonymized: pd.DataFrame, *, k: int, eq_col: str = "eq_class_id") -> None:
    """Raise if any equivalence class size is < k."""

    vc = df_anonymized[eq_col].value_counts(dropna=False)
    bad = vc[vc < k]
    if not bad.empty:
        raise AssertionError(f"Found {len(bad)} equivalence classes with size < k")
