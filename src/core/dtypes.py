"""Column type inference and conversion.

Three logical types: NUMERIC, CATEGORICAL, DATE.
Dates are converted to int64 as YYYYMMDD (e.g. 20251229) and then
treated as numeric by the tree engine.
"""
from __future__ import annotations

import enum

import numpy as np
import pandas as pd


class ColumnType(str, enum.Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"


# Header icons used across the UI.
TYPE_ICONS = {
    ColumnType.NUMERIC: "(Numeric)",
    ColumnType.CATEGORICAL: "(Categorical)",
    ColumnType.DATE: "(Date)", 
}
# Pale brown color for type labels (headers, cell labels, combo boxes).
TYPE_LABEL_COLOR = "#a87a5a"

_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y",
                 "%Y%m%d", "%d-%m-%Y")


def _looks_like_date(s: pd.Series) -> bool:
    """Heuristic: >=90% of non-null values parse as a date."""
    sample = s.dropna().astype(str).str.strip()
    if sample.empty:
        return False
    # Reject plain numbers (123, -4.5, 599,4) — they are numeric, not dates.
    if sample.str.fullmatch(r"[+-]?\d+([.,]\d+)?").mean() > 0.9:
        # ... except 8-digit YYYYMMDD stamps
        return bool(sample.str.fullmatch(r"(19|20)\d{6}").mean() > 0.9)
    for fmt in _DATE_FORMATS:
        try:
            parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        except (ValueError, TypeError):
            continue
        if parsed.notna().mean() >= 0.9:
            return True
    try:
        parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
        return bool(parsed.notna().mean() >= 0.9)
    except (ValueError, TypeError):
        return False


def infer_type(s: pd.Series) -> ColumnType:
    """Auto-detect logical type of a column (int, float, bool, object/category, date)."""
    if pd.api.types.is_bool_dtype(s):
        return ColumnType.CATEGORICAL
    if pd.api.types.is_numeric_dtype(s):
        return ColumnType.NUMERIC
    if pd.api.types.is_datetime64_any_dtype(s):
        return ColumnType.DATE
    if _looks_like_date(s):
        return ColumnType.DATE
    # Numeric values read as strings (preview keeps raw text)
    try:
        parsed = pd.to_numeric(s.dropna(), errors="coerce")
        if len(s.dropna()) and parsed.notna().mean() >= 0.9:
            return ColumnType.NUMERIC
    except (ValueError, TypeError):
        pass
    return ColumnType.CATEGORICAL


def date_to_int(s: pd.Series) -> pd.Series:
    """Convert dates to int64 YYYYMMDD. Unparseable values -> NA."""
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = s
    else:
        raw = s.astype(str).str.strip()
        dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        for fmt in _DATE_FORMATS:
            mask = dt.isna() & raw.notna()
            if not mask.any():
                break
            try:
                dt.loc[mask] = pd.to_datetime(raw[mask], format=fmt, errors="coerce")
            except (ValueError, TypeError):
                pass
    out = (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day)
    return out.astype("Int64")


def to_numeric(s: pd.Series, decimal: str = ".", thousands: str | None = None) -> pd.Series:
    """Convert a column to float, honouring the CSV decimal/thousands settings."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    txt = s.astype(str).str.strip()
    txt = txt.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    if thousands:
        txt = txt.str.replace(thousands, "", regex=False)
    if decimal and decimal != ".":
        txt = txt.str.replace(decimal, ".", regex=False)
    txt = txt.str.replace(" ", "", regex=False)  # NBSP-ish thousands
    return pd.to_numeric(txt, errors="coerce")


def convert_column(s: pd.Series, target: ColumnType,
                   decimal: str = ".", thousands: str | None = None) -> pd.Series:
    """Convert a column to the requested logical type.

    DATE returns Int64 YYYYMMDD so the tree treats it as numeric.
    """
    if target == ColumnType.NUMERIC:
        if pd.api.types.is_datetime64_any_dtype(s):
            return date_to_int(s).astype("Float64")
        return to_numeric(s, decimal, thousands).astype("Float64")
    if target == ColumnType.DATE:
        return date_to_int(s)
    # CATEGORICAL
    return s.astype("string")


def mixed_type_cells(s: pd.Series) -> pd.Index:
    """Indices of cells whose Python type differs from the column's majority type.

    Used to highlight heterogeneous cells in the preview.
    """
    non_null = s.dropna()
    if non_null.empty or not pd.api.types.is_object_dtype(s):
        return pd.Index([])
    counts = non_null.map(type).value_counts()
    if len(counts) <= 1:
        return pd.Index([])
    majority = counts.index[0]
    return non_null.index[non_null.map(type) != majority]
