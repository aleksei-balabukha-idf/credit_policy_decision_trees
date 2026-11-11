"""CSV loading: encoding/delimiter sniffing, fast preview, full load.

Defaults (per spec): encoding='utf-8', delimiter=',', first row is the header,
decimal separator '.' (e.g. 599.4).
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from .dtypes import ColumnType, convert_column, infer_type

log = logging.getLogger("dtree.csv")


class CSVLoadError(Exception):
    """User-facing load failure (missing/empty/corrupt file, bad encoding...)."""


@dataclass
class CSVOptions:
    encoding: str = "utf-8"
    delimiter: str = ","
    decimal: str = "."
    thousands: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CSVOptions":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


_ENCODINGS = ("utf-8", "utf-8-sig", "cp1251", "latin-1")
_DELIMITERS = (",", ";", "\t", "|")


def _detect_encoding(path: Path, sample_bytes: int = 65536) -> str:
    raw = path.read_bytes()[:sample_bytes]
    if not raw:
        raise CSVLoadError("File is empty.")
    for enc in _ENCODINGS:
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    raise CSVLoadError("Unsupported encoding: none of "
                       + ", ".join(_ENCODINGS) + " could decode the file.")


def sniff_options(path: str | Path) -> CSVOptions:
    """Guess encoding/delimiter/decimal. Fast: reads only a small sample."""
    path = Path(path)
    if not path.exists():
        raise CSVLoadError(f"File not found: {path}")
    enc = _detect_encoding(path)
    sample = path.read_text(encoding=enc, errors="replace")[:65536]
    delimiter = ","
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters="".join(_DELIMITERS)).delimiter
    except csv.Error:
        # Fallback: pick the most frequent candidate in the header line
        header = sample.splitlines()[0] if sample else ""
        counts = {d: header.count(d) for d in _DELIMITERS}
        if max(counts.values(), default=0) > 0:
            delimiter = max(counts, key=counts.get)
    decimal = "."
    if delimiter == "," and ";" not in sample:
        decimal = "."
    elif delimiter == ";":
        # European style: comma is likely the decimal separator
        decimal = "," if sample.count(",") > sample.count(".") else "."
    opts = CSVOptions(encoding=enc, delimiter=delimiter, decimal=decimal)
    log.debug("sniffed options: %s", opts)
    return opts


def _read(path: Path, opts: CSVOptions, nrows: int | None) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path, encoding=opts.encoding, delimiter=opts.delimiter,
            decimal=opts.decimal, thousands=opts.thousands,
            header=0, nrows=nrows, dtype=object,  # keep raw; types resolved later
            on_bad_lines="error",
        )
    except FileNotFoundError as exc:
        raise CSVLoadError(f"File not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise CSVLoadError(f"Encoding '{opts.encoding}' cannot decode this file.") from exc
    except pd.errors.EmptyDataError as exc:
        raise CSVLoadError("File has no data (header only or empty).") from exc
    except pd.errors.ParserError as exc:
        raise CSVLoadError(f"Corrupt CSV: {exc}") from exc


def load_preview(path: str | Path, opts: CSVOptions, n_rows: int = 20):
    """Sniff & Preview: first n_rows + inferred type per column."""
    path = Path(path)
    df = _read(path, opts, n_rows)
    types = {col: infer_type(df[col]) for col in df.columns}
    log.info("preview loaded: %d rows x %d cols", len(df), len(df.columns))
    return df, types


def load_csv(path: str | Path, opts: CSVOptions,
             type_overrides: dict[str, ColumnType] | None = None):
    """Full load. Returns (DataFrame, {column: ColumnType}).

    Columns are converted to their logical type; DATE columns become
    Int64 YYYYMMDD (numeric for the tree engine).
    """
    path = Path(path)
    raw = _read(path, opts, None)
    type_overrides = type_overrides or {}
    out = {}
    types: dict[str, ColumnType] = {}
    for col in raw.columns:
        ctype = type_overrides.get(col) or infer_type(raw[col])
        types[col] = ctype
        if ctype == ColumnType.CATEGORICAL:
            out[col] = raw[col].astype("string")
        else:
            out[col] = convert_column(raw[col], ctype, opts.decimal, opts.thousands)
    df = pd.DataFrame(out)
    log.info("csv loaded: %d rows x %d cols from %s", len(df), len(df.columns), path)
    return df, types
