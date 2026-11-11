"""Session persistence: .tree files.

A .tree file is a ZIP container:
  - session.json  — tree structure, params, types, csv options, log lines
  - data.csv.gz   — snapshot of the data (session survives a moved source)
"""
from __future__ import annotations

import gzip
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .csv_loader import CSVOptions
from .tree import DecisionTreeModel

log = logging.getLogger("dtree.session")

FORMAT = "interactive-tree-session"
VERSION = 1


class SessionError(Exception):
    pass


def save_session(model: DecisionTreeModel, path: str | Path,
                 csv_options: CSVOptions, source_path: str,
                 log_lines: list[str] | None = None) -> None:
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": source_path,
        "csv_options": csv_options.to_dict(),
        "model": model.to_dict(),
        "log": log_lines or [],
    }
    buf = model.df.to_csv(index=False).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("session.json", json.dumps(payload, ensure_ascii=False, indent=2))
        zf.writestr("data.csv.gz", gzip.compress(buf))
    log.info("session saved: %s", path)


def load_session(path: str | Path):
    """Returns (DecisionTreeModel, CSVOptions, source_path, log_lines)."""
    path = Path(path)
    if not path.exists():
        raise SessionError(f"Session file not found: {path}")
    try:
        with zipfile.ZipFile(path) as zf:
            meta = json.loads(zf.read("session.json").decode("utf-8"))
            raw = gzip.decompress(zf.read("data.csv.gz"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise SessionError(f"Corrupt session file: {exc}") from exc

    if meta.get("format") != FORMAT:
        raise SessionError("Not a decision-tree session file.")
    if meta.get("version", 0) > VERSION:
        raise SessionError(f"Session version {meta['version']} is newer than "
                           f"supported ({VERSION}).")

    import io
    df = pd.read_csv(io.BytesIO(raw))
    model = DecisionTreeModel.from_dict(df, meta["model"])
    log.info("session loaded: %s (%d nodes)", path, len(model.nodes))
    return (model, CSVOptions.from_dict(meta.get("csv_options", {})),
            meta.get("source_path", ""), meta.get("log", []))
