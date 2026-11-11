"""Application entry point.

Run:  python src/main.py
Build .exe:  pyinstaller app.spec
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PyQt6.QtWidgets import QApplication


def configure_logging() -> None:
    """File gets DEBUG+ (rotating, 5 MB); console/panel gets INFO+."""
    root = logging.getLogger("dtree")
    root.setLevel(logging.DEBUG)
    log_dir = Path.home() / ".dtree_logs"
    log_dir.mkdir(exist_ok=True)
    fh = RotatingFileHandler(log_dir / "dtree.log", maxBytes=5 * 1024 * 1024,
                             backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(ch)


def main() -> int:
    configure_logging()
    logging.getLogger("dtree").info("application started")
    app = QApplication(sys.argv)
    app.setApplicationName("Interactive Decision Tree")

    # Qt must exist before importing widgets-heavy modules lazily
    from ui.main_window import MainWindow, wire_signals
    window = MainWindow()
    wire_signals(window)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
