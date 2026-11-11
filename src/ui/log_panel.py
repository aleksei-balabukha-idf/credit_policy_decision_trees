"""In-app log console. File logging is configured in main.py; this panel
shows INFO+ user-facing events."""
from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QPlainTextEdit


class QtLogHandler(logging.Handler, QObject):
    record_ready = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record: logging.LogRecord):
        self.record_ready.emit(self.format(record))


class LogPanel(QPlainTextEdit):
    """Docked console. Also collects lines for saving into the session."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.lines: list[str] = []
        handler = QtLogHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        handler.record_ready.connect(self._append)
        logging.getLogger("dtree").addHandler(handler)

    def _append(self, text: str):
        self.lines.append(text)
        self.appendPlainText(text)
