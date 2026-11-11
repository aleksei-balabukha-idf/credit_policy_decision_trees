"""Background workers: split_best must not freeze the GUI on 100k+ rows."""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from core.splitter import CancelledError, best_split


class SplitBestWorker(QThread):
    progress = pyqtSignal(int, int)          # current feature, total
    finished_ok = pyqtSignal(object)         # SplitCandidate | None
    failed = pyqtSignal(str)

    def __init__(self, df, features, types, target, metric, min_leaf,
                 blocked, nan_as_category, parent=None):
        super().__init__(parent)
        self._args = (df, features, types, target, metric, min_leaf)
        self._blocked = set(blocked)
        self._nan_cat = nan_as_category
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            cand = best_split(
                *self._args,
                blocked=self._blocked,
                nan_as_category=self._nan_cat,
                progress_cb=lambda i, n: self.progress.emit(i, n),
                cancel_cb=lambda: self._cancelled,
            )
            self.finished_ok.emit(cand)
        except CancelledError:
            self.failed.emit("__cancelled__")
        except Exception as exc:  # surfaced to the user by the caller
            self.failed.emit(str(exc))
