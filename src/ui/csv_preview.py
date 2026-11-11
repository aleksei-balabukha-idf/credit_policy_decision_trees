"""Sniff & Preview dialog: CSV options + first-20-rows preview with
problem-cell highlighting + per-column type override."""
from __future__ import annotations

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                             QHBoxLayout, QHeaderView, QLabel, QMessageBox,
                             QTableView, QVBoxLayout)

from core.csv_loader import CSVLoadError, CSVOptions, load_preview, sniff_options
from core.dtypes import (TYPE_ICONS, TYPE_LABEL_COLOR, ColumnType,
                         mixed_type_cells)

import pandas as pd


class PreviewModel(QAbstractTableModel):
    """QTableView model with NaN / mixed-type cell highlighting."""

    def __init__(self, df: pd.DataFrame, types: dict[str, ColumnType]):
        super().__init__()
        self._df = df
        self._types = types
        self._mixed = {c: set(mixed_type_cells(df[c])) for c in df.columns}

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, idx, role=Qt.ItemDataRole.DisplayRole):
        if not idx.isValid():
            return None
        val = self._df.iat[idx.row(), idx.column()]
        col = self._df.columns[idx.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return "" if pd.isna(val) else str(val)
        if role == Qt.ItemDataRole.BackgroundRole:
            if pd.isna(val):
                return QColor("#f63e3e")          # NaN — soft red
            if idx.row() in self._mixed.get(col, ()):
                return QColor("#fcc511")          # mixed type — amber
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal:
            col = self._df.columns[section]
            icon = TYPE_ICONS.get(self._types.get(col), "?")
            if role == Qt.ItemDataRole.DisplayRole:
                return f"{icon} {col}"
            if role == Qt.ItemDataRole.ForegroundRole:
                return QColor(TYPE_LABEL_COLOR)
        return super().headerData(section, orientation, role)


class SniffPreviewDialog(QDialog):
    """Shows first 20 rows with guessed types; user tunes read options
    and column types before the full load."""

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.setWindowTitle("Sniff & Preview")
        self.resize(1000, 600)
        try:
            self.options = sniff_options(path)
        except CSVLoadError as exc:
            QMessageBox.critical(self, "Cannot open file", str(exc))
            self.options = CSVOptions()

        root = QVBoxLayout(self)

        # --- read options row
        opts = QHBoxLayout()
        self.enc = QComboBox(); self.enc.setEditable(True)
        self.enc.addItems(["utf-8", "utf-8-sig", "cp1251", "latin-1"])
        self.enc.setCurrentText(self.options.encoding)
        self.delim = QComboBox(); self.delim.setEditable(True)
        self.delim.addItems([",", ";", "\\t", "|"])
        self.delim.setCurrentText({"	": "\\t"}.get(self.options.delimiter,
                                                    self.options.delimiter))
        self.dec = QComboBox(); self.dec.setEditable(True)
        self.dec.addItems([".", ","])
        self.dec.setCurrentText(self.options.decimal)
        for lbl, w in (("Encoding", self.enc), ("Delimiter", self.delim),
                       ("Decimal", self.dec)):
            opts.addWidget(QLabel(lbl + ":")); opts.addWidget(w)
        from PyQt6.QtWidgets import QPushButton
        self.refresh = QPushButton("Refresh preview")
        self.refresh.clicked.connect(self._reload)
        opts.addWidget(self.refresh); opts.addStretch()
        root.addLayout(opts)

        # --- preview table
        self.table = QTableView()
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        root.addWidget(self.table, stretch=1)
        self.legend = QLabel("Highlight: red = NaN, amber = mixed types. "
                             "Showing first 20 rows.")
        root.addWidget(self.legend)

        # --- per-column type overrides
        root.addWidget(QLabel("Column types (changeable):"))
        self.type_combos: dict[str, QComboBox] = {}
        self.types_box = QVBoxLayout()
        root.addLayout(self.types_box)

        # --- buttons
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        self._reload()

    def current_options(self) -> CSVOptions:
        d = self.delim.currentText()
        return CSVOptions(encoding=self.enc.currentText(),
                          delimiter="\t" if d == "\\t" else d,
                          decimal=self.dec.currentText())

    def _reload(self):
        for w in list(self.type_combos.values()):
            w.setParent(None)
        self.type_combos.clear()
        while self.types_box.count():
            item = self.types_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
        try:
            df, types = load_preview(self.path, self.current_options(), n_rows=20)
        except CSVLoadError as exc:
            QMessageBox.critical(self, "Preview failed", str(exc))
            return
        self.table.setModel(PreviewModel(df, types))
        row_layout = None
        for i, col in enumerate(df.columns):
            if i % 4 == 0:
                row_layout = QHBoxLayout()
                self.types_box.addLayout(row_layout)
            combo = QComboBox()
            combo.addItems([t.value for t in ColumnType])
            combo.setCurrentText(types[col].value)
            self.type_combos[col] = combo
            lbl = QLabel(
                f"<span style='color:{TYPE_LABEL_COLOR}'>"
                f"{TYPE_ICONS[types[col]]}</span> {col}:")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            row_layout.addWidget(lbl)
            row_layout.addWidget(combo)

    def type_overrides(self) -> dict[str, ColumnType]:
        return {c: ColumnType(cb.currentText())
                for c, cb in self.type_combos.items()}
