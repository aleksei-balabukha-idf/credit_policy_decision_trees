"""Main window: file actions, tree setup, node split orchestration,
progress + cancel for split_best, session save/load, PNG export."""
from __future__ import annotations

import logging
import copy

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDockWidget,
                             QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                             QMainWindow, QMessageBox, QProgressDialog,
                             QSpinBox, QTableWidget, QTableWidgetItem,
                             QToolBar, QVBoxLayout, QWidget)
from PyQt6.QtGui import QAction

from core.csv_loader import CSVLoadError, CSVOptions, load_csv
from core.dtypes import TYPE_ICONS, TYPE_LABEL_COLOR, ColumnType
from core.metrics import Metric
from core.session import SessionError, load_session, save_session
from core.splitter import SplitCandidate, best_split
from core.tree import DecisionTreeModel, SplitNotAllowed
from .csv_preview import SniffPreviewDialog
from .log_panel import LogPanel
from .split_dialog import SplitDialog
from .tree_canvas import TreeCanvas
from .workers import SplitBestWorker

log = logging.getLogger("dtree.ui")


class TreeSetupDialog(QDialog):
    """Target / features (with type icons and type switch) / metric /
    min leaf / max depth."""

    def __init__(self, df, types, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tree settings")
        self.resize(700, 520)
        self.df, self.types = df, dict(types)
        cols = list(df.columns)

        root = QVBoxLayout(self)
        form = QFormLayout()
        self.target = QComboBox(); self.target.addItems(cols)
        self.metric = QComboBox()
        self.metric.addItem("Gini (classification)", Metric.GINI)
        self.metric.addItem("Entropy (classification)", Metric.ENTROPY)
        self.metric.addItem("Variance / MSE (regression)", Metric.MSE)
        self.metric.setCurrentIndex(0)  # Gini — по умолчанию (classification-first)
        self.min_leaf = QSpinBox(); self.min_leaf.setRange(1, 10**9)
        self.min_leaf.setValue(max(1, len(df) // 100))
        self.max_depth = QSpinBox(); self.max_depth.setRange(1, 100)
        self.max_depth.setValue(6)
        form.addRow("Target variable:", self.target)
        form.addRow("Split metric:", self.metric)
        form.addRow("Min leaf size:", self.min_leaf)
        form.addRow("Max depth:", self.max_depth)
        root.addLayout(form)

        root.addWidget(QLabel("Features used for splits "
                              "( (Numeric) numeric, (Categorical) categorical, (Date) date-as-int):"))
        self.table = QTableWidget(len(cols), 3)
        self.table.setHorizontalHeaderLabels(["Use", "Feature", "Type"])
        self._type_combos = {}
        for i, c in enumerate(cols):
            chk = QTableWidgetItem()
            chk.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(i, 0, chk)
            name_lbl = QLabel(
                f"<span style='color:{TYPE_LABEL_COLOR}'>"
                f"{TYPE_ICONS[self.types[c]]}</span> {c}")
            name_lbl.setTextFormat(Qt.TextFormat.RichText)
            name_lbl.setContentsMargins(4, 0, 4, 0)
            self.table.setCellWidget(i, 1, name_lbl)
            combo = QComboBox(); combo.addItems([t.value for t in ColumnType])
            combo.setCurrentText(self.types[c].value)
            combo.currentTextChanged.connect(
                lambda txt, row=i, col=c: self._on_type(row, col, txt))
            self.table.setCellWidget(i, 2, combo)
            self._type_combos[c] = combo
        root.addWidget(self.table)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    # --------------------------------------------------------------- helpers
    _REGRESSION_METRICS = (Metric.MSE, Metric.MEAN)

    def _target_type(self) -> ColumnType:
        return self.types.get(self.target.currentText(), ColumnType.NUMERIC)

   # --------------------------------------------------------------- валидация

    def accept(self):
        """Проверка совместимости типа целевой переменной и метрики.

        Categorical target + Regression metric (MSE/MEAN) — несовместимо:
        дерево не сможет вычислить среднее по строковым классам.
        Показываем предупреждение и НЕ закрываем диалог.
        """
        target = self.target.currentText()
        metric = self.metric.currentData()
        target_type = self.types.get(target, ColumnType.NUMERIC)

        if target_type == ColumnType.CATEGORICAL and metric in self._REGRESSION_METRICS:
            QMessageBox.warning(
                self,
                "Metric / target type mismatch",
                f"Target column “{target}” has type “{target_type.value}”, "
                f"but the selected split metric is a regression metric "
                f"(“{metric.value}”).\n\n"
                "Regression metrics (Variance/MSE, Mean) require a "
                "numeric or date target column.\n\n"
                "Please either:\n"
                "   • change the target column type to Numeric or Date, or\n"
                "   • select a classification metric (Gini or Entropy)."
            )
            return  # диалог остаётся открытым

        super().accept()

    def _on_type(self, row, col, txt):
        self.types[col] = ColumnType(txt)
        lbl = self.table.cellWidget(row, 1)
        if lbl:
            lbl.setText(
                f"<span style='color:{TYPE_LABEL_COLOR}'>"
                f"{TYPE_ICONS[self.types[col]]}</span> {col}")

    def values(self):
        checked = []
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).checkState() == Qt.CheckState.Checked:
                checked.append(self.df.columns[i])
        target = self.target.currentText()
        features = [c for c in checked if c != target]
        return {"target": target, "features": features, "types": self.types,
                "metric": self.metric.currentData(),
                "min_leaf": self.min_leaf.value(),
                "max_depth": self.max_depth.value()}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Decision Tree")
        self.resize(1400, 900)
        self.canvas = TreeCanvas()
        self.setCentralWidget(self.canvas)
        self.model: DecisionTreeModel | None = None
        self.csv_options = CSVOptions()
        self.source_path = ""
        self._worker: SplitBestWorker | None = None

        self.log_panel = LogPanel()
        dock = QDockWidget("Log")
        dock.setWidget(self.log_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        for text, slot in (("Open CSV…", self.open_csv),
                           ("Open session…", self.open_session),
                           ("Save session…", self.save_session),
                           ("Export PNG…", self.export_png),
                           ("Zoom +", lambda: self.canvas.scale(1.25, 1.25)),
                           ("Zoom −", lambda: self.canvas.scale(0.8, 0.8)),
                           ("Fit", self.canvas.zoom_fit)):
            act = QAction(text, self)
            act.triggered.connect(slot)
            tb.addAction(act)

    # ----------------------------------------------------------- loading

    def open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", "", "CSV files (*.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            dlg = SniffPreviewDialog(path, self)
        except CSVLoadError as exc:
            QMessageBox.critical(self, "Cannot open file", str(exc))
            return
        if not dlg.exec():
            return
        self.csv_options = dlg.current_options()
        try:
            df, types = load_csv(path, self.csv_options, dlg.type_overrides())
        except CSVLoadError as exc:
            QMessageBox.critical(self, "Cannot load file", str(exc))
            return
        if len(df) == 0:
            QMessageBox.critical(self, "Empty data",
                                 "All rows were dropped during parsing.")
            return
        self.source_path = path
        setup = TreeSetupDialog(df, types, self)
        if not setup.exec():
            return
        v = setup.values()
        try:
            self.model = DecisionTreeModel(df, v["types"], v["target"],
                                           v["features"], v["metric"],
                                           v["min_leaf"], v["max_depth"])
        except SplitNotAllowed as exc:
            QMessageBox.critical(self, "Cannot build tree", str(exc))
            return
        for w in self.model.warnings:
            QMessageBox.warning(self, "Warning", w)
        self.canvas.rebuild(self.model)
        self.canvas.zoom_fit()

    def open_session(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open session", "", "Tree sessions (*.tree)")
        if not path:
            return
        try:
            self.model, self.csv_options, self.source_path, lines = \
                load_session(path)
        except SessionError as exc:
            QMessageBox.critical(self, "Cannot open session", str(exc))
            return
        for line in lines:
            self.log_panel._append(line)
        self.canvas.rebuild(self.model)
        self.canvas.zoom_fit()

    def save_session(self):
        if not self.model:
            QMessageBox.information(self, "Nothing to save", "No tree yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", "session.tree", "Tree sessions (*.tree)")
        if not path:
            return
        try:
            save_session(self.model, path, self.csv_options,
                         self.source_path, self.log_panel.lines)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def export_png(self):
        if not self.model:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export tree as PNG", "tree.png", "PNG (*.png)")
        if path:
            self.canvas.export_png(path)
            log.info("tree exported to %s", path)

    # ---------------------------------------------------------- splitting

    def on_split_best(self, node_id: int):
        node = self.model.get_node(node_id)
        df = self.model.get_data(node_id)
        progress = QProgressDialog("Searching best split…", "Cancel",
                                   0, len(self.model.features), self)
        progress.setWindowTitle("split_best")
        progress.setMinimumDuration(300)
        self._worker = SplitBestWorker(
            df, self.model.features, self.model.types, self.model.target,
            self.model.metric, self.model.min_leaf, node.blocked_features,
            nan_as_category=True, parent=self)
        self._worker.progress.connect(
            lambda i, n: (progress.setMaximum(n), progress.setValue(i)))
        progress.canceled.connect(self._worker.cancel)

        def done(cand):
            progress.close()
            if cand is None:
                QMessageBox.information(
                    self, "split_best",
                    "No valid split: constraints (min leaf / depth) or "
                    "all features are constant.")
                return
            self._apply(node_id, cand)

        def failed(msg):
            progress.close()
            if msg != "__cancelled__":
                QMessageBox.critical(self, "split_best failed", msg)
            else:
                log.info("split_best cancelled by user")

        self._worker.finished_ok.connect(done)
        self._worker.failed.connect(failed)
        self._worker.start()

    def on_split_manual(self, node_id: int):
        node = self.model.get_node(node_id)
        df = self.model.get_data(node_id)
        dlg = SplitDialog(df, self.model.features, self.model.types,
                          self.model.metric, node.blocked_features, self)
        if not dlg.exec():
            return
        cand = dlg.build_candidate()
        if cand is None:
            QMessageBox.information(self, "Manual split",
                                    "Select at least one category value.")
            return
        self._apply(node_id, cand)

    def on_undo_split(self, node_id: int):
        """Undo: удалить split узла и всё поддерево под ним."""
        node = self.model.get_node(node_id)
        if node.split is None:
            return
        # сколько потомков будет удалено
        n_desc = 0
        stack = list(node.children.values())
        while stack:
            n = stack.pop()
            n_desc += 1
            stack.extend(n.children.values())
        # у разбитого узла всегда 2 потомка (left+right) или 3 (если NaN separate);
        # если больше — под ним есть ещё сплиты, предупреждаем
        if n_desc > 3:
            ans = QMessageBox.question(
                self, "Undo split",
                f"Undo the split on this node?\n\n"
                f"The subtree below it will be removed "
                f"({n_desc} node(s), including their splits), "
                f"as if this split was never made.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return
        removed = self.model.undo_split(node_id)
        log.info("undo split on node %d — %d descendant node(s) removed",
                 node_id, len(removed))
        self.canvas.rebuild(self.model)

    def _apply(self, node_id: int, cand: SplitCandidate):
        try:
            self.model.split_node(node_id, cand)
        except SplitNotAllowed as exc:
            QMessageBox.information(self, "Split not allowed", str(exc))
            return
        if self.model.last_split_warnings:
            QMessageBox.warning(
                self, "NaN leaf smaller than min leaf",
                "\n\n".join(self.model.last_split_warnings))
        self.canvas.rebuild(self.model)

    def on_nan_route_changed(self, node_id: int, new_route: str):
        """Пересобрать split с новым маршрутом NaN (поддерево удаляется)."""
        node = self.model.get_node(node_id)
        if node.split is None or node.split.nan_route == new_route:
            return

        # есть ли под узлом вложенные split'?
        has_deep = any(c.children for c in node.children.values())
        if has_deep:
            n_desc = sum(1 for c in node.children.values()
                         for _ in self._iter_subtree(c))
            ans = QMessageBox.question(
                self, "Change NaN handling",
                f"Changing NaN handling re-applies the split on this node.\n"
                f"The subtree below ({n_desc} node(s)) will be removed.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return

        new_cand = copy.deepcopy(node.split)
        new_cand.nan_route = new_route

        try:
            self.model.check_can_split(node, new_cand)
        except SplitNotAllowed as exc:
            QMessageBox.information(self, "Cannot change NaN handling",
                                    str(exc))
            return

        self.model.undo_split(node_id)
        self.model.split_node(node_id, new_cand)   # уже проверено, не упадёт
        if self.model.last_split_warnings:
            QMessageBox.warning(
                self, "NaN leaf smaller than min leaf",
                "\n\n".join(self.model.last_split_warnings))
        self.canvas.rebuild(self.model)

    def _iter_subtree(self, node):
        yield node
        for c in node.children.values():
            yield from self._iter_subtree(c)

    


def wire_signals(window: MainWindow):
    # Кнопки в NodeWidget запускают цепочку синхронных прямых вызовов.
    # on_* -> canvas.rebuild -> scene.clear() уничтожает виджет-источник,
    # из-за чего Qt падает. QueuedConnection переносит вызов слота в
    # следующий такт event loop, к моменту которого обработчик clicked
    # уже полностью завершён.
    Q = Qt.ConnectionType.QueuedConnection
    window.canvas.split_best_clicked.connect(window.on_split_best, Q)
    window.canvas.split_clicked.connect(window.on_split_manual, Q)
    window.canvas.undo_split_clicked.connect(window.on_undo_split, Q)
    window.canvas.nan_route_changed.connect(
        window.on_nan_route_changed, Q)