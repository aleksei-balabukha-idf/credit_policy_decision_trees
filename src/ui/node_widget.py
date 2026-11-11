"""Node card widget shown on the tree canvas."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMenu,
                             QPushButton, QToolButton, QVBoxLayout)

from core.tree import Node
from .class_visuals import StackedBar


class NodeWidget(QFrame):
    split_best_clicked = pyqtSignal(int)
    split_clicked = pyqtSignal(int)
    undo_split_clicked = pyqtSignal(int)
    blocked_changed = pyqtSignal(int)
    nan_route_changed = pyqtSignal(int, str)

    WIDTH = 300

    def __init__(self, node: Node, features: list[str],
                 class_colors: dict[str, QColor] | None = None,
                 parent=None):
        super().__init__(parent)
        self.node = node
        self.features = features
        self.class_colors = class_colors or {}
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(self.WIDTH)
        self.setStyleSheet(
            "NodeWidget { background: white; border: 1px solid #b6c2d2;"
            " border-radius: 8px; }"
            "QLabel#rule { color:#8a5a00; font-weight:bold; }"

            # --- NaN-надписи ---
            "QLabel#nan      { color:#b03030; font-weight:bold; }" 
            "QLabel#nan_none { color:#888888; }"                      #: серая "No NaN"

            # --- кнопка-шестерёнка смены маршрута NaN ---
            "QToolButton#btn_nan {"
            "    color:#b03030; border: 1px solid #c9a0a0;"
            "    border-radius: 4px; padding: 0 4px; }"               
            "QToolButton#btn_nan:hover { background:#ffe8e8; }"       

            # --- кнопки split (без изменений) ---
            "QPushButton#btn_best, QPushButton#btn_manual {"
            "    min-height: 28px;"
            "    background: #f0f0f0;"
            "    color: #333333;"
            "    border: 1px solid #a0a0a0;"
            "    border-radius: 6px;"
            "    padding: 4px 12px;"
            "}"
            "QPushButton#btn_best:hover, QPushButton#btn_manual:hover {"
            "    background: #e0e0e0;"
            "}"
            "QPushButton#btn_best:pressed, QPushButton#btn_manual:pressed {"
            "    background: #d0d0d0;"
            "}"

            # --- кнопка undo (без изменений) ---
            "QToolButton#btn_undo {"
            "    color: #a03030; font-weight: bold;"
            "    border: 1px solid #c9a0a0; border-radius: 4px;"
            "    padding: 1px 6px; }"
            "QToolButton#btn_undo:hover { background: #ffe8e8; }"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)

        # ---------- шапка: rule + undo + lock ----------
        head = QHBoxLayout()
        rule = QLabel(node.rule)
        rule.setObjectName("rule")
        rule.setWordWrap(True)
        head.addWidget(rule, stretch=1)

        # кнопка undo — видна только когда узел уже разбит
        self.undo_btn = QToolButton()
        self.undo_btn.setObjectName("btn_undo")
        self.undo_btn.setText("undo")
        self.undo_btn.setToolTip(
            "Undo this split — removes this node's split\n"
            "and every split below it, as if they never happened.")
        self.undo_btn.clicked.connect(
            lambda: self.undo_split_clicked.emit(node.id))
        self.undo_btn.setVisible(node.split is not None)
        head.addWidget(self.undo_btn)

        self.lock_btn = QToolButton()
        self.lock_btn.setText("🔒")
        self.lock_btn.setToolTip("Block features for split_best on this node")
        self.lock_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.lock_menu = QMenu(self.lock_btn)
        self._lock_actions = {}
        for f in features:
            act = self.lock_menu.addAction(f)
            act.setCheckable(True)
            act.setChecked(f in node.blocked_features)
            act.toggled.connect(lambda checked, feat=f: self._toggle(feat, checked))
            self._lock_actions[f] = act
        self.lock_btn.setMenu(self.lock_menu)
        head.addWidget(self.lock_btn)
        lay.addLayout(head)

        # ---------- метрика ----------
        s = node.summary
        n = s["n_rows"]
        leaf_badge = " <i>(leaf)</i>" if node.is_leaf else ""
        info = QLabel(
            f"<b>Observations: {n}</b>{leaf_badge} &nbsp;&nbsp; "
            f"{s['metric']}: {s['metric_value']:.6f}")
        info.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(info)
        info.setStyleSheet("color: black;")

        # ---------- предсказание + stacked bar ----------
        if "distribution" in s:
            dist = ", ".join(f"{k}: {v}" for k, v in
                             list(s["distribution"].items())[:6])
            pred = QLabel(f"class: {s.get('prediction')}   [{dist}]")
            pred.setWordWrap(True)
            lay.addWidget(pred)
            if self.class_colors and s["distribution"]:
                lay.addWidget(StackedBar(s["distribution"], self.class_colors))
        else:
            pred = QLabel(f"mean: {s.get('prediction'):.6g}   "
                          f"std: {s.get('std', 0):.6g}")
            pred.setWordWrap(True)
            lay.addWidget(pred)

        # ---------- NaN-строка ----------
        if node.split is not None:
            if node.nan_rows_feature == 0:
                nan_lbl = QLabel("No NaN")
                nan_lbl.setObjectName("nan_none")
                lay.addWidget(nan_lbl)
            else:
                pct = 100.0 * node.nan_rows_feature / max(node.n_rows, 1)
                row = QHBoxLayout()
                nan_lbl = QLabel(
                    f"NaN in {node.split.feature}: "
                    f"{node.nan_rows_feature} rows ({pct:.1f}%) "
                    f"→ {node.split.nan_route}")
                nan_lbl.setObjectName("nan")
                nan_lbl.setWordWrap(True)
                row.addWidget(nan_lbl, stretch=1)

                gear = QToolButton()
                gear.setObjectName("btn_nan")
                gear.setText("⚙")
                gear.setToolTip(
                    "Change how NaN rows are handled for this split.\n"
                    "This will re-apply the split (and remove the subtree below it).")
                gear.clicked.connect(lambda: self._show_nan_menu(gear))
                row.addWidget(gear)
                lay.addLayout(row)

        # ---------- кнопки split ----------
        btns = QHBoxLayout()
        self.btn_best = QPushButton("split_best")
        self.btn_manual = QPushButton("split")
        self.btn_best.setObjectName("btn_best")
        self.btn_manual.setObjectName("btn_manual")
        self.btn_best.clicked.connect(lambda: self.split_best_clicked.emit(node.id))
        self.btn_manual.clicked.connect(lambda: self.split_clicked.emit(node.id))
        btns.addWidget(self.btn_best)
        btns.addWidget(self.btn_manual)
        lay.addLayout(btns)

        if not node.is_leaf:
            self.btn_best.setEnabled(False)
            self.btn_manual.setEnabled(False)
        print(f"[DBG] node {node.id} is_leaf={node.is_leaf} "
            f"obs_text={info.text()!r} "
            f"sizeHint={info.sizeHint()} "
            f"widget_w={self.width()}")

    def _toggle(self, feature: str, checked: bool):
        if checked:
            self.node.blocked_features.add(feature)
        else:
            self.node.blocked_features.discard(feature)
        self.blocked_changed.emit(self.node.id)

    def _show_nan_menu(self, btn: QToolButton):
        """Меню смены маршрута NaN после уже сделанного split."""
        menu = QMenu(self)
        current = self.node.split.nan_route
        # nan_split: NaN и так выделен в отдельную ветку — менять нечего
        if self.node.split.kind == "nan_split":
            act = menu.addAction("Already split by NaN / not-NaN "
                                 "(nothing to change)")
            act.setEnabled(False)
            menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))
            return
        for route, label in (
            ("separate", "Separate child (NaN = 3rd branch, no data loss)"),
            ("left",     "Send NaN to LEFT child"),
            ("right",    "Send NaN to RIGHT child"),
            ("ignore",   "Ignore (DROP NaN rows from both children)"),
        ):
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(current == route)
            act.triggered.connect(
                lambda checked, r=route: self.nan_route_changed.emit(self.node.id, r))
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))