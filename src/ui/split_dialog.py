"""Manual split dialog: choose feature, cut-off / category subset / NaN-vs-filled,
and the NaN routing policy."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QRadioButton, QVBoxLayout, QWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from core.dtypes import TYPE_ICONS, TYPE_LABEL_COLOR, ColumnType
from core.splitter import NAN_POLICIES, SplitCandidate


class SplitDialog(QDialog):
    def __init__(self, df: pd.DataFrame, features: list[str],
                 types: dict[str, ColumnType], metric, blocked: set[str],
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual split")
        self.df, self.types, self.metric = df, types, metric
        self.features = [f for f in features if f not in blocked]
        self.resize(480, 560)

        root = QVBoxLayout(self)
        form = QFormLayout()
        self.feature_combo = QComboBox()
        for f in self.features:
            self.feature_combo.addItem(f"{TYPE_ICONS[types[f]]} {f}", f)
            idx = self.feature_combo.count() - 1
            self.feature_combo.setItemData(
                idx, QColor(TYPE_LABEL_COLOR), Qt.ItemDataRole.ForegroundRole)
        form.addRow("Feature:", self.feature_combo)
        self.nan_label = QLabel()
        form.addRow(self.nan_label)
        root.addLayout(form)

        # numeric area
        self.num_group = QGroupBox("Numeric split")
        nl = QVBoxLayout(self.num_group)
        self.rb_cut = QRadioButton("Cut-off")
        self.rb_cut.setChecked(True)
        self.rb_nan = QRadioButton("NaN vs filled values")
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setDecimals(6)
        self.cutoff.setRange(-1e15, 1e15)
        nl.addWidget(self.rb_cut)
        nl.addWidget(self.cutoff)
        nl.addWidget(self.rb_nan)
        root.addWidget(self.num_group)

        # categorical area
        self.cat_group = QGroupBox("Values going to the LEFT child "
                                   "(all others go right)")
        cl = QVBoxLayout(self.cat_group)
        self.cat_list = QListWidget()
        cl.addWidget(self.cat_list)
        root.addWidget(self.cat_group)

        # NaN policy
        self.nan_group = QGroupBox("NaN handling")
        gl = QVBoxLayout(self.nan_group)
        self.nan_radios: dict[str, QRadioButton] = {}
        labels = {
            "separate": "Separate NaN child (third branch — NaN rows stay)",
            "left":     "Send NaN to the LEFT child",
            "right":    "Send NaN to the RIGHT child",
            "ignore":   "Ignore — DROP NaN rows from both children "
                        "(they will NOT appear in this subtree)",
        }
        for key in ("separate", "left", "right", "ignore"):
            rb = QRadioButton(labels[key])
            self.nan_radios[key] = rb
            gl.addWidget(rb)
        self.nan_radios["separate"].setChecked(True)
        root.addWidget(self.nan_group)
        self.nan_hint = QLabel()
        self.nan_hint.setWordWrap(True)
        self.nan_hint.setStyleSheet(
            "color:#555; font-style:italic; background:#f5f7fa;"
            "padding:4px; border-radius:4px;")
        root.addWidget(self.nan_hint)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        self.feature_combo.currentIndexChanged.connect(self._on_feature)
        if self.features:
            self._on_feature(0)

    def _on_feature(self, _i):
        feat = self.feature_combo.currentData()
        if feat is None:
            return
        x = self.df[feat]
        n_nan = int(x.isna().sum())
        pct = 100.0 * n_nan / max(len(x), 1)
        self.nan_label.setText(f"NaN in feature: {n_nan} rows ({pct:.1f}%)")
        # если NaN нет — блок обработки неактуален
        self.nan_group.setEnabled(n_nan > 0)
        if n_nan == 0:
            self.nan_hint.setText(
                "No NaN in this feature — NaN handling does not apply.")
            self.nan_radios["ignore"].setChecked(True)
        else:
            if self.nan_radios["ignore"].isChecked():
                self.nan_radios["separate"].setChecked(True)
            self.nan_hint.setText(
                f"This feature has {n_nan} NaN rows ({pct:.1f}%). "
                "• Separate → they form a third child (no rows lost). "
                "• Left/Right → they join one of the two value children. "
                "• Ignore → they are dropped and will not appear in any child "
                "(the subtree below loses those observations).")
        is_num = self.types[feat] != ColumnType.CATEGORICAL
        self.num_group.setVisible(is_num)
        self.cat_group.setVisible(not is_num)
        if is_num:
            xv = pd.to_numeric(x, errors="coerce")
            if xv.notna().any():
                self.cutoff.setValue(float(xv.median()))
        else:
            self.cat_list.clear()
            for v in sorted(x.dropna().astype("string").unique().tolist()):
                item = QListWidgetItem(str(v))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.cat_list.addItem(item)

    def nan_route(self) -> str:
        for key, rb in self.nan_radios.items():
            if rb.isChecked():
                return key
        return "ignore"

    def build_candidate(self) -> SplitCandidate | None:
        feat = self.feature_combo.currentData()
        if feat is None:
            return None
        route = self.nan_route()
        is_num = self.types[feat] != ColumnType.CATEGORICAL
        x = self.df[feat]
        n_nan = int(x.isna().sum())
        if is_num and self.rb_nan.isChecked():
            return SplitCandidate(feat, "nan_split", nan_route=route,
                                  nan_rows=n_nan, metric=self.metric)
        if is_num:
            return SplitCandidate(feat, "numeric_cutoff",
                                  cutoff=self.cutoff.value(),
                                  nan_route=route, nan_rows=n_nan,
                                  metric=self.metric)
        cats = [self.cat_list.item(i).text()
                for i in range(self.cat_list.count())
                if self.cat_list.item(i).checkState() == Qt.CheckState.Checked]
        if not cats:
            return None
        return SplitCandidate(feat, "categorical_subset", categories=cats,
                              nan_route=route, nan_rows=n_nan,
                              metric=self.metric)
