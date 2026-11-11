# PROJECT BUNDLE: decision_tree_app_5

# Generated for AI context.

##################################################



##################################################

# FILE: src\core\dtypes.py | SECTION: CORE

##################################################


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



##################################################

# FILE: src\core\metrics.py | SECTION: CORE

##################################################


"""Split-quality metrics.

Classification: Gini (default), Entropy (information gain).
Regression: variance reduction (MSE).
All functions operate on 1-D arrays of the target variable.
"""
from __future__ import annotations

import enum

import numpy as np
import pandas as pd


class Metric(str, enum.Enum):
    GINI = "gini"
    ENTROPY = "entropy"
    MSE = "mse"          # regression (variance)
    MEAN = "mean"        # regression: split maximises |difference of means|-style gain


def validate_metric_vs_target(metric_name: str, y: pd.Series) -> tuple[bool, str]:
    """
    Проверяет совместимость метрики и целевой переменной.
    Возвращает (is_valid, warning_message).
    """
    # list of metricsm which can work with continuous variables (regression)
    REGRESSION_METRICS = {"mse", "mae", "variance"}
    # list of metrics for classification
    CLASSIFICATION_METRICS = {"gini", "entropy"}
    metric = metric_name.lower().strip()
    is_target_numeric = pd.api.types.is_numeric_dtype(y)
    # case when Target is categorical, but metric is used for Regression
    if not is_target_numeric and metric in REGRESSION_METRICS:
        return False, (
            f"Error: Chosen metric: '{metric_name}', which can be used for Regression and not Classification."
            f"Your target variable is of type: '{y.dtype}' (category).\n\n"
            "Solution: \n"
            "1. Change metric to Gini or Entropy (Classification).\n"
            "2. Or change data type for target variable (to numeric) \n."
            "3. Or change target variable"
        )
    # If target cariable is numeric, but Metric is Gini or Entropy (classification) (can be missleading)
    if is_target_numeric and metric in CLASSIFICATION_METRICS:
        # Additionally: если в числовой цели мало уникальных значений, это может быть категория
        if y.nunique(dropna=True) <= 10:
            return False, (
                f"Предупреждение: Целевая переменная числовая, но имеет мало уникальных значений ({y.nunique()}). "
                f"Выбрана метрика классификации '{metric_name}'.\n\n"
                "Решение: \n"
                "1. Если это категории, закодированные числами (0, 1, 2) — всё в порядке.\n"
                "2. Если это непрерывные числа — лучше использовать MSE."
            )    
    return True, ""


def is_classification_target(y: pd.Series) -> bool:
    """Classification if categorical/bool or few unique values."""
    if isinstance(y.dtype, pd.StringDtype) or pd.api.types.is_bool_dtype(y):
        return True
    if pd.api.types.is_numeric_dtype(y):
        return y.nunique(dropna=True) <= 20
    return True


def default_metric(y: pd.Series) -> Metric:
    return Metric.GINI if is_classification_target(y) else Metric.MSE


def gini(y: np.ndarray) -> float:
    y = y[~pd.isna(y)]
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - (p ** 2).sum())


def entropy(y: np.ndarray) -> float:
    y = y[~pd.isna(y)]
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def variance(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    return float(np.var(y)) if len(y) else 0.0


def impurity(y: pd.Series | np.ndarray, metric: Metric) -> float:
    arr = np.asarray(y)
    if metric == Metric.GINI:
        return gini(arr)
    if metric == Metric.ENTROPY:
        return entropy(arr)
    return variance(arr)


def gain(parent: np.ndarray, left: np.ndarray, right: np.ndarray,
         metric: Metric) -> float:
    """Impurity decrease; NaN rows of the target are ignored per-child."""
    n, nl, nr = len(parent), len(left), len(right)
    if n == 0 or nl == 0 or nr == 0:
        return -np.inf
    return (impurity(parent, metric)
            - nl / n * impurity(left, metric)
            - nr / n * impurity(right, metric))


def node_summary(y: pd.Series, metric: Metric) -> dict:
    """What a leaf shows: n, metric value, prediction / class distribution."""
    valid = y.dropna()
    out = {"n_rows": int(len(y)), "n_nan_target": int(y.isna().sum()),
           "metric": metric.value, "metric_value": impurity(y.to_numpy(), metric)}
    if metric in (Metric.GINI, Metric.ENTROPY):
        vc = valid.astype("string").value_counts()
        out["prediction"] = str(vc.index[0]) if len(vc) else None
        out["distribution"] = {str(k): int(v) for k, v in vc.items()}
    else:
        vals = pd.to_numeric(valid, errors="coerce").dropna()
        out["prediction"] = float(vals.mean()) if len(vals) else None
        out["std"] = float(vals.std()) if len(vals) > 1 else 0.0
    return out



##################################################

# FILE: src\core\splitter.py | SECTION: CORE

##################################################


"""Exhaustive best-split search and split application.

Strategy: Exhaustive (slow, accurate) — every feature, every candidate
cut-off (numeric) or category ordering boundary (categorical, Breiman
ordering which is optimal for binary targets and a strong heuristic
otherwise).

NaN policy for split_best: NaN rows are evaluated going left or right
(whichever gains more); 'ignore' and 'separate child' are available for
manual splits. Optionally NaN can be treated as its own category.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .dtypes import ColumnType
from .metrics import Metric, gain, impurity, is_classification_target

log = logging.getLogger("dtree.split")

NAN_POLICIES = ("left", "right", "separate", "ignore")


class CancelledError(Exception):
    """Raised inside split_best when the user presses Cancel."""


@dataclass
class SplitCandidate:
    feature: str
    kind: str                       # numeric_cutoff | categorical_subset | nan_split
    cutoff: float | None = None
    categories: list | None = None  # values going LEFT (categorical)
    nan_route: str = "ignore"       # left | right | separate | ignore
    nan_rows: int = 0
    gain_value: float = -np.inf
    metric: Metric = Metric.GINI

    def condition(self, side: str) -> str:
        """Human-readable rule leading to a child ('left'/'right'/'separate')."""
        if self.kind == "numeric_cutoff":
            if side == "left":
                return f"{self.feature} <= {self.cutoff:g}"
            if side == "right":
                return f"{self.feature} > {self.cutoff:g}"
            return f"{self.feature} is NaN"
        if self.kind == "nan_split":
            if side == "left":
                return f"{self.feature} is NaN"
            return f"{self.feature} is not NaN"
        cats = ", ".join(map(str, self.categories or []))
        if side == "left":
            return f"{self.feature} in {{{cats}}}"
        if side == "right":
            return f"{self.feature} not in {{{cats}}}"
        return f"{self.feature} is NaN"

    def to_dict(self) -> dict:
        return {"feature": self.feature, "kind": self.kind, "cutoff": self.cutoff,
                "categories": self.categories, "nan_route": self.nan_route,
                "nan_rows": self.nan_rows, "gain": self.gain_value,
                "metric": self.metric.value}

    @classmethod
    def from_dict(cls, d: dict) -> "SplitCandidate":
        return cls(feature=d["feature"], kind=d["kind"], cutoff=d.get("cutoff"),
                   categories=d.get("categories"), nan_route=d.get("nan_route", "ignore"),
                   nan_rows=d.get("nan_rows", 0), gain_value=d.get("gain", -np.inf),
                   metric=Metric(d.get("metric", "gini")))


# ---------------------------------------------------------------- masks

def split_masks(x: pd.Series, cand: SplitCandidate) -> dict[str, np.ndarray]:
    """Boolean masks for left/right (and NaN) given a split spec."""
    nan = x.isna().to_numpy()
    if cand.kind == "numeric_cutoff":
        xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64")
        left = (xv <= cand.cutoff) & ~nan
        right = (xv > cand.cutoff) & ~nan
    elif cand.kind == "nan_split":
        left = nan.copy()
        right = ~nan
    else:  # categorical_subset
        xs = x.astype("string")
        cats = pd.array([str(c) for c in cand.categories], dtype="string")
        isin = xs.isin(cats).to_numpy()
        left = isin & ~nan
        right = ~isin & ~nan

    if cand.kind == "nan_split":
        return {"left": left, "right": right, "nan": np.zeros(len(x), bool)}
    if cand.nan_route == "left":
        left = left | nan
    elif cand.nan_route == "right":
        right = right | nan
    # 'ignore' -> NaN dropped; 'separate' -> third child
    return {"left": left, "right": right,
            "nan": nan if cand.nan_route == "separate" else np.zeros(len(x), bool)}


def apply_split(df: pd.DataFrame, cand: SplitCandidate) -> dict:
    masks = split_masks(df[cand.feature], cand)
    log.info("split applied: %s | left=%d right=%d nan=%d",
             cand.condition("left"), masks["left"].sum(),
             masks["right"].sum(), masks["nan"].sum())
    return {"left": df[masks["left"]], "right": df[masks["right"]],
            "separate": df[masks["nan"]] if masks["nan"].any() else None,
            "nan_rows": int(df[cand.feature].isna().sum())}


# ------------------------------------------------------- numeric search

def _best_numeric_cutoff(x: np.ndarray, y: np.ndarray, metric: Metric,
                         min_leaf: int) -> tuple[float | None, float]:
    """Exhaustive over all unique-value boundaries. Vectorised cumulatives."""
    valid = ~np.isnan(x) & ~pd.isna(y)
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 2 * min_leaf or np.unique(x).size < 2:
        return None, -np.inf

    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    if metric in (Metric.GINI, Metric.ENTROPY):
        classes, y_idx = np.unique(y, return_inverse=True)
        K = len(classes)
        cum = np.zeros((n + 1, K))
        cum[1:] = np.cumsum(np.eye(K)[y_idx], axis=0)
        total = cum[-1]
        bounds = np.flatnonzero(x[1:] != x[:-1]) + 1      # first idx of new value
        bounds = bounds[(bounds >= min_leaf) & (bounds <= n - min_leaf)]
        if bounds.size == 0:
            return None, -np.inf
        cl, cr = cum[bounds], total - cum[bounds]
        if metric == Metric.GINI:
            imp = lambda c: 1.0 - ((c / c.sum(axis=1, keepdims=True)) ** 2).sum(axis=1)
        else:
            p = lambda c: np.where(c > 0, c / c.sum(axis=1, keepdims=True), 0)
            imp = lambda c: -(np.where(p(c) > 0, p(c) * np.log2(p(c)), 0)).sum(axis=1)
        imp_p = imp(total[None, :])[0]
        gains = imp_p - (bounds / n) * imp(cl) - ((n - bounds) / n) * imp(cr)
        i = int(np.argmax(gains))
        b = bounds[i]
        return float((x[b - 1] + x[b]) / 2), float(gains[i])

    # regression: variance reduction via cumulated sums
    yv = y.astype("float64")
    cs, cs2 = np.concatenate(([0.0], np.cumsum(yv))), np.concatenate(([0.0], np.cumsum(yv * yv)))
    bounds = np.flatnonzero(x[1:] != x[:-1]) + 1
    bounds = bounds[(bounds >= min_leaf) & (bounds <= n - min_leaf)]
    if bounds.size == 0:
        return None, -np.inf

    def var(sum_, sum2, cnt):
        v = sum2 / cnt - (sum_ / cnt) ** 2
        return np.maximum(v, 0.0)

    tot_s, tot_s2 = cs[-1], cs2[-1]
    imp_p = var(tot_s, tot_s2, n)
    bl = bounds
    gains = (imp_p
             - bl / n * var(cs[bl], cs2[bl], bl)
             - (n - bl) / n * var(tot_s - cs[bl], tot_s2 - cs2[bl], n - bl))
    i = int(np.argmax(gains))
    b = bl[i]
    return float((x[b - 1] + x[b]) / 2), float(gains[i])


# ---------------------------------------------------- categorical search

def _best_categorical_subset(x: pd.Series, y: np.ndarray, metric: Metric,
                             min_leaf: int) -> tuple[list | None, float]:
    """Order categories by target rate/mean, scan k-1 binary splits."""
    valid = x.notna() & ~pd.isna(y)
    xv, yv = x[valid].astype("string"), y[valid.to_numpy()]
    if len(xv) < 2 * min_leaf:
        return None, -np.inf
    cats = xv.unique().tolist()
    if len(cats) < 2:
        return None, -np.inf
    if metric in (Metric.GINI, Metric.ENTROPY):
        if is_classification_target(pd.Series(yv)):
            pos = (yv == pd.Series(yv).mode(dropna=True).iloc[0]).astype(float)
        else:
            pos = pd.to_numeric(pd.Series(yv), errors="coerce").to_numpy()
    else:
        pos = pd.to_numeric(pd.Series(yv), errors="coerce").to_numpy()
    stats = pd.DataFrame({"cat": xv, "score": pos}).groupby("cat")["score"].mean()
    ordered = stats.sort_values().index.tolist()

    best_gain, best_subset = -np.inf, None
    for i in range(1, len(ordered)):
        subset = ordered[:i]
        mask = xv.isin(subset).to_numpy()
        nl, nr = int(mask.sum()), int((~mask).sum())
        if nl < min_leaf or nr < min_leaf:
            continue
        g = gain(yv, yv[mask], yv[~mask], metric)
        if g > best_gain:
            best_gain, best_subset = g, [str(c) for c in subset]
    return best_subset, best_gain


# ------------------------------------------------------------ best_split

def best_split(df: pd.DataFrame, features: list[str],
               types: dict[str, ColumnType], target: str, metric: Metric,
               min_leaf: int, blocked: set[str] | None = None,
               nan_as_category: bool = False,
               progress_cb: Callable[[int, int], None] | None = None,
               cancel_cb: Callable[[], bool] | None = None) -> SplitCandidate | None:
    """Exhaustive search across all (non-blocked) features.

    Returns the best SplitCandidate or None if nothing is splittable.
    """
    blocked = blocked or set()
    y = df[target].to_numpy()
    best: SplitCandidate | None = None
    total = len(features)

    for i, feat in enumerate(features):
        if cancel_cb and cancel_cb():
            raise CancelledError()
        if progress_cb:
            progress_cb(i + 1, total)
        if feat in blocked:
            log.debug("feature %s blocked on this node, skipped", feat)
            continue
        x = df[feat]
        nan_rows = int(x.isna().sum())
        cand = SplitCandidate(feature=feat, kind="", nan_rows=nan_rows, metric=metric)

        if types.get(feat) == ColumnType.CATEGORICAL:
            subset, g = _best_categorical_subset(x, y, metric, min_leaf)
            if subset is None:
                continue
            cand.kind, cand.categories, cand.gain_value = "categorical_subset", subset, g
            cand.nan_route = "ignore"
        else:
            xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64")
            cut, g = _best_numeric_cutoff(xv, y, metric, min_leaf)
            nan_mask = np.isnan(xv)
            n_total = len(y)
            best_route, best_g = "ignore", g

            if nan_mask.any() and cut is not None:
                lm0 = (xv <= cut) & ~nan_mask
                rm0 = (xv > cut) & ~nan_mask

                # --- left / right ---
                for route in ("left", "right"):
                    lm = lm0 | nan_mask if route == "left" else lm0
                    rm = rm0 | nan_mask if route == "right" else rm0
                    if lm.sum() < min_leaf or rm.sum() < min_leaf:
                        continue
                    gr = gain(y, y[lm], y[rm], metric)
                    if gr > best_g:
                        best_g, best_route = gr, route

                # --- separate (3-way: L, R, NaN-ребёнок) ---
                if (lm0.sum() >= min_leaf and rm0.sum() >= min_leaf
                        and nan_mask.sum() >= min_leaf):
                    g_sep = (
                        impurity(y, metric)
                        - (lm0.sum() / n_total) * impurity(y[lm0], metric)
                        - (rm0.sum() / n_total) * impurity(y[rm0], metric)
                        - (nan_mask.sum() / n_total) * impurity(y[nan_mask], metric)
                    )
                    if g_sep > best_g:
                        best_g, best_route = g_sep, "separate"

            # --- nan_split: сам факт NaN как предиктор ---
            nan_split_gain = -np.inf
            if nan_mask.any() and nan_mask.sum() >= min_leaf and (~nan_mask).sum() >= min_leaf:
                nan_split_gain = gain(y, y[nan_mask], y[~nan_mask], metric)

            if cut is None and nan_split_gain == -np.inf:
                continue
            if nan_split_gain > best_g:
                cand.kind, cand.gain_value, cand.nan_route = "nan_split", nan_split_gain, "ignore"
            else:
                cand.kind, cand.cutoff = "numeric_cutoff", cut
                cand.gain_value, cand.nan_route = best_g, best_route

        if cand.gain_value > -np.inf and (best is None or cand.gain_value > best.gain_value):
            best = cand
            log.debug("new best: %s gain=%.6f", cand.condition("left"), cand.gain_value)

    if best:
        log.info("split_best winner: %s (gain=%.6f, NaN rows=%d, route=%s)",
                 best.condition("left"), best.gain_value, best.nan_rows, best.nan_route)
    else:
        log.info("split_best: no valid split found")
    return best



##################################################

# FILE: src\core\tree.py | SECTION: CORE

##################################################


"""Tree model: nodes, constraints, splitting, serialization."""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import pandas as pd

from .dtypes import ColumnType
from .metrics import Metric, node_summary
from .splitter import SplitCandidate, apply_split

log = logging.getLogger("dtree.tree")


class SplitNotAllowed(Exception):
    """Split blocked by constraints (min_leaf, max_depth, degenerate split)."""


@dataclass
class Node:
    id: int
    depth: int
    n_rows: int
    summary: dict                                   # metric value, prediction, distribution
    rule: str = "ROOT"                              # condition leading to this leaf
    blocked_features: set[str] = field(default_factory=set)
    split: SplitCandidate | None = None
    nan_rows_feature: int = 0                       # NaN count of the split feature
    children: dict[str, "Node"] = field(default_factory=dict)  # left/right/separate

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def to_dict(self) -> dict:
        return {"id": self.id, "depth": self.depth, "n_rows": self.n_rows,
                "summary": self.summary, "rule": self.rule,
                "blocked_features": sorted(self.blocked_features),
                "split": self.split.to_dict() if self.split else None,
                "children": {k: c.to_dict() for k, c in self.children.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        node = cls(id=d["id"], depth=d["depth"], n_rows=d["n_rows"],
                   summary=d["summary"], rule=d.get("rule", "ROOT"),
                   blocked_features=set(d.get("blocked_features", [])),
                   split=(SplitCandidate.from_dict(d["split"]) if d.get("split") else None))
        node.children = {k: cls.from_dict(c) for k, c in d.get("children", {}).items()}
        return node


class DecisionTreeModel:
    """Holds data + tree structure. GUI talks only to this class."""

    def __init__(self, df: pd.DataFrame, types: dict[str, ColumnType],
                 target: str, features: list[str], metric: Metric,
                 min_leaf: int = 20, max_depth: int = 6):
        if target not in df.columns:
            raise SplitNotAllowed(f"Target column '{target}' is missing.")
        self.df = df
        self.types = types
        self.target = target
        self.features = features
        self.metric = metric
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self._ids = itertools.count()
        self.root = self._make_node(df, depth=0, rule="ROOT")
        self.nodes: dict[int, tuple[Node, pd.DataFrame]] = {}
        self._store(self.root, df)

        # Warning (not a blocker): single-class classification target
        self.warnings: list[str] = []
        # Warnings emitted by the most recent split_node() call, e.g.
        # "NaN leaf < min_leaf, but allowed". UI reads them right after.
        self.last_split_warnings: list[str] = []
        if metric in (Metric.GINI, Metric.ENTROPY) and df[target].nunique(dropna=True) <= 1:
            self.warnings.append(
                f"Target '{target}' has a single class — the tree cannot "
                "separate anything, but you may still split manually.")
            log.warning("single-class target: %s", target)
        log.info("root created: n=%d metric=%s value=%.6f",
                 len(df), metric.value, self.root.summary["metric_value"])

    # ------------------------------------------------------------- core

    def _make_node(self, df: pd.DataFrame, depth: int, rule: str) -> Node:
        return Node(id=next(self._ids), depth=depth, n_rows=len(df),
                    summary=node_summary(df[self.target], self.metric), rule=rule)

    def _store(self, node: Node, df: pd.DataFrame) -> None:
        self.nodes[node.id] = (node, df)

    def get_data(self, node_id: int) -> pd.DataFrame:
        return self.nodes[node_id][1]

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id][0]

    def check_can_split(self, node: Node, cand: SplitCandidate) -> list[str]:
        """Raise SplitNotAllowed if the split cannot be performed.

        Returns a list of user-facing warnings (may be empty) for splits
        that ARE allowed but deserve attention — e.g. a NaN-only leaf
        smaller than min_leaf, which is permitted because it isolates
        missing values into their own branch.
        """
        if node.depth >= self.max_depth:
            raise SplitNotAllowed(
                f"Max depth {self.max_depth} reached — node {node.id} cannot be split.")

        masks_df = self.get_data(node.id)
        res = apply_split(masks_df, cand)

        warnings: list[str] = []
        is_nan_split = (cand.kind == "nan_split")

        # ---- LEFT ----
        # nan_split: left == NaN-only leaf -> soft (warning, not block)
        # other kinds: regular value branch -> hard
        if len(res["left"]) < self.min_leaf:
            if is_nan_split:
                warnings.append(
                    f"NaN leaf has only {len(res['left'])} rows "
                    f"(< min leaf {self.min_leaf}). Split is still allowed "
                    f"because NaN is isolated into its own branch.")
            else:
                raise SplitNotAllowed(
                    f"Left child would have {len(res['left'])} rows "
                    f"(< min leaf {self.min_leaf}).")

        # ---- RIGHT ----
        # nan_split: right == non-NaN branch (the "real" data) -> hard.
        # other kinds: regular value branch -> hard.
        if len(res["right"]) < self.min_leaf:
            if is_nan_split:
                raise SplitNotAllowed(
                    f"Leaf without NaN rows would have {len(res['right'])} rows "
                    f"(< min leaf {self.min_leaf}). A NaN split is allowed only "
                    f"when the non-NaN branch is large enough.")
            raise SplitNotAllowed(
                f"Right child would have {len(res['right'])} rows "
                f"(< min leaf {self.min_leaf}).")

        # ---- SEPARATE (third NaN child for nan_route="separate") ----
        # Always a NaN-only leaf -> soft (warning, not block).
        if res["separate"] is not None and len(res["separate"]) > 0:
            if len(res["separate"]) < self.min_leaf:
                warnings.append(
                    f"Separate NaN leaf has only {len(res['separate'])} rows "
                    f"(< min leaf {self.min_leaf}). Split is still allowed "
                    f"because NaN is isolated into its own branch.")

        if len(res["left"]) == 0 or len(res["right"]) == 0:
            raise SplitNotAllowed("Split is degenerate (one side is empty).")

        return warnings

    def split_node(self, node_id: int, cand: SplitCandidate) -> list[Node]:
        node = self.get_node(node_id)
        df = self.get_data(node_id)
        self.last_split_warnings = self.check_can_split(node, cand)
        for w in self.last_split_warnings:
            log.warning("split on node %d: %s", node_id, w)
        res = apply_split(df, cand)
        node.split = cand
        node.nan_rows_feature = res["nan_rows"]

        created = []
        for side, sub in (("left", res["left"]), ("right", res["right"])):
            child = self._make_node(sub, node.depth + 1, cand.condition(side))
            node.children[side] = child
            self._store(child, sub)
            created.append(child)
        if res["separate"] is not None and len(res["separate"]):
            child = self._make_node(res["separate"], node.depth + 1,
                                    cand.condition("separate"))
            node.children["separate"] = child
            self._store(child, res["separate"])
            created.append(child)
        log.info("node %d split -> children %s", node_id,
                 [(c.id, c.n_rows) for c in created])
        return created

    def undo_split(self, node_id: int) -> list[int]:
        """Remove the split made on this node AND delete the whole subtree.

        After this call the node is a leaf again, exactly as if split_node
        had never been called on it: node.split is None, node.children is
        empty, and all descendant nodes are unregistered from self.nodes.
        blocked_features of *this* node are preserved (user intent).

        Returns the list of descendant node ids that were removed
        (the node itself is kept, of course).
        """
        node = self.get_node(node_id)
        if node.split is None:
            return []

        # breadth-first collection of all descendants
        removed: list[int] = []
        stack = list(node.children.values())
        while stack:
            n = stack.pop()
            removed.append(n.id)
            stack.extend(n.children.values())

        for nid in removed:
            self.nodes.pop(nid, None)

        node.children.clear()
        node.split = None
        node.nan_rows_feature = 0
        log.info("split undone on node %d — removed %d descendant node(s)",
                 node_id, len(removed))
        return removed

    # ------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        return {"target": self.target, "features": self.features,
                "metric": self.metric.value, "min_leaf": self.min_leaf,
                "max_depth": self.max_depth,
                "types": {k: v.value for k, v in self.types.items()},
                "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, df: pd.DataFrame, d: dict) -> "DecisionTreeModel":
        model = cls(df, {k: ColumnType(v) for k, v in d["types"].items()},
                    d["target"], d["features"], Metric(d["metric"]),
                    d["min_leaf"], d["max_depth"])
        model.root = Node.from_dict(d["root"])
        # rebuild node->data mapping by replaying splits
        model.nodes = {}
        model._rebuild(model.root, df)
        return model

    def _rebuild(self, node: Node, df: pd.DataFrame) -> None:
        self._store(node, df)
        if node.split:
            res = apply_split(df, node.split)
            for side, sub in (("left", res["left"]), ("right", res["right"]),
                              ("separate", res["separate"])):
                if side in node.children and sub is not None:
                    self._rebuild(node.children[side], sub)
        max_id = max(self.nodes) if self.nodes else -1
        self._ids = itertools.count(max_id + 1)



##################################################

# FILE: src\core\session.py | SECTION: CORE

##################################################


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



##################################################

# FILE: src\core\csv_loader.py | SECTION: CORE

##################################################


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



##################################################

# FILE: src\core\__init__.py | SECTION: CORE

##################################################





##################################################

# FILE: src\ui\__init__.py | SECTION: UI

##################################################





##################################################

# FILE: src\ui\class_visuals.py | SECTION: UI

##################################################


"""Class-visualization helpers: palette, stacked bar."""
from __future__ import annotations

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


CLASS_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def build_class_colors(model) -> dict[str, QColor]:
    """Stable class -> color mapping derived from the root distribution.

    Returns an empty dict for regression trees (no `distribution` key in
    the root summary), which the widgets use as a signal to skip the bar.
    """
    summary = model.root.summary
    if "distribution" not in summary:
        return {}
    classes = sorted(summary["distribution"].keys())
    return {cls: QColor(CLASS_PALETTE[i % len(CLASS_PALETTE)])
            for i, cls in enumerate(classes)}


class StackedBar(QWidget):
    """
    Horizontal stacked bar: one colored segment per class.
    Теперь поддерживает подписи долей в процентах с адаптивным цветом текста.
    """

    def __init__(self, distribution: dict[str, int], colors: dict[str, QColor], parent=None):
        super().__init__(parent)
        self._colors = colors
        total = sum(distribution.values()) or 1
        
        # Сортируем для стабильного порядка (как в легенде)
        self._parts = [(cls, v / total) for cls, v in sorted(distribution.items())]
        
        # Минимальная высота для читаемости текста
        self.setFixedHeight(24)  
        self.setMinimumWidth(140)

    def paintEvent(self, event):
        from PyQt6.QtGui import QFontMetrics
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        # Фон бара (опционально, для четких границ)
        bg_color = QColor("#f5f7fa")
        painter.fillRect(0, 0, w, h, bg_color)
        
        x = 0.0
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        fm = QFontMetrics(font)
        
        for cls, frac in self._parts:
            seg_w = w * frac
            color = self._colors.get(cls, QColor("#cccccc"))
            
            # Рисуем сегмент
            painter.fillRect(QRectF(x, 0, seg_w, h), color)
            
            # --- НОВОЕ: Подпись доли ---
            if seg_w > 12:  # Не рисуем текст, если сегмент слишком узкий
                pct = frac * 100
                
                # Форматируем: для малых долей (NPL) показываем 1 знак, для больших — без лишних нулей
                if pct < 10:
                    text = f"{pct:.1f}%"
                else:
                    text = f"{pct:.0f}%"
                
                text_w = fm.horizontalAdvance(text)
                text_h = fm.height()
                
                # Вычисляем центр сегмента
                text_x = x + (seg_w - text_w) / 2
                text_y = (h + text_h) / 2 - 2  # -2 для оптического центрирования
                
                # Выбираем цвет текста: черный на светлом, белый на темном
                # Порог яркости 0.5 в HSV или просто проверка компонентов RGB
                r, g, b = color.red(), color.green(), color.blue()
                brightness = (r * 299 + g * 587 + b * 114) / 1000
                
                if brightness > 140:
                    text_color = QColor("#222222")  # Темный текст на светлом фоне
                else:
                    text_color = QColor("#ffffff")  # Светлый текст на темном фоне
                
                painter.setPen(text_color)
                painter.drawText(int(text_x), int(text_y), text)
            
            x += seg_w
            
        # Рамка всего виджета
        painter.setPen(QPen(QColor("#666666"), 1))
        painter.drawRect(0, 0, w - 1, h - 1)
        painter.end()


##################################################

# FILE: src\ui\csv_preview.py | SECTION: UI

##################################################


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



##################################################

# FILE: src\ui\log_panel.py | SECTION: UI

##################################################


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



##################################################

# FILE: src\ui\main_window.py | SECTION: UI

##################################################


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


##################################################

# FILE: src\ui\node_widget.py | SECTION: UI

##################################################


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


##################################################

# FILE: src\ui\split_dialog.py | SECTION: UI

##################################################


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



##################################################

# FILE: src\ui\tree_canvas.py | SECTION: UI

##################################################


"""Tree canvas: QGraphicsView with zoom / pan / PNG export and a simple
tidy level-based layout."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QGraphicsLineItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsTextItem,
                             QGraphicsView)

from core.tree import DecisionTreeModel, Node
from .node_widget import NodeWidget
from .class_visuals import build_class_colors

NODE_W, NODE_H = NodeWidget.WIDTH + 20, 210       # стартовые оценки
HGAP, VGAP = 40, 90


class TreeCanvas(QGraphicsView):
    split_best_clicked = pyqtSignal(int)
    split_clicked = pyqtSignal(int)
    undo_split_clicked = pyqtSignal(int)
    nan_route_changed = pyqtSignal(int, str)          

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)   # pan by drag
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.model: DecisionTreeModel | None = None

    # --------------------------------------------------------- navigation

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def zoom_fit(self):
        if self._scene.items():
            self.fitInView(self._scene.itemsBoundingRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)

    # ------------------------------------------------------------- render

    def rebuild(self, model: DecisionTreeModel):
        self.model = model
        self._scene.clear()

        # 1) цвета классов из распределения в корне
        colors = build_class_colors(model)

        # 2) первичная раскладка по слотам / глубине
        pos: dict[int, tuple[float, float]] = {}
        self._layout(model.root, [0], pos)

        # 3) создаём карточки, запоминаем реальные высоты
        proxies: dict[int, object] = {}
        heights: dict[int, int] = {}
        for nid, (x, y) in pos.items():
            node = model.get_node(nid)
            w = NodeWidget(node, model.features, colors)
            w.split_best_clicked.connect(self.split_best_clicked)
            w.split_clicked.connect(self.split_clicked)
            w.undo_split_clicked.connect(self.undo_split_clicked)
            w.nan_route_changed.connect(self.nan_route_changed)
            proxy = self._scene.addWidget(w)
            proxy.setPos(x, y)
            proxies[nid] = proxy
            heights[nid] = w.sizeHint().height()

        # 4) пересчёт Y — карточки разной высоты не должны пересекаться
        depth_h: dict[int, int] = {}
        for nid in pos:
            d = model.get_node(nid).depth
            depth_h[d] = max(depth_h.get(d, 0), heights[nid])
        depth_y = {0: 0.0}
        prev = 0
        for d in sorted(depth_h):
            if d != 0:
                depth_y[d] = depth_y[prev] + depth_h[prev] + VGAP
            prev = d
        for nid, (x, _y) in list(pos.items()):
            d = model.get_node(nid).depth
            new_y = depth_y.get(d, 0.0)
            pos[nid] = (x, new_y)
            proxies[nid].setPos(x, new_y)

        # 5) рёбра + подписи
        self._draw_edges(model.root, pos, heights)

        # 6) легенда над корнем — прямыми items, без proxy-виджета
        if colors:
            rx, _ry = pos[model.root.id]
            self._add_legend(colors, rx, -70.0)

        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(
            -50, -50, 50, 50))

    # --------------------------------------------------------------------

    def _layout(self, node: Node, counter: list[int],
                pos: dict[int, tuple[float, float]]) -> tuple[float, float]:
        """In-order slot assignment; returns (x_start_slot, x_end_slot)."""
        if node.is_leaf:
            slot = counter[0]
            counter[0] += 1
            pos[node.id] = (slot * (NODE_W + HGAP),
                            node.depth * (NODE_H + VGAP))
            return slot, slot
        start, end = None, None
        for child in node.children.values():
            s, e = self._layout(child, counter, pos)
            start = s if start is None else min(start, s)
            end = e if end is None else max(end, e)
        x = ((start + end) / 2) * (NODE_W + HGAP)
        pos[node.id] = (x, node.depth * (NODE_H + VGAP))
        return start, end

    def _draw_edges(self, node: Node, pos, heights: dict[int, int]):
        x1, y1 = pos[node.id]
        h1 = heights.get(node.id, NODE_H)
        for side, child in node.children.items():
            x2, y2 = pos[child.id]
            line = QGraphicsLineItem(x1 + NODE_W / 2, y1 + h1,
                                     x2 + NODE_W / 2, y2)
            line.setPen(QPen(QColor("#7f8fa6"), 2))
            self._scene.addItem(line)
            label = QGraphicsTextItem(child.rule)
            label.setDefaultTextColor(QColor("#555f6e"))
            label.setPos((x1 + x2) / 2 + NODE_W / 4 - 40,
                         (y1 + h1 + y2) / 2 - 24)
            self._scene.addItem(label)
            self._draw_edges(child, pos, heights)

    def _add_legend(self, colors: dict, x: float, y: float) -> None:
        """Draw the class legend as direct scene items.

        Background is sized from the ACTUAL laid-out content: we place
        title/labels/swatches first, measure the rightmost edge, and only
        then draw the background behind them (setZValue(-1)). This avoids
        the previous bug where the width formula and the layout disagreed
        on the number of `gap`s and clipped the last class label.
        """
        from PyQt6.QtGui import QBrush, QFont

        pad, gap, swatch = 10.0, 8.0, 14.0
        text_color = QColor("#222222")
        border_pen = QPen(QColor("#b6c2d2"), 1)

        # ---- title ------------------------------------------------
        title = QGraphicsTextItem("Classes:")
        f = QFont(title.font())
        f.setBold(True)
        title.setFont(f)
        title.setDefaultTextColor(text_color)
        title.setPos(x + pad, y + pad)
        self._scene.addItem(title)

        title_w = title.boundingRect().width()
        title_h = title.boundingRect().height()
        row_h = max(swatch, title_h)

        # ---- one swatch + label per class -------------------------
        cur_x = x + pad + title_w + gap * 2
        for cls in colors:
            # swatch, vertically centered in the row
            rect = QGraphicsRectItem(
                cur_x, y + pad + (row_h - swatch) / 2, swatch, swatch)
            rect.setBrush(QBrush(colors[cls]))
            rect.setPen(QPen(QColor("#666666"), 1))
            self._scene.addItem(rect)
            cur_x += swatch + gap

            # label, vertically centered in the row
            lbl = QGraphicsTextItem(str(cls))
            lbl.setDefaultTextColor(text_color)
            lbl_h = lbl.boundingRect().height()
            lbl.setPos(cur_x, y + pad + (row_h - lbl_h) / 2)
            self._scene.addItem(lbl)
            cur_x += lbl.boundingRect().width() + gap * 2

        # ---- background from real content bounds ------------------
        # cur_x currently points past the trailing gap*2 of the last
        # label; drop that trailing gap, then add `pad` on the right.
        content_right = cur_x - gap * 2
        total_w = (content_right - x) + pad
        total_h = row_h + pad * 2

        bg = QGraphicsRectItem(x, y, total_w, total_h)
        bg.setBrush(QBrush(QColor("#ffffff")))
        bg.setPen(border_pen)
        bg.setZValue(-1)                      # behind title / labels / swatches
        self._scene.addItem(bg)

    # ------------------------------------------------------------ export

    def export_png(self, path: str) -> None:
        rect = self._scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        img = QImage(int(rect.width()), int(rect.height()),
                     QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.white)
        painter = QPainter(img)
        self._scene.render(painter, target=rect)
        painter.end()
        img.save(path, "PNG")


##################################################

# FILE: src\ui\workers.py | SECTION: UI

##################################################


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



##################################################

# FILE: src\main.py | SECTION: ENTRY POINT

##################################################


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



##################################################

# FILE: tests\test_csv_loader.py | SECTION: TESTS

##################################################


import tempfile
import unittest
from pathlib import Path

from core.csv_loader import CSVLoadError, CSVOptions, load_csv, load_preview, sniff_options
from core.dtypes import ColumnType, date_to_int, infer_type

import pandas as pd


class TestCsvLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text, encoding="utf-8"):
        p = self.dir / name
        p.write_text(text, encoding=encoding)
        return p

    def test_sniff_defaults_utf8_comma(self):
        p = self.write("a.csv", "x,y\n1,2\n3,4\n")
        opts = sniff_options(p)
        self.assertEqual(opts.encoding, "utf-8")
        self.assertEqual(opts.delimiter, ",")
        self.assertEqual(opts.decimal, ".")

    def test_sniff_semicolon_decimal_comma(self):
        p = self.write("b.csv", "x;y\n1,5;2\n3,25;4\n", encoding="cp1251")
        opts = sniff_options(p)
        self.assertEqual(opts.delimiter, ";")
        self.assertEqual(opts.decimal, ",")

    def test_missing_file(self):
        with self.assertRaises(CSVLoadError):
            sniff_options(self.dir / "nope.csv")

    def test_empty_file(self):
        p = self.write("empty.csv", "")
        with self.assertRaises(CSVLoadError):
            sniff_options(p)

    def test_preview_20_rows(self):
        rows = "a,b\n" + "\n".join(f"{i},cat{i % 3}" for i in range(100))
        p = self.write("big.csv", rows)
        df, types = load_preview(p, CSVOptions(), n_rows=20)
        self.assertEqual(len(df), 20)
        self.assertEqual(types["a"], ColumnType.NUMERIC)
        self.assertEqual(types["b"], ColumnType.CATEGORICAL)

    def test_load_date_to_int(self):
        p = self.write("d.csv", "d,x\n2025-12-29,1\n2026-01-01,2\n")
        df, types = load_csv(p, CSVOptions())
        self.assertEqual(types["d"], ColumnType.DATE)
        self.assertEqual(int(df["d"].iloc[0]), 20251229)
        self.assertEqual(int(df["d"].iloc[1]), 20260101)

    def test_type_override(self):
        p = self.write("o.csv", "x\n1\n2\n")
        df, types = load_csv(p, CSVOptions(), {"x": ColumnType.CATEGORICAL})
        self.assertEqual(types["x"], ColumnType.CATEGORICAL)
        self.assertTrue(isinstance(df["x"].dtype, pd.StringDtype))

    def test_date_int_conversion(self):
        s = pd.Series(["2025-12-29", "bad", None])
        out = date_to_int(s)
        self.assertEqual(int(out.iloc[0]), 20251229)
        self.assertTrue(out.iloc[1] is pd.NA or pd.isna(out.iloc[1]))

    def test_infer_bool_is_categorical(self):
        self.assertEqual(infer_type(pd.Series([True, False])), ColumnType.CATEGORICAL)


if __name__ == "__main__":
    unittest.main()



##################################################

# FILE: tests\test_metrics.py | SECTION: TESTS

##################################################


import unittest

import numpy as np
import pandas as pd

from core.metrics import Metric, entropy, gini, impurity, variance, node_summary


class TestMetrics(unittest.TestCase):
    def test_gini_pure(self):
        self.assertAlmostEqual(gini(np.array([1, 1, 1])), 0.0)

    def test_gini_balanced(self):
        self.assertAlmostEqual(gini(np.array([0, 0, 1, 1])), 0.5)

    def test_entropy_balanced(self):
        self.assertAlmostEqual(entropy(np.array(["a", "a", "b", "b"])), 1.0)

    def test_variance(self):
        self.assertAlmostEqual(variance(np.array([1.0, 2.0, 3.0])), np.var([1, 2, 3]))

    def test_impurity_ignores_nan(self):
        self.assertAlmostEqual(gini(np.array([0, 0, np.nan], dtype=object)), 0.0)

    def test_node_summary_classification(self):
        s = node_summary(pd.Series(["a", "a", "b", None], dtype="string"), Metric.GINI)
        self.assertEqual(s["n_rows"], 4)
        self.assertEqual(s["n_nan_target"], 1)
        self.assertEqual(s["prediction"], "a")
        self.assertEqual(s["distribution"], {"a": 2, "b": 1})

    def test_node_summary_regression(self):
        s = node_summary(pd.Series([1.0, 3.0]), Metric.MSE)
        self.assertEqual(s["prediction"], 2.0)
        self.assertEqual(s["n_rows"], 2)


if __name__ == "__main__":
    unittest.main()



##################################################

# FILE: tests\test_splitter.py | SECTION: TESTS

##################################################


import unittest

import numpy as np
import pandas as pd

from core.dtypes import ColumnType
from core.metrics import Metric
from core.splitter import (CancelledError, SplitCandidate, apply_split,
                           best_split, split_masks)


def make_separable(n=200, cut=0.5):
    rng = np.random.default_rng(42)
    x = rng.random(n)
    y = (x < cut).astype(int)  # perfect separation at 0.5
    return pd.DataFrame({"x": x, "noise": rng.random(n), "y": y})


TYPES = {"x": ColumnType.NUMERIC, "noise": ColumnType.NUMERIC}


class TestSplitter(unittest.TestCase):
    def test_recovers_known_cutoff(self):
        df = make_separable()
        cand = best_split(df, ["x", "noise"], TYPES, "y",
                          Metric.GINI, min_leaf=10)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.feature, "x")
        self.assertAlmostEqual(cand.cutoff, 0.5, delta=0.02)
        self.assertGreater(cand.gain_value, 0.4)

    def test_blocked_feature_not_used(self):
        df = make_separable()
        cand = best_split(df, ["x", "noise"], TYPES, "y",
                          Metric.GINI, min_leaf=10, blocked={"x"})
        if cand is not None:
            self.assertNotEqual(cand.feature, "x")

    def test_min_leaf_blocks_split(self):
        df = make_separable(n=40)
        cand = best_split(df, ["x"], TYPES, "y", Metric.GINI, min_leaf=30)
        self.assertIsNone(cand)  # 40 rows can't give two 30-row leaves

    def test_categorical_subset(self):
        rng = np.random.default_rng(1)
        n = 300
        cat = rng.choice(["good", "bad", "mid"], n)
        y = pd.Series(cat).isin(["bad"]).astype(int).to_numpy()
        df = pd.DataFrame({"c": pd.array(cat, dtype="string"), "y": y})
        cand = best_split(df, ["c"], {"c": ColumnType.CATEGORICAL}, "y",
                          Metric.GINI, min_leaf=10)
        self.assertEqual(cand.kind, "categorical_subset")
        self.assertEqual(cand.categories, ["bad"])
        self.assertGreater(cand.gain_value, 0.4)

    def test_regression_split(self):
        x = np.arange(200, dtype=float)
        y = np.where(x < 100, 0.0, 10.0)
        df = pd.DataFrame({"x": x, "y": y})
        cand = best_split(df, ["x"], TYPES, "y", Metric.MSE, min_leaf=10)
        self.assertAlmostEqual(cand.cutoff, 99.5, delta=1.0)
        self.assertGreater(cand.gain_value, 20.0)

    def test_nan_routing_left_right(self):
        df = make_separable()
        df.loc[df.sample(20, random_state=0).index, "x"] = np.nan
        cand = best_split(df, ["x"], TYPES, "y", Metric.GINI, min_leaf=5)
        self.assertIn(cand.nan_route, ("left", "right", "ignore", "separate"))
        self.assertEqual(cand.nan_rows, 20)

    def test_nan_as_category(self):
        n = 200
        x = np.random.default_rng(3).random(n)
        y = np.zeros(n, dtype=int)
        nan_idx = np.arange(n) % 4 == 0
        y[nan_idx] = 1
        df = pd.DataFrame({"x": np.where(nan_idx, np.nan, x), "y": y})
        cand = best_split(df, ["x"], TYPES, "y", Metric.GINI, min_leaf=10,
                          nan_as_category=True)
        self.assertEqual(cand.kind, "nan_split")
        m = split_masks(df["x"], cand)
        self.assertTrue(df.loc[m["left"], "y"].eq(1).all())

    def test_apply_split_separate_nan(self):
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "y": [0, 0, 1, 1]})
        cand = SplitCandidate("x", "numeric_cutoff", cutoff=2.5,
                              nan_route="separate")
        res = apply_split(df, cand)
        self.assertEqual(len(res["left"]), 2)
        self.assertEqual(len(res["right"]), 1)
        self.assertEqual(len(res["separate"]), 1)
        self.assertEqual(res["nan_rows"], 1)

    def test_cancellation(self):
        df = make_separable()
        with self.assertRaises(CancelledError):
            best_split(df, ["x"], TYPES, "y", Metric.GINI, 10,
                       cancel_cb=lambda: True)

    def test_progress_callback(self):
        df = make_separable()
        seen = []
        best_split(df, ["x", "noise"], TYPES, "y", Metric.GINI, 10,
                   progress_cb=lambda i, n: seen.append((i, n)))
        self.assertEqual(seen[-1], (2, 2))

    def test_performance_100k(self):
        import time
        rng = np.random.default_rng(7)
        n = 100_000
        data = {f"f{i}": rng.random(n) for i in range(10)}
        data["y"] = (data["f0"] < 0.5).astype(int)
        df = pd.DataFrame(data)
        types = {f"f{i}": ColumnType.NUMERIC for i in range(10)}
        t0 = time.perf_counter()
        cand = best_split(df, list(types), types, "y", Metric.GINI, 50)
        dt = time.perf_counter() - t0
        self.assertEqual(cand.feature, "f0")
        self.assertLess(dt, 10.0, f"split_best too slow: {dt:.1f}s")
        print(f"\n100k x 10 numeric split_best: {dt:.2f}s")


if __name__ == "__main__":
    unittest.main()



##################################################

# FILE: tests\test_tree_session.py | SECTION: TESTS

##################################################


import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.csv_loader import CSVOptions
from core.dtypes import ColumnType
from core.metrics import Metric
from core.session import SessionError, load_session, save_session
from core.splitter import SplitCandidate, best_split
from core.tree import DecisionTreeModel, SplitNotAllowed


def make_model(n=200):
    rng = np.random.default_rng(5)
    x = rng.random(n)
    df = pd.DataFrame({"x": x, "y": (x < 0.5).astype(int)})
    types = {"x": ColumnType.NUMERIC, "y": ColumnType.NUMERIC}
    return DecisionTreeModel(df, types, "y", ["x"], Metric.GINI,
                             min_leaf=10, max_depth=3)


class TestTree(unittest.TestCase):
    def test_root_created(self):
        m = make_model()
        self.assertEqual(m.root.n_rows, 200)
        self.assertEqual(m.root.depth, 0)
        self.assertIn("metric_value", m.root.summary)

    def test_single_class_warning_not_blocker(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1, 1, 1, 1]})
        m = DecisionTreeModel(df, {"x": ColumnType.NUMERIC}, "y", ["x"],
                              Metric.GINI, min_leaf=1, max_depth=2)
        self.assertTrue(m.warnings)  # warning present but model exists

    def test_split_and_rules(self):
        m = make_model()
        cand = best_split(m.get_data(m.root.id), ["x"],
                          {"x": ColumnType.NUMERIC}, "y", Metric.GINI, 10)
        kids = m.split_node(m.root.id, cand)
        self.assertEqual(len(kids), 2)
        self.assertIn("<=", kids[0].rule)
        self.assertIn(">", kids[1].rule)
        self.assertEqual(kids[0].depth, 1)

    def test_max_depth_message(self):
        m = make_model()
        m.max_depth = 0
        cand = SplitCandidate("x", "numeric_cutoff", cutoff=0.5)
        with self.assertRaises(SplitNotAllowed) as ctx:
            m.split_node(m.root.id, cand)
        self.assertIn("Max depth", str(ctx.exception))

    def test_min_leaf_message(self):
        m = make_model()
        cand = SplitCandidate("x", "numeric_cutoff", cutoff=0.999)
        with self.assertRaises(SplitNotAllowed) as ctx:
            m.split_node(m.root.id, cand)
        self.assertIn("min leaf", str(ctx.exception))

    def test_missing_target(self):
        df = pd.DataFrame({"x": [1.0]})
        with self.assertRaises(SplitNotAllowed):
            DecisionTreeModel(df, {"x": ColumnType.NUMERIC}, "y", ["x"],
                              Metric.GINI)


class TestSession(unittest.TestCase):
    def test_roundtrip(self):
        m = make_model()
        cand = best_split(m.get_data(m.root.id), ["x"],
                          {"x": ColumnType.NUMERIC}, "y", Metric.GINI, 10)
        m.split_node(m.root.id, cand)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.tree"
            save_session(m, p, CSVOptions(), "orig.csv", ["log line 1"])
            m2, opts, src, logs = load_session(p)
        self.assertEqual(m2.root.n_rows, 200)
        self.assertEqual(len(m2.root.children), 2)
        self.assertEqual(m2.target, "y")
        self.assertEqual(src, "orig.csv")
        self.assertEqual(logs, ["log line 1"])
        # rebuilt data mapping consistent
        kid = next(iter(m2.root.children.values()))
        self.assertEqual(m2.get_data(kid.id).shape[0], kid.n_rows)

    def test_corrupt_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.tree"
            p.write_text("not a zip")
            with self.assertRaises(SessionError):
                load_session(p)


if __name__ == "__main__":
    unittest.main()
