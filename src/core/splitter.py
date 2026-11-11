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
