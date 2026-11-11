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
