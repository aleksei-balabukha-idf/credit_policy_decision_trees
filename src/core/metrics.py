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
