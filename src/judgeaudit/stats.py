"""Cohen's kappa and the percentile bootstrap."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Literal

import numpy as np

Weights = Literal["none", "quadratic"]


def kappa_from_confusion(confusion: np.ndarray, weights: Weights = "none") -> float:
    """Cohen's kappa from a square confusion table (rows: rater A, columns: rater B).

    With ``weights="quadratic"`` disagreement between categories i and j costs
    ``(i - j)^2 / (k - 1)^2``, so near misses on an ordinal scale count less than far ones.
    Returns 1.0 when there is no disagreement to explain (a constant, perfectly matched pair of
    raters) and 0.0 when the raters never agree beyond chance by construction.
    """
    table = np.asarray(confusion, dtype=float)
    k = table.shape[0]
    n = table.sum()
    if n == 0:
        return 0.0
    idx = np.arange(k)
    if weights == "quadratic":
        cost = (idx[:, None] - idx[None, :]) ** 2 / max(k - 1, 1) ** 2
    else:
        cost = (idx[:, None] != idx[None, :]).astype(float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    observed_cost = (cost * table).sum()
    expected_cost = (cost * expected).sum()
    if expected_cost == 0.0:
        return 1.0 if observed_cost == 0.0 else 0.0
    return float(1.0 - observed_cost / expected_cost)


def confusion_matrix(a: Sequence[int], b: Sequence[int], labels: Sequence[int]) -> np.ndarray:
    lookup = {label: i for i, label in enumerate(labels)}
    table = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b, strict=True):
        table[lookup[x], lookup[y]] += 1
    return table


def cohen_kappa(
    a: Sequence[int], b: Sequence[int], labels: Sequence[int], weights: Weights = "none"
) -> float:
    return kappa_from_confusion(confusion_matrix(a, b, labels), weights)


def mean_pairwise_kappa(raters: Sequence[Sequence[int]], labels: Sequence[int]) -> float:
    """Mean Cohen's kappa over every pair of raters."""
    values = [cohen_kappa(a, b, labels) for a, b in combinations(raters, 2)]
    return float(np.mean(values))


def bootstrap_ci(
    n_items: int,
    statistic: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    n_boot: int = 2000,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling item indices with replacement.

    ``statistic`` receives an index array and returns the statistic of that resample.
    """
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = statistic(rng.integers(0, n_items, size=n_items))
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(low), float(high)
