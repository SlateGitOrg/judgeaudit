"""Release comparison: v2 against v1, raw and with lengths equalised."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from judgeaudit.data import equalise_lengths
from judgeaudit.models import Dataset
from judgeaudit.runner import ORDERS, Verdicts
from judgeaudit.seeding import stable_rng
from judgeaudit.stats import bootstrap_ci

N_BOOT = 2000
SHIP = "ship"
NO_DIFFERENCE = "no significant difference"
REGRESSION = "regression"


@dataclass(frozen=True)
class WinRate:
    rate: float
    low: float
    high: float


@dataclass(frozen=True)
class ComparisonResult:
    raw: WinRate
    equalised: WinRate
    decision: str
    n_pairs: int
    median_length_ratio: float
    max_equalised_gap: float


def decide(ci_low: float, ci_high: float) -> str:
    """Release decision from the interval of the length-equalised v2 win rate."""
    if ci_low > 0.5:
        return SHIP
    if ci_high < 0.5:
        return REGRESSION
    return NO_DIFFERENCE


def pair_win_rates(pair_ids: list[str], verdicts: Verdicts) -> np.ndarray:
    """v2's win rate on each pair, averaged over both presentation orders (0, 0.5 or 1)."""
    return np.array(
        [np.mean([verdicts.side_b_wins(p, o) for o in ORDERS]) for p in pair_ids], dtype=float
    )


def win_rate(rates: np.ndarray, seed: int, label: str) -> WinRate:
    """Mean win rate with a percentile bootstrap interval over pairs."""
    low, high = bootstrap_ci(
        len(rates), lambda idx: float(rates[idx].mean()), stable_rng(seed, label), N_BOOT
    )
    return WinRate(float(rates.mean()), low, high)


def compare_releases(dataset: Dataset, verdicts: Verdicts, seed: int = 7) -> ComparisonResult:
    """Raw v2 win rate, then the win rate after padding the shorter answer of each pair.

    The padded pairs differ only in quality, so the second number is the release decision with
    the length advantage taken away.
    """
    raw = win_rate(pair_win_rates([q.id for q in dataset.questions], verdicts), seed, "raw")
    equalised_pairs = equalise_lengths(dataset)
    equalised = win_rate(pair_win_rates([e.id for e in equalised_pairs], verdicts), seed, "eq")
    ratios = [
        max(q.v1.n_words, q.v2.n_words) / min(q.v1.n_words, q.v2.n_words) for q in dataset.questions
    ]
    gaps = [
        abs(e.v1.n_words - e.v2.n_words) / max(e.v1.n_words, e.v2.n_words) for e in equalised_pairs
    ]
    return ComparisonResult(
        raw=raw,
        equalised=equalised,
        decision=decide(equalised.low, equalised.high),
        n_pairs=len(dataset.questions),
        median_length_ratio=float(np.median(ratios)),
        max_equalised_gap=float(max(gaps)),
    )
