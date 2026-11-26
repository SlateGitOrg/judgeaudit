"""Length bias: does the judge favour the longer answer when quality is identical?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from scipy.special import expit

from judgeaudit.audits._fit import Coefficient, JointFit
from judgeaudit.models import Dataset
from judgeaudit.runner import Verdicts


@dataclass(frozen=True)
class LengthResult:
    b_len: Coefficient
    p_longer_wins_at_2x: float
    pointwise_slope: Coefficient
    n_pairs: int

    @property
    def fails(self) -> bool:
        """Gate: the 95% CI of b_len excludes 0."""
        return self.b_len.excludes_zero


def pointwise_length_slope(dataset: Dataset, verdicts: Verdicts) -> Coefficient:
    """OLS slope of the 1-5 score on ``ln(len)``, with a fixed effect per padded pair.

    Within a pair the quality is equal, so differencing the two scores removes it: the slope of
    ``score(padded) - score(original)`` on ``ln(len_padded) - ln(len_original)`` through the
    origin is the extra score a judge gives for length alone.
    """
    d_score = np.array(
        [verdicts.score(p.padded.id) - verdicts.score(p.original.id) for p in dataset.padded_pairs],
        dtype=float,
    )
    d_len = np.array(
        [np.log(p.padded.n_words) - np.log(p.original.n_words) for p in dataset.padded_pairs]
    )
    fit = sm.OLS(d_score, d_len).fit()
    low, high = np.asarray(fit.conf_int(alpha=0.05))[0]
    return Coefficient(float(fit.params[0]), float(low), float(high), float(fit.pvalues[0]))


def audit_length(dataset: Dataset, verdicts: Verdicts, fit: JointFit) -> LengthResult:
    shift = fit.b_len.estimate * np.log(2.0)
    p_2x = 0.5 * (expit(fit.b_pos.estimate + shift) + expit(-fit.b_pos.estimate + shift))
    return LengthResult(
        b_len=fit.b_len,
        p_longer_wins_at_2x=float(p_2x),
        pointwise_slope=pointwise_length_slope(dataset, verdicts),
        n_pairs=fit.n_pairs,
    )
