"""The joint logistic fit on padded pairs shared by the position and length audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from judgeaudit.models import PaddedPair
from judgeaudit.runner import ORDERS, Verdicts


@dataclass(frozen=True)
class Coefficient:
    """A point estimate with a 95% confidence interval and a two-sided test against zero."""

    estimate: float
    low: float
    high: float
    p_value: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    @property
    def excludes_zero(self) -> bool:
        return not self.contains(0.0)


@dataclass(frozen=True)
class JointFit:
    b_pos: Coefficient
    b_len: Coefficient
    n_pairs: int


def padded_rows(pairs: list[PaddedPair], verdicts: Verdicts) -> pd.DataFrame:
    """One row per (pair, order): did the answer shown first win, and by how much was it longer.

    Latent quality is identical within a pair, so the only things that can move the verdict are
    the position and the length difference ``ln(len_first) - ln(len_second)``.
    """
    records = []
    for p in pairs:
        for order in ORDERS:
            first, second = (p.original, p.padded) if order == "ab" else (p.padded, p.original)
            records.append(
                {
                    "pair_id": p.id,
                    "order": order,
                    "first_wins": int(verdicts.first_wins(p.id, order)),
                    "dln": float(np.log(first.n_words) - np.log(second.n_words)),
                }
            )
    return pd.DataFrame.from_records(records)


def fit_joint(rows: pd.DataFrame) -> JointFit:
    """Logit of P(first wins) on a constant (b_pos) and the log-length difference (b_len).

    Standard errors are clustered by pair, because the same pair is judged in both orders.
    """
    design = sm.add_constant(rows[["dln"]].to_numpy(dtype=float))
    groups = pd.factorize(rows["pair_id"])[0]
    result = sm.Logit(rows["first_wins"].to_numpy(dtype=float), design).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": groups}
    )
    ci = np.asarray(result.conf_int(alpha=0.05))
    params = np.asarray(result.params)
    pvals = np.asarray(result.pvalues)
    coefs = [
        Coefficient(float(params[i]), float(ci[i, 0]), float(ci[i, 1]), float(pvals[i]))
        for i in range(2)
    ]
    return JointFit(b_pos=coefs[0], b_len=coefs[1], n_pairs=int(rows["pair_id"].nunique()))
