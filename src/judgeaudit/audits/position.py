"""Position bias: does the judge favour whichever answer is shown first?"""

from __future__ import annotations

from dataclasses import dataclass

from judgeaudit.audits._fit import Coefficient, JointFit
from judgeaudit.models import Dataset
from judgeaudit.runner import Verdicts


@dataclass(frozen=True)
class PositionResult:
    consistency_matched: float
    n_matched: int
    consistency_padded: float
    n_padded: int
    b_pos: Coefficient

    @property
    def fails(self) -> bool:
        """Gate: the 95% CI of b_pos excludes 0."""
        return self.b_pos.excludes_zero


def order_consistency(pair_ids: list[str], verdicts: Verdicts) -> float:
    """Share of pairs where the same answer wins whichever order it is shown in."""
    same = sum(verdicts.side_b_wins(p, "ab") == verdicts.side_b_wins(p, "ba") for p in pair_ids)
    return same / len(pair_ids)


def audit_position(dataset: Dataset, verdicts: Verdicts, fit: JointFit) -> PositionResult:
    """Order-consistency on both pair sets, and b_pos from the padded-pair fit.

    b_pos is the intercept of the logit on padded pairs, where quality is equal by construction
    and the length difference is a covariate, so it is not confounded with either.
    """
    matched = [q.id for q in dataset.questions]
    padded = [p.id for p in dataset.padded_pairs]
    return PositionResult(
        consistency_matched=order_consistency(matched, verdicts),
        n_matched=len(matched),
        consistency_padded=order_consistency(padded, verdicts),
        n_padded=len(padded),
        b_pos=fit.b_pos,
    )
