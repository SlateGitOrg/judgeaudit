"""Agreement with human annotators, with the human-human kappa as the ceiling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from judgeaudit.models import Dataset
from judgeaudit.runner import Verdicts
from judgeaudit.seeding import stable_rng
from judgeaudit.stats import bootstrap_ci, kappa_from_confusion, mean_pairwise_kappa

N_BOOT = 2000
GATE_FRACTION = 0.8


@dataclass(frozen=True)
class AgreementResult:
    kappa: float
    ci_low: float
    ci_high: float
    human_ceiling: float
    threshold: float
    score_kappa: float
    score_ci_low: float
    score_ci_high: float
    n_pairs: int

    @property
    def fails(self) -> bool:
        """Gate: judge-human-majority kappa is below 0.8 x the mean human-human kappa."""
        return self.kappa < self.threshold


def _pooled_confusion(j_ab: np.ndarray, j_ba: np.ndarray, human: np.ndarray) -> np.ndarray:
    """2x2 table of judge verdict against human majority, both orders pooled."""
    table = np.zeros((2, 2))
    for judged in (j_ab, j_ba):
        table += np.bincount(judged * 2 + human, minlength=4).reshape(2, 2)
    return table


def audit_agreement(dataset: Dataset, verdicts: Verdicts, seed: int = 7) -> AgreementResult:
    """Cohen's kappa of the judge's pairwise verdicts against the human majority.

    Each pair contributes its verdict in both orders (1 = v2 preferred), so a judge that flips
    with position loses agreement. The interval is a percentile bootstrap over pairs. Pointwise
    scores are compared with the annotator panel's 1-5 scores by quadratic-weighted kappa.
    """
    qs = dataset.questions
    j_ab = np.array([int(verdicts.side_b_wins(q.id, "ab")) for q in qs])
    j_ba = np.array([int(verdicts.side_b_wins(q.id, "ba")) for q in qs])
    human = np.array([q.human_majority for q in qs])

    def pairwise(idx: np.ndarray) -> float:
        return kappa_from_confusion(_pooled_confusion(j_ab[idx], j_ba[idx], human[idx]))

    everyone = np.arange(len(qs))
    kappa = pairwise(everyone)
    low, high = bootstrap_ci(len(qs), pairwise, stable_rng(seed, "kappa"), N_BOOT)

    raters = [[q.human_labels[k] for q in qs] for k in range(len(qs[0].human_labels))]
    ceiling = mean_pairwise_kappa(raters, labels=[0, 1])

    answers = [a for q in qs for a in (q.v1, q.v2)]
    judged = np.array([verdicts.score(a.id) for a in answers]) - 1
    reference = np.array([a.human_score for a in answers]) - 1

    def scores(idx: np.ndarray) -> float:
        table = np.bincount(judged[idx] * 5 + reference[idx], minlength=25).reshape(5, 5)
        return kappa_from_confusion(table.astype(float), "quadratic")

    score_kappa = scores(np.arange(len(answers)))
    s_low, s_high = bootstrap_ci(len(answers), scores, stable_rng(seed, "score-kappa"), N_BOOT)
    return AgreementResult(
        kappa=kappa,
        ci_low=low,
        ci_high=high,
        human_ceiling=ceiling,
        threshold=GATE_FRACTION * ceiling,
        score_kappa=score_kappa,
        score_ci_low=s_low,
        score_ci_high=s_high,
        n_pairs=len(qs),
    )
