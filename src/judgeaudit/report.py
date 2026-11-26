"""Run the three audits and render the Markdown report with its pass/fail gates."""

from __future__ import annotations

from dataclasses import dataclass

from judgeaudit.audits._fit import Coefficient, fit_joint, padded_rows
from judgeaudit.audits.agreement import GATE_FRACTION, AgreementResult, audit_agreement
from judgeaudit.audits.length import LengthResult, audit_length
from judgeaudit.audits.position import PositionResult, audit_position
from judgeaudit.compare import ComparisonResult
from judgeaudit.models import Dataset
from judgeaudit.runner import Verdicts


@dataclass(frozen=True)
class Gate:
    name: str
    criterion: str
    estimate: str
    passed: bool


@dataclass(frozen=True)
class AuditResult:
    judge_name: str
    config_hash: str
    n_questions: int
    n_padded: int
    position: PositionResult
    length: LengthResult
    agreement: AgreementResult

    @property
    def gates(self) -> list[Gate]:
        return [
            Gate(
                "Position bias",
                "95% CI of b_pos excludes 0",
                fmt_coef(self.position.b_pos),
                not self.position.fails,
            ),
            Gate(
                "Length bias",
                "95% CI of b_len excludes 0",
                fmt_coef(self.length.b_len),
                not self.length.fails,
            ),
            Gate(
                "Agreement with humans",
                f"kappa >= {GATE_FRACTION} x human ceiling ({self.agreement.threshold:.3f})",
                f"{self.agreement.kappa:.3f} [{self.agreement.ci_low:.3f}, "
                f"{self.agreement.ci_high:.3f}]",
                not self.agreement.fails,
            ),
        ]

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)


def fmt_coef(c: Coefficient) -> str:
    return f"{c.estimate:+.3f} [{c.low:+.3f}, {c.high:+.3f}]"


def run_audit(
    dataset: Dataset, verdicts: Verdicts, judge_name: str, config_hash: str, seed: int = 7
) -> AuditResult:
    fit = fit_joint(padded_rows(dataset.padded_pairs, verdicts))
    return AuditResult(
        judge_name=judge_name,
        config_hash=config_hash,
        n_questions=len(dataset.questions),
        n_padded=len(dataset.padded_pairs),
        position=audit_position(dataset, verdicts, fit),
        length=audit_length(dataset, verdicts, fit),
        agreement=audit_agreement(dataset, verdicts, seed),
    )


def render_audit(result: AuditResult) -> str:
    failed = [g.name for g in result.gates if not g.passed]
    verdict = "PASS" if result.passed else f"FAIL ({len(failed)} of {len(result.gates)} gates)"
    pos, length, agr = result.position, result.length, result.agreement
    lines = [
        f"# Judge audit: {result.judge_name}",
        "",
        f"Judge config `{result.config_hash}`. Data: {result.n_questions} questions judged in "
        f"both orders, {result.n_padded} padded pairs (synthetic).",
        "",
        f"**Result: {verdict}**",
        "",
        "| Gate | Criterion | Estimate (95% CI) | Result |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {g.name} | {g.criterion} | {g.estimate} | {'pass' if g.passed else 'FAIL'} |"
        for g in result.gates
    ]
    lines += [
        "",
        "## Position",
        "",
        f"- b_pos: {fmt_coef(pos.b_pos)}, p = {pos.b_pos.p_value:.2g} "
        f"(positive favours the answer shown first)",
        f"- Order consistency on v1/v2 pairs: {pos.consistency_matched:.1%} "
        f"({pos.n_matched} pairs)",
        f"- Order consistency on padded pairs: {pos.consistency_padded:.1%} "
        f"({pos.n_padded} pairs; 50% is expected if each order's verdict is an independent "
        "fair coin flip: equal quality, no preference for either answer, no position preference)",
        "",
        "## Length",
        "",
        f"- b_len: {fmt_coef(length.b_len)}, p = {length.b_len.p_value:.2g} "
        f"(log-odds of winning per unit of ln length ratio)",
        f"- At twice the length, the longer answer wins {length.p_longer_wins_at_2x:.1%} "
        "of the time with quality equal and the order averaged",
        f"- Pointwise length slope: {fmt_coef(length.pointwise_slope)} score points per unit "
        "of ln length",
        "",
        "## Agreement",
        "",
        f"- Judge vs human majority, pairwise kappa: {agr.kappa:.3f} "
        f"[{agr.ci_low:.3f}, {agr.ci_high:.3f}]",
        f"- Mean pairwise human-human kappa (ceiling): {agr.human_ceiling:.3f}; "
        f"gate threshold {agr.threshold:.3f}",
        f"- Pointwise scores vs annotator panel, quadratic-weighted kappa: "
        f"{agr.score_kappa:.3f} [{agr.score_ci_low:.3f}, {agr.score_ci_high:.3f}]",
        "",
    ]
    return "\n".join(lines)


def render_comparison(result: ComparisonResult, judge_name: str) -> str:
    raw, eq = result.raw, result.equalised
    return "\n".join(
        [
            f"# Release comparison: v2 vs v1 ({judge_name})",
            "",
            f"{result.n_pairs} question pairs, each judged in both orders. The median v2/v1 "
            f"length ratio is {result.median_length_ratio:.2f}.",
            "",
            "| Win rate of v2 | Estimate | 95% CI |",
            "| --- | --- | --- |",
            f"| Raw | {raw.rate:.3f} | [{raw.low:.3f}, {raw.high:.3f}] |",
            f"| Length-equalised | {eq.rate:.3f} | [{eq.low:.3f}, {eq.high:.3f}] |",
            "",
            f"**Decision: {result.decision}**",
            "",
        ]
    )
