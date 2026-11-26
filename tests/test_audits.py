import numpy as np
import pytest

from judgeaudit.audits._fit import fit_joint, padded_rows
from judgeaudit.audits.length import pointwise_length_slope
from judgeaudit.data import make_padded_pairs
from judgeaudit.judge import SimulatedJudge
from judgeaudit.runner import ORDERS, Verdicts


def verdicts_for_pairs(pairs, judge):
    values = {}
    for p in pairs:
        for order in ORDERS:
            first, second = (p.original, p.padded) if order == "ab" else (p.padded, p.original)
            values[(p.id, order)] = judge.compare(p.question, first, second, "r")
    return Verdicts(values)


def test_biased_judge_planted_position_and_length_biases_are_recovered(audits):
    position, length = audits["biased"].position, audits["biased"].length
    assert position.b_pos.contains(0.6) and position.b_pos.excludes_zero
    assert length.b_len.contains(1.2) and length.b_len.excludes_zero
    assert position.b_pos.p_value < 1e-6 and length.b_len.p_value < 1e-6


def test_clean_judge_has_no_detectable_bias_and_passes_every_gate(audits):
    result = audits["clean"]
    assert result.position.b_pos.contains(0.0)
    assert result.length.b_len.contains(0.0)
    assert [g.passed for g in result.gates] == [True, True, True]
    assert result.passed


def test_biased_judge_fails_the_position_and_length_gates(audits):
    result = audits["biased"]
    assert [g.name for g in result.gates if not g.passed] == ["Position bias", "Length bias"]
    assert not result.passed


@pytest.mark.parametrize(("b_pos", "b_len"), [(0.3, 0.0), (-0.5, 0.8), (0.0, 2.0)])
def test_other_planted_sizes_are_recovered_within_their_intervals(b_pos, b_len):
    pairs = make_padded_pairs(300, seed=21)
    judge = SimulatedJudge(seed=21, b_pos=b_pos, b_len=b_len)
    fit = fit_joint(padded_rows(pairs, verdicts_for_pairs(pairs, judge)))
    assert fit.b_pos.contains(b_pos)
    assert fit.b_len.contains(b_len)


def test_position_gate_false_alarm_rate_over_100_replicates_is_near_nominal():
    position_alarms = length_alarms = 0
    for rep in range(100):
        pairs = make_padded_pairs(300, seed=1000 + rep)
        judge = SimulatedJudge(seed=1000 + rep)
        fit = fit_joint(padded_rows(pairs, verdicts_for_pairs(pairs, judge)))
        position_alarms += fit.b_pos.excludes_zero
        length_alarms += fit.b_len.excludes_zero
    # Nominal 5%; P(more than 11 of 100 | p = 0.05) is under 1%.
    assert position_alarms <= 11
    assert length_alarms <= 11


def test_the_intervals_are_not_so_wide_that_they_hide_a_real_bias():
    detected = 0
    for rep in range(30):
        pairs = make_padded_pairs(300, seed=2000 + rep)
        judge = SimulatedJudge(seed=2000 + rep, b_pos=0.4)
        fit = fit_joint(padded_rows(pairs, verdicts_for_pairs(pairs, judge)))
        detected += fit.b_pos.excludes_zero
    assert detected >= 25  # power about 0.97 at this size


def test_padded_rows_encode_length_difference_with_opposite_signs_per_order(dataset, verdicts):
    rows = padded_rows(dataset.padded_pairs, verdicts["biased"][1])
    assert len(rows) == 600
    pair = rows[rows["pair_id"] == dataset.padded_pairs[0].id]
    ab, ba = pair.iloc[0], pair.iloc[1]
    assert ab["dln"] == pytest.approx(-ba["dln"])
    assert ab["dln"] < 0  # the original is shown first and is the shorter answer


def test_pointwise_length_slope_is_positive_for_the_biased_judge_and_zero_for_the_clean(
    dataset, verdicts
):
    biased = pointwise_length_slope(dataset, verdicts["biased"][1])
    clean = pointwise_length_slope(dataset, verdicts["clean"][1])
    assert biased.excludes_zero and biased.estimate == pytest.approx(0.5, abs=0.2)
    assert clean.contains(0.0)


def test_biased_judge_is_less_order_consistent_than_the_clean_judge(audits):
    biased, clean = audits["biased"].position, audits["clean"].position
    assert biased.consistency_matched < clean.consistency_matched - 0.01
    assert biased.n_matched == clean.n_matched == 600


def test_biased_judge_agrees_less_with_humans_than_the_clean_judge(audits):
    biased, clean = audits["biased"].agreement, audits["clean"].agreement
    assert biased.kappa < clean.kappa
    assert biased.score_kappa < clean.score_kappa


def test_agreement_gate_uses_08_times_the_human_ceiling(audits):
    agreement = audits["clean"].agreement
    assert agreement.human_ceiling == pytest.approx(0.65, abs=0.06)
    assert agreement.threshold == pytest.approx(0.8 * agreement.human_ceiling)
    assert agreement.ci_low < agreement.kappa < agreement.ci_high
    assert agreement.kappa > agreement.threshold


def test_agreement_gate_fails_a_judge_that_ignores_quality(dataset):
    from judgeaudit.audits.agreement import audit_agreement

    values = {}
    judge = SimulatedJudge(seed=3, b_q=0.0)
    for q in dataset.questions:
        values[(q.id, "ab")] = judge.compare(q.text, q.v1, q.v2, "r")
        values[(q.id, "ba")] = judge.compare(q.text, q.v2, q.v1, "r")
        for a in (q.v1, q.v2):
            values[(a.id, "pointwise")] = judge.score(q.text, a, "r")
    result = audit_agreement(dataset, Verdicts(values))
    assert result.fails
    assert abs(result.kappa) < 0.1
    assert np.isfinite(result.score_kappa)
