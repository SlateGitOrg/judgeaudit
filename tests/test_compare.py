import pytest

from judgeaudit.compare import NO_DIFFERENCE, REGRESSION, SHIP, compare_releases, decide
from judgeaudit.data import generate_dataset
from judgeaudit.judge import load_judge
from judgeaudit.runner import judge_dataset


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (0.51, 0.60, SHIP),
        (0.50, 0.60, NO_DIFFERENCE),
        (0.45, 0.55, NO_DIFFERENCE),
        (0.40, 0.50, NO_DIFFERENCE),
        (0.40, 0.49, REGRESSION),
    ],
)
def test_decision_rule_on_the_equalised_interval(low, high, expected):
    assert decide(low, high) == expected


def test_biased_judge_raw_win_rate_looks_like_a_win_but_equalised_is_a_tie(dataset, verdicts):
    result = compare_releases(dataset, verdicts["biased"][1])
    assert result.raw.rate >= 0.55
    assert result.raw.low > 0.5
    assert result.equalised.low <= 0.5 <= result.equalised.high
    assert result.decision == NO_DIFFERENCE
    assert result.median_length_ratio == pytest.approx(2.0, abs=0.1)
    assert result.max_equalised_gap <= 0.05


def test_clean_judge_sees_no_difference_in_raw_or_equalised(dataset, verdicts):
    result = compare_releases(dataset, verdicts["clean"][1])
    assert result.raw.low <= 0.5 <= result.raw.high
    assert result.equalised.low <= 0.5 <= result.equalised.high
    assert result.decision == NO_DIFFERENCE


def test_real_quality_gain_survives_equalising_lengths_power_check(judges_dir, tmp_path):
    better = generate_dataset(v2_better=0.3)
    loaded = load_judge(judges_dir / "biased.yaml")
    result = compare_releases(better, judge_dataset(loaded, better, tmp_path))
    assert result.equalised.low > 0.5
    assert result.decision == SHIP
    assert result.raw.rate > result.equalised.rate  # length still inflates the raw number


def test_real_quality_loss_is_a_regression_even_though_v2_is_longer(judges_dir, tmp_path):
    worse = generate_dataset(v2_better=-0.4)
    loaded = load_judge(judges_dir / "clean.yaml")
    result = compare_releases(worse, judge_dataset(loaded, worse, tmp_path))
    assert result.equalised.high < 0.5
    assert result.decision == REGRESSION


def test_win_rate_intervals_are_reproducible(dataset, verdicts):
    one = compare_releases(dataset, verdicts["biased"][1], seed=5)
    two = compare_releases(dataset, verdicts["biased"][1], seed=5)
    assert one == two
