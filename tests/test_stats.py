import numpy as np
import pytest

from judgeaudit.stats import (
    bootstrap_ci,
    cohen_kappa,
    kappa_from_confusion,
    mean_pairwise_kappa,
)


def test_kappa_for_the_reference_confusion_table_is_040():
    # Observed agreement 35/50 = 0.70; chance agreement (25*30 + 25*20) / 50^2 = 0.50.
    assert kappa_from_confusion(np.array([[20, 5], [10, 15]])) == pytest.approx(0.40)


def test_quadratic_weighted_kappa_matches_hand_computation():
    # Row sums 7, 7, 6; column sums 6, 7, 7; N = 20; weights (i - j)^2 / 4.
    # Observed weighted disagreement 1.5; expected 6.525; kappa = 1 - 1.5 / 6.525 = 67 / 87.
    table = np.array([[5, 2, 0], [1, 4, 2], [0, 1, 5]])
    assert kappa_from_confusion(table, "quadratic") == pytest.approx(67 / 87)


def test_weighted_kappa_penalises_far_misses_more_than_near_misses():
    near = np.array([[5, 5, 0], [0, 5, 5], [0, 0, 10]])
    far = np.array([[5, 0, 5], [0, 10, 0], [5, 0, 5]])
    assert kappa_from_confusion(near, "quadratic") > kappa_from_confusion(far, "quadratic")


def test_kappa_is_one_for_perfect_agreement_and_zero_for_independent_raters():
    labels = [0, 1, 0, 1, 1, 0]
    assert cohen_kappa(labels, labels, [0, 1]) == 1.0
    independent = np.array([[25, 25], [25, 25]])
    assert kappa_from_confusion(independent) == pytest.approx(0.0)


def test_kappa_is_symmetric_in_the_raters():
    a = [0, 1, 1, 0, 1, 1, 0, 0]
    b = [0, 1, 0, 0, 1, 1, 1, 0]
    assert cohen_kappa(a, b, [0, 1]) == pytest.approx(cohen_kappa(b, a, [0, 1]))


def test_mean_pairwise_kappa_averages_all_rater_pairs():
    r1, r2, r3 = [0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 1, 1]
    expected = np.mean(
        [cohen_kappa(r1, r2, [0, 1]), cohen_kappa(r1, r3, [0, 1]), cohen_kappa(r2, r3, [0, 1])]
    )
    assert mean_pairwise_kappa([r1, r2, r3], [0, 1]) == pytest.approx(expected)


def test_bootstrap_interval_brackets_the_mean_and_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = rng.normal(0.5, 1.0, 50)
    large = rng.normal(0.5, 1.0, 2000)

    def interval(x):
        return bootstrap_ci(len(x), lambda i: float(x[i].mean()), np.random.default_rng(1), 1000)

    lo_s, hi_s = interval(small)
    lo_l, hi_l = interval(large)
    assert lo_s < small.mean() < hi_s
    assert lo_l < large.mean() < hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s) / 4
