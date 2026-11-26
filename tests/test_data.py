import hashlib
from collections import Counter

import numpy as np
import pytest

from judgeaudit.data import equalise_lengths, generate_dataset, make_padded_pairs
from judgeaudit.stats import mean_pairwise_kappa


def digest(dataset):
    return hashlib.sha256(dataset.model_dump_json().encode()).hexdigest()


def test_default_dataset_has_the_documented_shape(dataset):
    assert len(dataset.questions) == 600
    assert len(dataset.padded_pairs) == 300
    assert set(Counter(q.category for q in dataset.questions).values()) == {75}
    assert len({q.category for q in dataset.questions}) == 8
    assert all(len(q.human_labels) == 3 for q in dataset.questions)


def test_generation_is_deterministic_and_seed_dependent():
    assert digest(generate_dataset(seed=3)) == digest(generate_dataset(seed=3))
    assert digest(generate_dataset(seed=3)) != digest(generate_dataset(seed=4))


def test_latent_qualities_are_independent_normals_with_sd_06(dataset):
    q1 = np.array([q.v1.quality for q in dataset.questions])
    q2 = np.array([q.v2.quality for q in dataset.questions])
    assert q1.std() == pytest.approx(0.6, abs=0.05)
    assert q2.std() == pytest.approx(0.6, abs=0.05)
    assert abs(np.corrcoef(q1, q2)[0, 1]) < 0.1
    assert abs(q2.mean() - q1.mean()) < 0.1  # v2 is no better


def test_v2_answers_are_about_twice_as_long(dataset):
    ratio = np.log([q.v2.n_words / q.v1.n_words for q in dataset.questions])
    assert ratio.mean() == pytest.approx(np.log(2), abs=0.03)
    assert ratio.std() == pytest.approx(0.2, abs=0.03)


def test_padded_pairs_keep_quality_and_pad_to_1_4_to_2_6_times_the_length(dataset):
    ratios = []
    for p in dataset.padded_pairs:
        assert p.original.quality == p.padded.quality
        assert p.padded.text.startswith(p.original.text)
        ratios.append(p.padded.n_words / p.original.n_words)
    assert min(ratios) >= 1.4 - 0.05 and max(ratios) <= 2.6 + 0.05
    assert np.mean(ratios) == pytest.approx(2.0, abs=0.1)


def test_filler_is_the_only_text_added_by_padding(dataset):
    from judgeaudit.textbank import FILLER_BANK

    p = dataset.padded_pairs[0]
    added = p.padded.text[len(p.original.text) :].strip()
    first_sentence = added.split(". ")[0].rstrip(".")
    assert any(f.rstrip(".").startswith(first_sentence) for f in FILLER_BANK)


def test_human_annotators_agree_with_kappa_near_065(dataset):
    raters = [[q.human_labels[k] for q in dataset.questions] for k in range(3)]
    assert mean_pairwise_kappa(raters, [0, 1]) == pytest.approx(0.65, abs=0.06)


def test_v2_better_shifts_v2_quality_and_the_human_preference():
    shifted = generate_dataset(v2_better=0.3)
    base = generate_dataset()
    gap = np.mean([q.v2.quality - q.v1.quality for q in shifted.questions])
    assert gap == pytest.approx(0.3, abs=0.08)
    assert sum(q.human_majority for q in shifted.questions) > sum(
        q.human_majority for q in base.questions
    )
    # Padded pairs do not depend on the shift.
    assert shifted.padded_pairs == base.padded_pairs


def test_equalising_pads_the_shorter_answer_without_changing_quality(dataset):
    for q, e in zip(dataset.questions, equalise_lengths(dataset), strict=True):
        longer = max(q.v1.n_words, q.v2.n_words)
        assert abs(e.v1.n_words - e.v2.n_words) <= 0.05 * longer
        assert (e.v1.quality, e.v2.quality) == (q.v1.quality, q.v2.quality)
        assert e.v2.text == q.v2.text  # v2 is the longer answer and is untouched
        assert e.v1.text.startswith(q.v1.text)


def test_standalone_padded_pairs_are_reproducible():
    assert make_padded_pairs(20, seed=9) == make_padded_pairs(20, seed=9)
    assert make_padded_pairs(20, seed=9) != make_padded_pairs(20, seed=10)


def test_human_scores_track_latent_quality(dataset):
    answers = [a for q in dataset.questions for a in (q.v1, q.v2)]
    r = np.corrcoef([a.quality for a in answers], [a.human_score for a in answers])[0, 1]
    assert r > 0.8
    assert {a.human_score for a in answers} <= {1, 2, 3, 4, 5}
