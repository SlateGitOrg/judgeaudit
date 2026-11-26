"""Seeded generator for the synthetic product-question data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from judgeaudit.models import Answer, Dataset, EqualisedPair, PaddedPair, Question
from judgeaudit.seeding import stable_rng
from judgeaudit.textbank import CATEGORIES, PRODUCTS, QUESTIONS, build_text_family, pad_to

QUALITY_SD = 0.6
V1_MEDIAN_WORDS = 40
V1_LOG_LENGTH_SD = 0.25
V2_LOG_RATIO_MEAN = float(np.log(2.0))
V2_LOG_RATIO_SD = 0.2
HUMAN_NOISE_SD = 0.35
HUMAN_SCORE_NOISE_SD = 0.4
N_ANNOTATORS = 3
PAD_RATIO_RANGE = (1.4, 2.6)
DEFAULT_SEED = 7
DEFAULT_QUESTIONS = 600
DEFAULT_PADDED = 300


def _human_score(quality: float, rng: np.random.Generator) -> int:
    raw = 3.0 + 1.2 * quality + rng.normal(0.0, HUMAN_SCORE_NOISE_SD)
    return int(np.clip(np.floor(raw + 0.5), 1, 5))


def generate_questions(
    n: int = DEFAULT_QUESTIONS, seed: int = DEFAULT_SEED, v2_better: float = 0.0
) -> list[Question]:
    """``n`` questions spread evenly over the 8 categories, each with v1 and v2 answers.

    Latent qualities are drawn independently (v2 shifted by ``v2_better``), v2 is about twice as
    long, and three simulated annotators label each pair as ``sign(dq + N(0, 0.35))``.
    """
    rng = stable_rng(seed, "questions")
    questions: list[Question] = []
    for i in range(n):
        category = CATEGORIES[i % len(CATEGORIES)]
        product = PRODUCTS[category][int(rng.integers(len(PRODUCTS[category])))]
        template = QUESTIONS[category][int(rng.integers(len(QUESTIONS[category])))]
        q1 = float(rng.normal(0.0, QUALITY_SD))
        q2 = float(rng.normal(v2_better, QUALITY_SD))
        n1 = max(12, round(float(np.exp(rng.normal(np.log(V1_MEDIAN_WORDS), V1_LOG_LENGTH_SD)))))
        n2 = max(n1 + 1, round(n1 * float(np.exp(rng.normal(V2_LOG_RATIO_MEAN, V2_LOG_RATIO_SD)))))
        t1, t2 = build_text_family(category, product, [n1, n2], rng)
        dq = q2 - q1
        labels = [int(dq + rng.normal(0.0, HUMAN_NOISE_SD) > 0) for _ in range(N_ANNOTATORS)]
        qid = f"q{i:04d}"
        questions.append(
            Question(
                id=qid,
                category=category,
                text=template.format(p=product),
                v1=Answer(id=f"{qid}-v1", text=t1, quality=q1, human_score=_human_score(q1, rng)),
                v2=Answer(id=f"{qid}-v2", text=t2, quality=q2, human_score=_human_score(q2, rng)),
                human_labels=labels,
            )
        )
    return questions


def make_padded_pairs(n: int = DEFAULT_PADDED, seed: int = DEFAULT_SEED) -> list[PaddedPair]:
    """``n`` matched pairs: an answer against a copy padded with filler to 1.4-2.6x its length.

    Both members carry the same latent quality, so any preference a judge shows is position or
    length, not quality. Original answers alternate between the v1 and v2 length distributions.
    """
    rng = stable_rng(seed, "padded")
    pairs: list[PaddedPair] = []
    for i in range(n):
        category = CATEGORIES[i % len(CATEGORIES)]
        product = PRODUCTS[category][int(rng.integers(len(PRODUCTS[category])))]
        template = QUESTIONS[category][int(rng.integers(len(QUESTIONS[category])))]
        median = V1_MEDIAN_WORDS * (2.0 if i % 2 else 1.0)
        n_orig = max(12, round(float(np.exp(rng.normal(np.log(median), V1_LOG_LENGTH_SD)))))
        ratio = float(rng.uniform(*PAD_RATIO_RANGE))
        n_pad = round(n_orig * ratio)
        quality = float(rng.normal(0.0, QUALITY_SD))
        (text,) = build_text_family(category, product, [n_orig], rng)
        padded_text = pad_to(text, n_pad, rng)
        pid = f"pad{i:04d}"
        pairs.append(
            PaddedPair(
                id=pid,
                question=template.format(p=product),
                category=category,
                original=Answer(id=f"{pid}-orig", text=text, quality=quality),
                padded=Answer(id=f"{pid}-pad", text=padded_text, quality=quality),
            )
        )
    return pairs


def generate_dataset(
    seed: int = DEFAULT_SEED,
    v2_better: float = 0.0,
    n_questions: int = DEFAULT_QUESTIONS,
    n_padded: int = DEFAULT_PADDED,
) -> Dataset:
    return Dataset(
        seed=seed,
        v2_better=v2_better,
        questions=generate_questions(n_questions, seed, v2_better),
        padded_pairs=make_padded_pairs(n_padded, seed),
    )


def equalise_lengths(dataset: Dataset) -> list[EqualisedPair]:
    """Pad the shorter answer of every v1/v2 pair to the longer one's word count.

    The padded answer keeps its latent quality, so the pair differs only in quality. The result
    is within 0% of the longer length, inside the 5% tolerance the release check requires.
    """
    out: list[EqualisedPair] = []
    for q in dataset.questions:
        v1, v2 = q.v1, q.v2
        target = max(v1.n_words, v2.n_words)
        rng = stable_rng(dataset.seed, "equalise", q.id)
        if v1.n_words < target:
            v1 = v1.model_copy(update={"id": f"{v1.id}-eq", "text": pad_to(v1.text, target, rng)})
        elif v2.n_words < target:
            v2 = v2.model_copy(update={"id": f"{v2.id}-eq", "text": pad_to(v2.text, target, rng)})
        out.append(EqualisedPair(id=f"eq-{q.id}", question=q.text, v1=v1, v2=v2))
    return out


def save_dataset(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset.model_dump(), sort_keys=True) + "\n")


def load_dataset(path: Path) -> Dataset:
    return Dataset.model_validate_json(path.read_text())
