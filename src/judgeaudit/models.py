"""Typed records shared by the generator, judges and audits."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """One candidate answer.

    ``quality`` is the latent quality a simulated judge sees; a real judge only reads ``text``.
    ``human_score`` is a simulated annotator-panel score on the 1-5 scale (original answers only).
    """

    id: str
    text: str
    quality: float
    human_score: int | None = None

    @property
    def n_words(self) -> int:
        return len(self.text.split())


class Question(BaseModel):
    """A product question with the v1 and v2 bot answers and three human preferences.

    ``human_labels[k] == 1`` means annotator k preferred v2.
    """

    id: str
    category: str
    text: str
    v1: Answer
    v2: Answer
    human_labels: list[int]

    @property
    def human_majority(self) -> int:
        return int(sum(self.human_labels) * 2 > len(self.human_labels))


class PaddedPair(BaseModel):
    """An answer and a copy of itself padded with neutral filler. Quality is identical."""

    id: str
    question: str
    category: str
    original: Answer
    padded: Answer


class EqualisedPair(BaseModel):
    """A v1/v2 pair where the shorter answer has been padded to the longer one's length."""

    id: str
    question: str
    v1: Answer
    v2: Answer


class Dataset(BaseModel):
    seed: int
    v2_better: float = 0.0
    questions: list[Question] = Field(default_factory=list)
    padded_pairs: list[PaddedPair] = Field(default_factory=list)
