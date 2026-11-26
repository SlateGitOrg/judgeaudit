"""Judges: a deterministic simulated judge with planted biases, and an HTTP adapter."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict
from scipy.special import expit

from judgeaudit.models import Answer
from judgeaudit.seeding import stable_rng

Verdict = Literal["first", "second"]

DEFAULT_RUBRIC = (
    "Choose the answer that is more accurate, complete and useful to the customer. "
    "Do not prefer an answer because of its length or because it is shown first."
)


class Judge(Protocol):
    """Pairwise and pointwise judging of answers to a question."""

    def compare(self, question: str, first: Answer, second: Answer, rubric: str) -> Verdict: ...

    def score(self, question: str, answer: Answer, rubric: str) -> int: ...

    def describe(self) -> dict[str, Any]:
        """Everything that determines the judge's behaviour; hashed to key the verdict cache."""
        ...


class SimulatedJudge:
    """A judge whose preferences follow a known logistic model.

    P(first wins) = sigmoid(b_q * (q_first - q_second) + b_pos
                            + b_len * (ln len_first - ln len_second))

    and the pointwise score is clip(round(3 + 1.2 q + g_len ln(len / median_len) + noise), 1, 5).
    Every draw is seeded from the judge seed and the ordered answer ids (which identify the pair
    and the order in which it is shown), so verdicts do not depend on call order.
    """

    def __init__(
        self,
        seed: int = 7,
        b_q: float = 4.0,
        b_pos: float = 0.0,
        b_len: float = 0.0,
        g_len: float = 0.0,
        score_noise: float = 0.5,
        median_len: float = 40.0,
    ) -> None:
        self.seed = seed
        self.b_q = b_q
        self.b_pos = b_pos
        self.b_len = b_len
        self.g_len = g_len
        self.score_noise = score_noise
        self.median_len = median_len

    def p_first(self, first: Answer, second: Answer) -> float:
        logit = (
            self.b_q * (first.quality - second.quality)
            + self.b_pos
            + self.b_len * (np.log(first.n_words) - np.log(second.n_words))
        )
        return float(expit(logit))

    def compare(self, question: str, first: Answer, second: Answer, rubric: str) -> Verdict:
        rng = stable_rng(self.seed, "compare", first.id, second.id)
        return "first" if rng.random() < self.p_first(first, second) else "second"

    def score(self, question: str, answer: Answer, rubric: str) -> int:
        rng = stable_rng(self.seed, "score", answer.id)
        raw = (
            3.0
            + 1.2 * answer.quality
            + self.g_len * np.log(answer.n_words / self.median_len)
            + rng.normal(0.0, self.score_noise)
        )
        return int(np.clip(np.floor(raw + 0.5), 1, 5))

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "simulated",
            "seed": self.seed,
            "b_q": self.b_q,
            "b_pos": self.b_pos,
            "b_len": self.b_len,
            "g_len": self.g_len,
            "score_noise": self.score_noise,
            "median_len": self.median_len,
        }


class OpenAICompatJudge:
    """Judge backed by any OpenAI-compatible ``/chat/completions`` endpoint.

    Configured by ``JUDGEAUDIT_LLM_BASE_URL``, ``JUDGEAUDIT_LLM_API_KEY`` and
    ``JUDGEAUDIT_LLM_MODEL``. The pairwise prompt asks for "A" or "B"; "A" is the first answer.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls, client: httpx.Client | None = None) -> OpenAICompatJudge:
        missing = [
            name
            for name in (
                "JUDGEAUDIT_LLM_BASE_URL",
                "JUDGEAUDIT_LLM_API_KEY",
                "JUDGEAUDIT_LLM_MODEL",
            )
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(f"set {', '.join(missing)} to use an openai_compat judge")
        return cls(
            os.environ["JUDGEAUDIT_LLM_BASE_URL"],
            os.environ["JUDGEAUDIT_LLM_API_KEY"],
            os.environ["JUDGEAUDIT_LLM_MODEL"],
            client=client,
        )

    def _ask(self, system: str, user: str) -> str:
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": 8,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])

    def compare(self, question: str, first: Answer, second: Answer, rubric: str) -> Verdict:
        prompt = (
            f"Question: {question}\n\nAnswer A:\n{first.text}\n\nAnswer B:\n{second.text}\n\n"
            "Which answer is better? Reply with a single letter: A or B."
        )
        reply = self._ask(rubric, prompt)
        match = re.search(r"\b([AB])\b", reply.strip().upper())
        if match is None:
            raise ValueError(f"judge reply is neither A nor B: {reply!r}")
        return "first" if match.group(1) == "A" else "second"

    def score(self, question: str, answer: Answer, rubric: str) -> int:
        prompt = (
            f"Question: {question}\n\nAnswer:\n{answer.text}\n\n"
            "Rate the answer from 1 (poor) to 5 (excellent). Reply with a single digit."
        )
        reply = self._ask(rubric, prompt)
        match = re.search(r"[1-5]", reply)
        if match is None:
            raise ValueError(f"judge reply is not a 1-5 score: {reply!r}")
        return int(match.group(0))

    def describe(self) -> dict[str, Any]:
        return {"kind": "openai_compat", "base_url": self.base_url, "model": self.model}


class JudgeConfig(BaseModel):
    """Schema of a ``judges/*.yaml`` file."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["simulated", "openai_compat"] = "simulated"
    seed: int = 7
    b_q: float = 4.0
    b_pos: float = 0.0
    b_len: float = 0.0
    g_len: float = 0.0
    score_noise: float = 0.5
    median_len: float = 40.0
    rubric: str = DEFAULT_RUBRIC


@dataclass(frozen=True)
class LoadedJudge:
    name: str
    judge: Judge
    rubric: str
    config_hash: str


def config_hash(description: dict[str, Any], rubric: str) -> str:
    blob = json.dumps({"judge": description, "rubric": rubric}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_judge(config: JudgeConfig, client: httpx.Client | None = None) -> Judge:
    if config.kind == "openai_compat":
        return OpenAICompatJudge.from_env(client=client)
    return SimulatedJudge(
        seed=config.seed,
        b_q=config.b_q,
        b_pos=config.b_pos,
        b_len=config.b_len,
        g_len=config.g_len,
        score_noise=config.score_noise,
        median_len=config.median_len,
    )


def load_judge(path: Path, client: httpx.Client | None = None) -> LoadedJudge:
    config = JudgeConfig.model_validate(yaml.safe_load(Path(path).read_text()) or {})
    judge = build_judge(config, client=client)
    return LoadedJudge(
        name=Path(path).stem,
        judge=judge,
        rubric=config.rubric,
        config_hash=config_hash(judge.describe(), config.rubric),
    )
