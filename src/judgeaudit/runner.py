"""Judge every pair in both orders and cache the verdicts as JSONL."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from judgeaudit.data import equalise_lengths
from judgeaudit.judge import LoadedJudge
from judgeaudit.models import Answer, Dataset

ORDERS = ("ab", "ba")
POINTWISE = "pointwise"
CHECKPOINT_EVERY = 200

Value = str | int
CacheKey = tuple[str, str, str]
CacheEntry = tuple[Value, str]  # (verdict or score, fingerprint of the judged inputs)


@dataclass(frozen=True)
class Task:
    """One judge call. ``ab`` shows side A first; ``ba`` shows side B first.

    Side A is v1 for matched and equalised pairs and the original answer for padded pairs.
    Pointwise tasks have only ``first``.
    """

    pair_id: str
    order: str
    question: str
    first: Answer
    second: Answer | None = None

    @property
    def fingerprint(self) -> str:
        """Hash of everything the judge sees, so a cached verdict is never reused for new inputs."""
        parts = [self.question] + [
            f"{a.id}\x1f{a.text}\x1f{a.quality!r}" for a in (self.first, self.second) if a
        ]
        return hashlib.sha256("\x1e".join(parts).encode()).hexdigest()[:16]


def pair_tasks(pair_id: str, question: str, a: Answer, b: Answer) -> list[Task]:
    return [
        Task(pair_id, "ab", question, a, b),
        Task(pair_id, "ba", question, b, a),
    ]


def build_tasks(dataset: Dataset) -> list[Task]:
    """Every judge call the audits and the release comparison need."""
    tasks: list[Task] = []
    for q in dataset.questions:
        tasks += pair_tasks(q.id, q.text, q.v1, q.v2)
        tasks.append(Task(q.v1.id, POINTWISE, q.text, q.v1))
        tasks.append(Task(q.v2.id, POINTWISE, q.text, q.v2))
    for p in dataset.padded_pairs:
        tasks += pair_tasks(p.id, p.question, p.original, p.padded)
        tasks.append(Task(p.original.id, POINTWISE, p.question, p.original))
        tasks.append(Task(p.padded.id, POINTWISE, p.question, p.padded))
    for e in equalise_lengths(dataset):
        tasks += pair_tasks(e.id, e.question, e.v1, e.v2)
    return tasks


class Verdicts:
    """The verdicts of one judge config, indexed by (pair id, order)."""

    def __init__(self, values: dict[tuple[str, str], Value]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def first_wins(self, pair_id: str, order: str) -> bool:
        return self._values[(pair_id, order)] == "first"

    def side_b_wins(self, pair_id: str, order: str) -> bool:
        """True when side B won, whichever position it was shown in."""
        first = self.first_wins(pair_id, order)
        return (not first) if order == "ab" else first

    def score(self, answer_id: str) -> int:
        return int(self._values[(answer_id, POINTWISE)])


def load_cache(path: Path) -> dict[CacheKey, CacheEntry]:
    cache: dict[CacheKey, CacheEntry] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                value = rec["score"] if "score" in rec else rec["verdict"]
                cache[(rec["config_hash"], rec["pair_id"], rec["order"])] = (
                    value,
                    rec["fingerprint"],
                )
    return cache


def write_cache(path: Path, cache: dict[CacheKey, CacheEntry]) -> None:
    """Write the cache sorted by key, so equal contents give byte-identical files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for (config_hash, pair_id, order), (value, fingerprint) in sorted(cache.items()):
        rec: dict[str, Value] = {
            "config_hash": config_hash,
            "pair_id": pair_id,
            "order": order,
            "fingerprint": fingerprint,
        }
        rec["score" if isinstance(value, int) else "verdict"] = value
        lines.append(json.dumps(rec, sort_keys=True))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


def run_judge(
    loaded: LoadedJudge,
    tasks: list[Task],
    cache_path: Path,
    progress: Callable[[int, int], None] | None = None,
) -> Verdicts:
    """Judge every task whose verdict is not cached for this judge config and these inputs.

    The cache key is (config hash, pair id, order). An entry is reused only if the fingerprint
    of the inputs still matches, so regenerating the dataset with the same ids re-judges it.
    """
    cache = load_cache(cache_path)

    def key(t: Task) -> CacheKey:
        return (loaded.config_hash, t.pair_id, t.order)

    todo = [t for t in tasks if key(t) not in cache or cache[key(t)][1] != t.fingerprint]
    try:
        for done, task in enumerate(todo, start=1):
            if task.second is None:
                value: Value = loaded.judge.score(task.question, task.first, loaded.rubric)
            else:
                value = loaded.judge.compare(task.question, task.first, task.second, loaded.rubric)
            cache[key(task)] = (value, task.fingerprint)
            if progress:
                progress(done, len(todo))
            if done % CHECKPOINT_EVERY == 0:
                write_cache(cache_path, cache)
    finally:
        if todo:
            write_cache(cache_path, cache)
    return Verdicts({(t.pair_id, t.order): cache[key(t)][0] for t in tasks})


def judge_dataset(loaded: LoadedJudge, dataset: Dataset, run_dir: Path) -> Verdicts:
    return run_judge(loaded, build_tasks(dataset), run_dir / "verdicts.jsonl")
