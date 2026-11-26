from __future__ import annotations

from pathlib import Path

import pytest

from judgeaudit.data import generate_dataset
from judgeaudit.judge import load_judge
from judgeaudit.report import run_audit
from judgeaudit.runner import judge_dataset

JUDGES = Path(__file__).resolve().parent.parent / "judges"


@pytest.fixture(scope="session")
def judges_dir():
    return JUDGES


@pytest.fixture(scope="session")
def dataset():
    return generate_dataset()


@pytest.fixture(scope="session")
def verdicts(dataset, tmp_path_factory):
    """Verdicts of the biased and the clean judge on the default dataset."""
    out = {}
    for name in ("biased", "clean"):
        loaded = load_judge(JUDGES / f"{name}.yaml")
        out[name] = (loaded, judge_dataset(loaded, dataset, tmp_path_factory.mktemp(name)))
    return out


@pytest.fixture(scope="session")
def audits(dataset, verdicts):
    return {
        name: run_audit(dataset, v, loaded.name, loaded.config_hash)
        for name, (loaded, v) in verdicts.items()
    }
