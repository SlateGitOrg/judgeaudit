from pathlib import Path

import pytest
from typer.testing import CliRunner

from judgeaudit.cli import app

runner = CliRunner()


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["seed"])
    assert result.exit_code == 0, result.output
    return tmp_path


def test_seed_writes_the_dataset_and_reports_its_properties(workdir):
    assert (workdir / "data" / "dataset.json").exists()
    result = runner.invoke(app, ["seed", "--v2-better", "0.3", "--out", "alt.json"])
    assert result.exit_code == 0
    assert "600 questions in 8 categories, 300 padded pairs" in result.output
    assert "quality shift +0.30" in result.output
    assert Path("alt.json").exists()


def test_audit_exits_1_for_the_biased_judge_and_0_for_the_clean_judge(workdir, judges_dir):
    biased = runner.invoke(app, ["audit", "--judge", str(judges_dir / "biased.yaml")])
    assert biased.exit_code == 1
    assert "FAIL (2 of 3 gates)" in biased.output
    clean = runner.invoke(app, ["audit", "--judge", str(judges_dir / "clean.yaml")])
    assert clean.exit_code == 0
    assert "**Result: PASS**" in clean.output
    assert (workdir / "reports" / "biased.md").read_text() == biased.output.rstrip("\n") + "\n"


def test_run_caches_verdicts_and_a_second_run_changes_nothing(workdir, judges_dir):
    args = ["run", "--judge", str(judges_dir / "biased.yaml")]
    assert runner.invoke(app, args).exit_code == 0
    path = workdir / "runs" / "biased" / "verdicts.jsonl"
    before = path.read_bytes()
    assert runner.invoke(app, args).exit_code == 0
    assert path.read_bytes() == before
    assert len(before.splitlines()) == 4800


def test_compare_prints_both_win_rates_and_the_decision(workdir, judges_dir):
    result = runner.invoke(app, ["compare", "--judge", str(judges_dir / "biased.yaml")])
    assert result.exit_code == 0
    assert "| Raw |" in result.output and "| Length-equalised |" in result.output
    assert "**Decision: no significant difference**" in result.output


def test_commands_explain_a_missing_dataset(tmp_path, monkeypatch, judges_dir):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["audit", "--judge", str(judges_dir / "clean.yaml")])
    assert result.exit_code == 2
    assert "judgeaudit seed" in result.output


def test_compare_on_a_dataset_with_a_real_gain_ships(workdir, judges_dir):
    assert runner.invoke(app, ["seed", "--v2-better", "0.3", "--out", "better.json"]).exit_code == 0
    args = ["compare", "--judge", str(judges_dir / "biased.yaml"), "--data"]
    default = runner.invoke(app, [*args, "data/dataset.json"])
    better = runner.invoke(app, [*args, "better.json"])
    assert "**Decision: no significant difference**" in default.output
    assert "**Decision: ship**" in better.output
