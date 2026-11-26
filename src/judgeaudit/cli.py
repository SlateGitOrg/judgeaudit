"""Command line interface: seed, run, audit, compare."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from judgeaudit.compare import compare_releases
from judgeaudit.data import (
    DEFAULT_PADDED,
    DEFAULT_QUESTIONS,
    DEFAULT_SEED,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from judgeaudit.judge import LoadedJudge, load_judge
from judgeaudit.models import Dataset
from judgeaudit.report import render_audit, render_comparison, run_audit
from judgeaudit.runner import Verdicts, judge_dataset
from judgeaudit.stats import mean_pairwise_kappa

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

DataOpt = Annotated[Path, typer.Option("--data", help="Dataset written by `seed`.")]
RunsOpt = Annotated[Path, typer.Option("--runs", help="Directory for cached verdicts.")]
JudgeOpt = Annotated[Path, typer.Option("--judge", help="Judge config, e.g. judges/biased.yaml.")]


def _prepare(judge_path: Path, data: Path, runs: Path) -> tuple[LoadedJudge, Dataset, Verdicts]:
    if not data.exists():
        typer.echo(f"{data} not found; run `judgeaudit seed` first.", err=True)
        raise typer.Exit(2)
    loaded = load_judge(judge_path)
    dataset = load_dataset(data)
    verdicts = judge_dataset(loaded, dataset, runs / loaded.name)
    return loaded, dataset, verdicts


@app.command()
def seed(
    v2_better: Annotated[float, typer.Option(help="Shift v2's latent quality by this much.")] = 0.0,
    seed: Annotated[int, typer.Option(help="Generator seed.")] = DEFAULT_SEED,
    out: Annotated[Path, typer.Option(help="Where to write the dataset.")] = Path(
        "data/dataset.json"
    ),
) -> None:
    """Generate the synthetic product-question dataset."""
    dataset = generate_dataset(seed, v2_better, DEFAULT_QUESTIONS, DEFAULT_PADDED)
    save_dataset(dataset, out)
    qs = dataset.questions
    raters = [[q.human_labels[k] for q in qs] for k in range(3)]
    ratio = sum(q.v2.n_words for q in qs) / sum(q.v1.n_words for q in qs)
    typer.echo(f"wrote {out}")
    typer.echo(f"{len(qs)} questions in 8 categories, {len(dataset.padded_pairs)} padded pairs")
    typer.echo(f"v2 is {ratio:.2f}x as long as v1; v2 quality shift {v2_better:+.2f}")
    typer.echo(f"mean pairwise human-human kappa {mean_pairwise_kappa(raters, [0, 1]):.3f}")


@app.command()
def run(judge: JudgeOpt, data: DataOpt = Path("data/dataset.json"), runs: RunsOpt = Path("runs")):
    """Judge every pair in both orders and cache the verdicts."""
    loaded, _, verdicts = _prepare(judge, data, runs)
    typer.echo(f"{loaded.name} ({loaded.config_hash}): {len(verdicts)} verdicts in {runs}")


@app.command()
def audit(
    judge: JudgeOpt,
    data: DataOpt = Path("data/dataset.json"),
    runs: RunsOpt = Path("runs"),
    reports: Annotated[Path, typer.Option(help="Directory for the Markdown report.")] = Path(
        "reports"
    ),
) -> None:
    """Audit the judge; exit 1 if any gate fails."""
    loaded, dataset, verdicts = _prepare(judge, data, runs)
    result = run_audit(dataset, verdicts, loaded.name, loaded.config_hash)
    markdown = render_audit(result)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{loaded.name}.md").write_text(markdown)
    typer.echo(markdown)
    if not result.passed:
        raise typer.Exit(1)


@app.command()
def compare(
    judge: JudgeOpt,
    data: DataOpt = Path("data/dataset.json"),
    runs: RunsOpt = Path("runs"),
) -> None:
    """Compare v2 with v1, raw and with lengths equalised."""
    loaded, dataset, verdicts = _prepare(judge, data, runs)
    typer.echo(render_comparison(compare_releases(dataset, verdicts), loaded.name))


if __name__ == "__main__":
    app()
