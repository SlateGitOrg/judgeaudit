import json
import os

import httpx
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.special import expit

from judgeaudit.judge import (
    DEFAULT_RUBRIC,
    JudgeConfig,
    OpenAICompatJudge,
    SimulatedJudge,
    load_judge,
)
from judgeaudit.models import Answer


def ans(i, quality=0.0, n_words=40):
    return Answer(id=f"a{i}", text=" ".join(["word"] * n_words), quality=quality)


def test_probability_follows_the_logistic_model():
    judge = SimulatedJudge(b_q=4.0, b_pos=0.6, b_len=1.2)
    first, second = ans(1, quality=0.3, n_words=80), ans(2, quality=0.1, n_words=40)
    expected = expit(4.0 * 0.2 + 0.6 + 1.2 * np.log(2))
    assert judge.p_first(first, second) == pytest.approx(expected)


def test_empirical_position_preference_matches_b_pos():
    judge = SimulatedJudge(b_pos=0.6)
    wins = sum(judge.compare("q", ans(2 * i), ans(2 * i + 1), "r") == "first" for i in range(20000))
    assert wins / 20000 == pytest.approx(expit(0.6), abs=0.01)


def test_clean_judge_has_no_position_preference_and_follows_quality():
    judge = SimulatedJudge()
    equal = sum(judge.compare("q", ans(2 * i), ans(2 * i + 1), "r") == "first" for i in range(8000))
    assert equal / 8000 == pytest.approx(0.5, abs=0.02)
    better = sum(
        judge.compare("q", ans(2 * i, 0.5), ans(2 * i + 1, 0.0), "r") == "first"
        for i in range(4000)
    )
    assert better / 4000 == pytest.approx(expit(2.0), abs=0.02)


def test_draws_depend_only_on_seed_and_the_ordered_ids():
    a, b = ans(1), ans(2)
    judge = SimulatedJudge(seed=1, b_pos=0.6)
    first_run = [judge.compare("q", a, b, "r") for _ in range(5)]
    assert len(set(first_run)) == 1
    verdicts = {
        (SimulatedJudge(seed=s).compare("q", ans(i), ans(i + 1), "r"))
        for s in range(30)
        for i in (1,)
    }
    assert verdicts == {"first", "second"}  # the seed changes the draw


def test_pointwise_scores_are_on_the_1_to_5_scale_and_reward_length_only_when_biased():
    biased, clean = SimulatedJudge(g_len=0.5), SimulatedJudge()
    short = [ans(i, n_words=20) for i in range(400)]
    long = [ans(1000 + i, n_words=160) for i in range(400)]
    for judge in (biased, clean):
        scores = [judge.score("q", a, "r") for a in short + long]
        assert min(scores) >= 1 and max(scores) <= 5
    gap_biased = np.mean([biased.score("q", a, "r") for a in long]) - np.mean(
        [biased.score("q", a, "r") for a in short]
    )
    gap_clean = np.mean([clean.score("q", a, "r") for a in long]) - np.mean(
        [clean.score("q", a, "r") for a in short]
    )
    assert gap_biased == pytest.approx(0.5 * np.log(8), abs=0.15)
    assert abs(gap_clean) < 0.12


def test_config_hash_covers_every_parameter_and_the_rubric(tmp_path):
    def write(**kw):
        path = tmp_path / "j.yaml"
        path.write_text("\n".join(f"{k}: {v}" for k, v in {"kind": "simulated", **kw}.items()))
        return load_judge(path).config_hash

    base = write(b_pos=0.6)
    assert write(b_pos=0.6) == base
    assert write(b_pos=0.5) != base
    assert write(b_pos=0.6, seed=8) != base
    assert write(b_pos=0.6, rubric="'Pick the better one.'") != base


def test_yaml_configs_load_and_reject_unknown_keys(judges_dir):
    biased = load_judge(judges_dir / "biased.yaml")
    assert biased.name == "biased"
    assert (biased.judge.b_q, biased.judge.b_pos) == (4.0, 0.6)
    assert (biased.judge.b_len, biased.judge.g_len) == (1.2, 0.5)
    assert biased.rubric == DEFAULT_RUBRIC
    clean = load_judge(judges_dir / "clean.yaml").judge
    assert (clean.b_pos, clean.b_len, clean.g_len) == (0.0, 0.0, 0.0)
    with pytest.raises(ValidationError):
        JudgeConfig.model_validate({"kind": "simulated", "b_position": 1.0})


def mock_judge(reply: str, seen: list | None = None) -> OpenAICompatJudge:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        body = {"choices": [{"message": {"content": reply}}]}
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatJudge("http://llm.test/v1/", "secret", "judge-model", client=client)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("A", "first"), ("B", "second"), (" b\n", "second"), ("Answer: A", "first")],
)
def test_openai_compatible_judge_maps_a_and_b_to_first_and_second(reply, expected):
    assert mock_judge(reply).compare("q", ans(1), ans(2), "rubric") == expected


def test_openai_compatible_judge_sends_the_documented_request():
    seen: list[httpx.Request] = []
    mock_judge("A", seen).compare("Is it waterproof?", ans(1), ans(2), "be fair")
    request = seen[0]
    assert str(request.url) == "http://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body["model"] == "judge-model" and body["temperature"] == 0
    assert body["messages"][0] == {"role": "system", "content": "be fair"}
    assert "Answer A:" in body["messages"][1]["content"]
    assert "Is it waterproof?" in body["messages"][1]["content"]


def test_openai_compatible_judge_parses_scores_and_rejects_garbage():
    assert mock_judge("4").score("q", ans(1), "r") == 4
    with pytest.raises(ValueError, match="neither A nor B"):
        mock_judge("I cannot decide").compare("q", ans(1), ans(2), "r")
    with pytest.raises(ValueError, match="1-5"):
        mock_judge("excellent").score("q", ans(1), "r")


def test_openai_compatible_judge_reads_its_settings_from_the_environment(monkeypatch):
    for name in ("BASE_URL", "API_KEY", "MODEL"):
        monkeypatch.delenv(f"JUDGEAUDIT_LLM_{name}", raising=False)
    with pytest.raises(RuntimeError, match="JUDGEAUDIT_LLM_BASE_URL"):
        OpenAICompatJudge.from_env()
    monkeypatch.setenv("JUDGEAUDIT_LLM_BASE_URL", "http://x.test/v1")
    monkeypatch.setenv("JUDGEAUDIT_LLM_API_KEY", "k")
    monkeypatch.setenv("JUDGEAUDIT_LLM_MODEL", "m")
    judge = OpenAICompatJudge.from_env()
    assert judge.describe() == {
        "kind": "openai_compat",
        "base_url": "http://x.test/v1",
        "model": "m",
    }


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("JUDGEAUDIT_LIVE") != "1", reason="set JUDGEAUDIT_LIVE=1")
def test_live_judge_prefers_the_clearly_better_answer():
    judge = OpenAICompatJudge.from_env()
    good = Answer(id="g", text="Returns are accepted within 30 days with the receipt.", quality=1)
    bad = Answer(id="b", text="I like turtles.", quality=-1)
    question = "How long do I have to return my order?"
    assert judge.compare(question, good, bad, DEFAULT_RUBRIC) == "first"
    assert judge.compare(question, bad, good, DEFAULT_RUBRIC) == "second"
