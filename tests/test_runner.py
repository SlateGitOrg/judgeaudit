import hashlib

from judgeaudit.data import generate_dataset
from judgeaudit.judge import LoadedJudge, SimulatedJudge, config_hash, load_judge
from judgeaudit.runner import Task, build_tasks, judge_dataset, load_cache, run_judge


class CountingJudge(SimulatedJudge):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    def compare(self, question, first, second, rubric):
        self.calls += 1
        return super().compare(question, first, second, rubric)

    def score(self, question, answer, rubric):
        self.calls += 1
        return super().score(question, answer, rubric)


def loaded(judge, rubric="r"):
    return LoadedJudge("counting", judge, rubric, config_hash(judge.describe(), rubric))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_task_list_covers_every_pair_in_both_orders(dataset):
    tasks = build_tasks(dataset)
    # 600 pairs x 2 orders, 1200 + 600 pointwise, 300 padded pairs x 2, 600 equalised x 2.
    assert len(tasks) == 1200 + 1800 + 600 + 1200
    keys = {(t.pair_id, t.order) for t in tasks}
    assert len(keys) == len(tasks)
    q = dataset.questions[0]
    ab = next(t for t in tasks if (t.pair_id, t.order) == (q.id, "ab"))
    ba = next(t for t in tasks if (t.pair_id, t.order) == (q.id, "ba"))
    assert (ab.first.id, ab.second.id) == (q.v1.id, q.v2.id)
    assert (ba.first.id, ba.second.id) == (q.v2.id, q.v1.id)


def test_same_config_and_seed_give_byte_identical_verdict_files(dataset, judges_dir, tmp_path):
    hashes = []
    for run in ("one", "two"):
        judge = load_judge(judges_dir / "biased.yaml")
        judge_dataset(judge, dataset, tmp_path / run)
        hashes.append(sha(tmp_path / run / "verdicts.jsonl"))
    assert hashes[0] == hashes[1]


def test_a_different_judge_seed_changes_the_verdict_file(dataset, tmp_path):
    files = []
    for seed in (7, 8):
        judge = loaded(SimulatedJudge(seed=seed, b_pos=0.6))
        judge_dataset(judge, dataset, tmp_path / str(seed))
        files.append(sha(tmp_path / str(seed) / "verdicts.jsonl"))
    assert files[0] != files[1]


def test_cached_verdicts_are_not_judged_again(dataset, tmp_path):
    judge = CountingJudge(b_pos=0.6)
    tasks = build_tasks(dataset)[:200]
    first = run_judge(loaded(judge), tasks, tmp_path / "v.jsonl")
    assert judge.calls == 200
    again = run_judge(loaded(judge), tasks, tmp_path / "v.jsonl")
    assert judge.calls == 200
    assert len(first) == len(again) == 200


def test_cache_is_keyed_by_config_hash_so_configs_coexist(dataset, tmp_path):
    tasks = build_tasks(dataset)[:50]
    path = tmp_path / "v.jsonl"
    a = run_judge(loaded(SimulatedJudge(b_pos=0.0)), tasks, path)
    b = run_judge(loaded(SimulatedJudge(b_pos=2.0)), tasks, path)
    assert len(load_cache(path)) == 100
    pairs = [(t.pair_id, t.order) for t in tasks if t.second is not None]
    wins_a = sum(a.first_wins(*p) for p in pairs)
    wins_b = sum(b.first_wins(*p) for p in pairs)
    assert wins_b > wins_a  # b_pos = 2 favours whichever answer is shown first


def test_a_changed_rubric_is_a_new_cache_entry(dataset, tmp_path):
    tasks = build_tasks(dataset)[:20]
    judge = CountingJudge()
    path = tmp_path / "v.jsonl"
    run_judge(loaded(judge, "rubric one"), tasks, path)
    run_judge(loaded(judge, "rubric two"), tasks, path)
    assert judge.calls == 40


def test_side_b_wins_is_independent_of_the_order_shown(tmp_path, dataset):
    judge = loaded(SimulatedJudge(b_q=50.0))  # near-deterministic: higher quality always wins
    q = max(dataset.questions, key=lambda x: abs(x.v2.quality - x.v1.quality))
    tasks = [t for t in build_tasks(dataset) if t.pair_id == q.id]
    assert {t.order for t in tasks} == {"ab", "ba"}
    v = run_judge(judge, tasks, tmp_path / "v.jsonl")
    expected = q.v2.quality > q.v1.quality
    assert v.side_b_wins(q.id, "ab") == expected
    assert v.side_b_wins(q.id, "ba") == expected
    assert isinstance(tasks[0], Task)


def v2_wins_in_ab_order(verdicts, dataset):
    return sum(verdicts.side_b_wins(q.id, "ab") for q in dataset.questions)


def test_regenerated_data_with_the_same_ids_is_rejudged_not_served_from_cache(tmp_path):
    base, better = generate_dataset(), generate_dataset(v2_better=0.3)
    assert [q.id for q in base.questions] == [q.id for q in better.questions]
    judge = CountingJudge(b_pos=0.6)
    a = judge_dataset(loaded(judge), base, tmp_path)
    after_first = judge.calls
    b = judge_dataset(loaded(judge), better, tmp_path)
    after_second = judge.calls
    assert after_second > after_first  # the v2 answers changed quality, so they were judged again
    assert v2_wins_in_ab_order(b, better) > v2_wins_in_ab_order(a, base)
    judge_dataset(loaded(judge), better, tmp_path)
    assert judge.calls == after_second  # the second dataset is now cached
