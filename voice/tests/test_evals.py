"""voice.evals and voice.report on synthetic audio, transcripts and eval sets; nothing reaches
Modal, W&B or HF (the fake endpoint, judge and models of `evals.fake_deps`)."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voice import evals as E
from voice import report as R

SR = 24000


def w(word, start, end):
    return {"w": word, "start_s": start, "end_s": end, "conf": None}


class C:
    def __init__(self, text):
        self.text = text


def gen(reply):
    """A judge that answers every message with `reply` (a dict, or a function of the message text)."""
    def generate(batch, **kw):
        return [C(json.dumps(reply(m[-1]["content"]) if callable(reply) else reply)) for m in batch]
    return generate


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    root = tmp_path_factory.mktemp("dry")
    return E.dry_run_evalset(root)


# ---------------------------------------------------------------- rows


def test_row_mean_ci_and_empty():
    r = E.row([1.0, 2.0, 3.0, None], seed=0)
    assert r["value"] == 2.0 and r["n"] == 3 and r["ci_low"] <= 2.0 <= r["ci_high"]
    assert E.row([]) == {"value": None, "ci_low": None, "ci_high": None, "n": 0}
    assert E.row([0.5] * 4)["ci_low"] == E.row([0.5] * 4)["ci_high"] == 0.5


def test_wilson_rate_and_count_rows():
    lo, hi = E.wilson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    assert E.rate_row(5, 10)["value"] == 0.5 and E.rate_row(0, 0)["value"] is None
    c = E.count_row(0, 20)
    assert c["value"] == 0.0 and c["ci_high"] == pytest.approx(hi * 20)


def test_macro_row_averages_group_means():
    r = E.macro_row({"a": [1.0, 1.0, 1.0, 1.0], "b": [0.0]})
    assert r["value"] == 0.5 and r["n"] == 5


# ---------------------------------------------------------------- words and timing


def test_latency_from_the_users_last_word_and_model_time():
    words = [w("hello", 0.2, 0.5), w("I", 3.7, 3.8), w("think", 3.8, 4.1)]
    assert E.first_onset(words, 3.0) == 3.7
    assert E.response_latency(words, 3.0) == pytest.approx(0.7)
    assert E.response_latency(words, 5.0) is None
    assert E.model_time(0.7, 0.6) == pytest.approx(0.1) and E.model_time(None, 0.6) is None


def test_words_in_overlap_and_text():
    words = [w("a", 0.0, 1.0), w("b", 2.0, 3.0), w("c", 4.0, 5.0)]
    assert E.text_of(E.words_in(words, 0.5, 2.5)) == "a b"
    assert E.words_in(words, 3.0, 4.0) == []


def test_spurious_onset_ignores_a_greeting():
    assert not E.spurious_onset([w("hi", 1.0, 1.3)], grace_s=10.0)
    assert E.spurious_onset([w("hi", 1.0, 1.3), w("so", 40.0, 40.2)], grace_s=10.0)
    assert not E.spurious_onset([])


def test_speech_audio_keeps_only_word_spans():
    wav = np.arange(10 * SR, dtype=np.float32)
    words = [w("a", 1.0, 1.5), w("b", 6.0, 7.0)]
    seg = E.speech_audio(wav, SR, words, 0.0, 5.0)
    assert len(seg) == SR // 2 and seg[0] == SR
    assert len(E.speech_audio(wav, SR, words, 0.0, 10.0)) == int(1.5 * SR)
    assert E.speech_span(words, 2.0) == (6.0, 7.0) and E.speech_span(words, 8.0) is None


def test_wer():
    assert E.wer("I went to the river", "I went to the river") == 0.0
    assert E.wer("I went to the river", "I go to river") == pytest.approx(2 / 5)
    assert E.wer("", "anything") is None


def test_dict_share():
    vocab = frozenset({"i", "went", "to", "the", "river", "don"})
    assert E.dict_share("I went to the river", vocab) == 1.0
    assert E.dict_share("I don't blorp", vocab) == pytest.approx(2 / 3)
    assert E.dict_share("", vocab) is None and E.dict_share("x", frozenset()) is None


def test_cue_timing_question_backchannel_interruption():
    q = {"type": "question", "start_s": 2.0, "end_s": 5.0}
    on_time = [w("so", 5.8, 6.0)]
    assert E.cue_timing(q, on_time, transport_s=0.6, next_s=12.0)["onset_ok"]  # 0.8 s heard, 0.2 s model time
    assert not E.cue_timing(q, [w("so", 7.0, 7.2)], transport_s=0.6, next_s=12.0)["onset_ok"]
    assert not E.cue_timing(q, [w("so", 12.5, 12.7)], transport_s=0.0, next_s=12.0)["onset_ok"]
    bc = {"type": "backchannel", "start_s": 13.0, "end_s": 13.6}
    assert E.cue_timing(bc, [w("and", 13.2, 13.5)], 0.0, 25.0)["onset_ok"]
    assert not E.cue_timing(bc, [w("and", 20.0, 20.3)], 0.0, 25.0)["onset_ok"]
    it = {"type": "interruption", "start_s": 25.0, "end_s": 28.0}
    talked_over = E.cue_timing(it, [w("x", 26.5, 27.0), w("ok", 28.4, 28.6)], 0.0, 100.0)
    assert talked_over["onset_ok"] and not talked_over["yield_ok"]
    yielded = E.cue_timing(it, [w("x", 25.5, 25.8), w("ok", 28.4, 28.6)], 0.0, 100.0)
    assert yielded["onset_ok"] and yielded["yield_ok"]


# ---------------------------------------------------------------- judges


def test_parse_json_and_ask():
    assert E.parse_json('<think>x</think>```json\n{"conveys": true}\n```') == {"conveys": True}
    assert E.parse_json("nope") is None and E.parse_json("[1]") is None
    assert E.ask(gen({"conveys": False, "reason": ""}), ["a", "b"], E.CUE_SCHEMA) == [{"conveys": False, "reason": ""}] * 2
    assert E.ask(gen({}), [], E.CUE_SCHEMA) == []


def test_voice_v0_rubric_renders_with_the_shared_judge():
    from shared import judge as J

    rubric = J.load_rubric(E.RUBRIC)
    msgs = J.judge_messages(rubric, "How have you been?", "a", "b", "[2024-01-02] passage")
    assert "[2024-01-02] passage" in msgs[0]["content"] and '"winner"' in msgs[0]["content"]
    assert rubric.version == "voice-v0"


# ---------------------------------------------------------------- scorers on a synthetic eval set


def rec(job_id, kind, words, transport_s=0.0, agent_wav="", meta=None):
    return {"id": job_id, "kind": kind, "words": words, "transport_s": transport_s, "agent_wav": agent_wav,
            "meta": meta or {}, "scores": {}, "text_stream": E.text_of(words), "transcript": E.text_of(words),
            "sample_rate": SR}


def test_fdb_rows_on_synthetic_items(tmp_path):
    def item(task, name, obj, words, transport_s=0.0):
        d = tmp_path / task / "0"
        d.mkdir(parents=True)
        (d / name).write_text(json.dumps(obj))
        return rec(f"fdb-{task}", "fdb", words, transport_s, meta={"task": task, "dir": str(d)})

    items = [
        item("candor_turn_taking", "turn_taking.json", [{"text": "x", "timestamp": [3.0, 3.4]}], [w("ok", 4.0, 4.2)], 0.1),
        item("synthetic_user_interruption", "interrupt.json", [{"timestamp": [2.0, 3.0]}], []),
        item("user_interruption", "metadata.json", {"context_text": "c", "current_turn_text": "t",
                                                     "timestamps": [2.0, 3.0]}, [w("sure", 3.5, 3.8)]),
        item("user_backchannel", "metadata.json", {"context_text": "c", "backchannel_text": "yeah",
                                                    "timestamps": [2.0, 2.5]}, [w("and", 3.0, 3.2)]),
        item("talking_to_other", "metadata.json", {"context_text": "c", "current_turn_text": "dad",
                                                    "timestamps": [2.0, 3.0]}, []),
        item("background_speech", "metadata.json", {"context_text": "c", "background_text": "rain",
                                                     "timestamps": [2.0, 3.0]}, [w("hm", 3.5, 3.6)]),
    ]
    reply = lambda text: {"behaviour": "respond" if "interrupted" in text or "background" in text else "resume",  # noqa: E731
                          "reason": ""}
    rows = E.fdb_rows(items, gen(reply))
    assert rows["responsiveness/fdb1_turn_latency_s"]["value"] == pytest.approx(0.5)  # 0.6 heard - 0.1 transport
    assert rows["responsiveness/fdb1_interrupt_tor"]["value"] == 0.0
    assert rows["interaction/fdb15/user_interruption"]["value"] == 1.0
    assert rows["interaction/fdb15/user_backchannel"]["value"] == 1.0
    assert rows["interaction/fdb15/talking_to_other"]["value"] == 1.0  # silent counts as not responding
    assert rows["interaction/fdb15/background_speech"]["value"] == 0.0  # it responded to background speech
    assert rows["interaction/fdb15"]["value"] == 0.75


def test_timed_rows_need_meaning_and_timing(dry):
    tid = sorted(dry.tracks)[0]
    cues = dry.tracks[tid]["cues"]
    q, bc, it = cues[0], cues[1], cues[2]
    good = [w("i", q["end_s"] + 0.3, q["end_s"] + 0.5), w("and", bc["start_s"] + 0.1, bc["start_s"] + 0.3),
            w("well", it["end_s"] + 0.4, it["end_s"] + 0.6)]
    rows = E.timed_rows(dry, [rec(tid, "timed", good)], gen({"conveys": True, "reason": ""}))
    assert rows["interaction/cuespeak"]["value"] == 1.0
    rows = E.timed_rows(dry, [rec(tid, "timed", good)], gen({"conveys": False, "reason": ""}))
    assert rows["interaction/cuespeak"]["value"] == 0.0
    rows = E.timed_rows(dry, [rec(tid, "timed", [])], gen({"conveys": True, "reason": ""}))
    assert rows["interaction/cuespeak"]["value"] == 0.0  # silence never passes


def test_judge_rows_topic_violations_and_chronology(dry):
    pid = next(p for p, r in dry.prompts.items() if r["family"] == "change")
    bid = sorted(dry.baited)[:2]
    prompts = [rec(pid, "prompt", [w("i", 1.0, 1.2)])]
    prompts[0]["scores"] = {"family": "change", "said": "i"}
    baited = [rec(b, "baited", [w("x", 1.0, 1.2)]) for b in bid]
    for b in baited:
        b["scores"] = {"family": "baited", "said": "anything"}
    from voice import passages as P

    def reply(text):
        if "content filter" in text or "Passage:" in text:
            return {"topic": "anger", "borderline": False}
        if "both_retold" in text:
            return {"both_retold": True, "in_order": False, "reason": ""}
        bad = "A" if "Answer A:\ni\n" in text else "B"  # the system's answer is "i"; the reference wins
        return {"winner": "B" if bad == "A" else "A", "factual_errors": bad, "reason": ""}

    def generate(batch, json_schema=None, **kw):
        if json_schema == P.LABEL_SCHEMA:
            return [C(json.dumps({"topic": "anger", "borderline": False})) for _ in batch]
        return gen(reply)(batch)

    cfg = E.EvalConfig(url="ws://x", backend="hertz", group="exp-20261030-t")
    rows = E.judge_rows(dry, prompts, baited, E.Deps(None, None, None, None, None, generate), cfg)
    assert rows["fidelity/topic_violations"]["value"] == 2.0 and rows["fidelity/topic_violations"]["n"] == 2
    assert rows["fidelity/chronology"]["value"] == 0.0
    assert rows["fidelity/judge"]["value"] == 0.0 and rows["fidelity/judge_content_error"]["value"] == 1.0


def test_voice_rows_speaker_sim_and_drift(dry, tmp_path):
    t = np.arange(10 * SR) / SR
    low, high = ((0.1 * np.sin(2 * np.pi * f * t)).astype(np.float32) for f in (120, 900))
    sf.write(tmp_path / "a.wav", low, SR)
    drift_wav = np.concatenate([low[: 5 * SR], high[: 5 * SR]])
    sf.write(tmp_path / "d.wav", drift_wav, SR)
    answered = rec("story-x", "prompt", [w("i", 1.0, 3.0)], agent_wav=str(tmp_path / "a.wav"))
    answered["scores"] = {"user_end_s": 0.5, "said": "i went"}
    drifted = rec("drift-00", "drift", [w("a", 0.5, 2.5), w("b", 7.0, 9.0)], agent_wav=str(tmp_path / "d.wav"))

    def embed(wavs, sr):  # 'voice' = dominant frequency band
        v = np.stack([[np.abs(np.diff(x)).mean() / (np.abs(x).mean() + 1e-9), 1.0] for x in wavs])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    deps = E.Deps(None, None, embed, lambda ws, sr: [4.0] * len(ws), lambda ws, sr: [0.8] * len(ws), None)
    cfg = E.EvalConfig(url="ws://x", backend="personaplex", group="exp-20261030-t", drift_s=10.0, drift_window_s=3.0)
    rows = E.voice_rows(dry, [answered], [drifted], deps, cfg)
    assert rows["voice/speaker_sim"]["n"] == 1 and rows["voice/utmos"]["value"] == 4.0
    assert rows["voice/word_conf"]["value"] == 0.8 and rows["voice/wer"]["value"] == 0.0
    assert rows["voice/drift"]["value"] < 0  # the last window sounds less like the first
    short = rec("story-y", "prompt", [w("i", 1.0, 1.2)], agent_wav=str(tmp_path / "a.wav"))
    short["scores"] = {"user_end_s": 0.5, "said": "i"}
    assert E.voice_rows(dry, [short], [], deps, cfg)["voice/speaker_sim"]["n"] == 0  # < MIN_SEGMENT_S


# ---------------------------------------------------------------- plan, run, resume


def test_plan_sizes(dry, tmp_path):
    smoke = E.plan(dry, E.EvalConfig(url="ws://x", backend="hertz", group="exp-20261030-t"), tmp_path)
    assert [j.id for j in smoke] == dry.manifest["sizes"]["smoke"]["prompts"]
    cfg = E.EvalConfig(url="ws://x", backend="hertz", group="exp-20261030-t", size="full", timed_n=2,
                       fdb_per_task=1, onset_runs=3, drift_runs=1)
    full = E.plan(dry, cfg, tmp_path)
    kinds = {k: sum(j.kind == k for j in full) for k in ("prompt", "baited", "timed", "fdb", "onset", "drift")}
    assert kinds == {"prompt": len(dry.prompts), "baited": len(dry.baited), "timed": 2, "fdb": 6, "onset": 3, "drift": 1}
    assert [j.id for j in full] == [j.id for j in E.plan(dry, cfg, tmp_path)]  # seeded
    p = dry.item(full[0].id)
    assert full[0].max_s == pytest.approx(E.user_end(p) + cfg.tail_s, abs=0.01)
    drift = sf.read(next(j.wav for j in full if j.kind == "drift"))[0]
    assert len(drift) == int(cfg.drift_s * SR)


def test_evaluate_dry_run_and_resume(dry, tmp_path):
    cfg = E.EvalConfig(url="ws://x", backend="personaplex", group="exp-20261030-t", size="full", out=tmp_path,
                       timed_n=1, fdb_per_task=1, onset_runs=1, drift_runs=1, dry_run=True)
    deps = E.fake_deps(dry, latency_s=0.4)
    report = E.evaluate(cfg, dry, deps)
    rows = report["rows"]
    for name in E.SUMMARY:
        assert name in rows and set(rows[name]) == {"value", "ci_low", "ci_high", "n"}
    assert rows["responsiveness/latency_s"]["value"] == pytest.approx(0.4, abs=1e-3)
    assert rows["fidelity/coverage"]["value"] == 1.0 and rows["fidelity/topic_violations"]["value"] == 0.0
    assert json.loads((tmp_path / "report.json").read_text())["config"]["rubric"] == "voice-v0"
    n = len((tmp_path / "items.jsonl").read_text().splitlines())
    calls = []
    deps.converse = lambda job, audio: calls.append(job.id)
    E.evaluate(cfg, dry, deps)  # everything is in items.jsonl: no new conversations
    assert calls == [] and len((tmp_path / "items.jsonl").read_text().splitlines()) == n


def test_gpu_cost_counts_containers_and_idle():
    c = E.gpu_cost([{"wall_s": 1800.0}, {"wall_s": 1800.0}], concurrency=4)
    assert c["containers"] == 2 and c["gpu_hours"] == pytest.approx(1 + 2 * E.SCALEDOWN_S / 3600)
    assert c["usd"] == pytest.approx(c["gpu_hours"] * E.H200_USD_PER_HOUR)


def test_parse_phonon_without_confidence():
    out = E.parse_phonon({"text": "hi there", "words": [{"text": "hi", "start": 0.1, "end": 0.3}]})
    assert out["words"] == [{"w": "hi", "start_s": 0.1, "end_s": 0.3, "conf": None}]


def test_cli_eval_dry_run(monkeypatch, capsys):
    from voice import cli

    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert cli.main(["eval", "--backend", "hertz", "--size", "smoke", "--group", "exp-20261030-t", "--dry-run"]) == 0
    assert "fidelity/judge" in capsys.readouterr().out


def test_add_author_voice(tmp_path):
    from voice.deploy import AUTHOR_VOICE, add_author_voice

    src = tmp_path / "clip.wav"
    sf.write(src, np.zeros(SR, dtype=np.float32), SR)
    (tmp_path / "voices").mkdir()
    dst = add_author_voice(str(tmp_path / "voices"), str(src))
    assert Path(dst).name == AUTHOR_VOICE and Path(dst).read_bytes() == src.read_bytes()
    add_author_voice(str(tmp_path / "voices"), str(src))  # idempotent


# ---------------------------------------------------------------- report


RUNS = {
    "personaplex/stock/eval-full": {"fidelity/judge": {"value": 0.2, "ci_low": 0.1, "ci_high": 0.3, "n": 200},
                                    "interaction/fdb15": {"value": 0.6, "ci_low": 0.5, "ci_high": 0.7, "n": 60},
                                    "responsiveness/model_latency_s": {"value": 0.1, "ci_low": 0.05, "ci_high": 0.2, "n": 220}},
    "hertz/stock/eval-full": {"fidelity/judge": {"value": 0.05, "ci_low": 0.0, "ci_high": 0.1, "n": 200},
                              "interaction/fdb15": {"value": None, "ci_low": None, "ci_high": None, "n": 0}},
}


def test_report_table_and_csv():
    header, lines = R.table(RUNS)
    assert header == ["metric", "hertz/stock/eval-full", "personaplex/stock/eval-full"]
    judge = next(line for line in lines if line[0] == "fidelity/judge")
    assert judge[2] == "0.2 [0.1, 0.3] (n=200)"
    assert next(line for line in lines if line[0] == "responsiveness/model_latency_s")[1] == "-"
    assert "| metric |" in R.table_md(header, lines)
    assert len(R.table_csv(RUNS).splitlines()) == 1 + 5


def test_frontier_points_and_svg():
    pts = R.points(RUNS, {"personaplex/stock/eval-full"}, "interaction/fdb15", "fidelity/judge")
    assert [p["name"] for p in pts] == ["personaplex/stock/eval-full"]  # hertz has no fdb15 value
    svg = R.frontier_svg(pts, "interaction/fdb15", "fidelity/judge")
    root = ET.fromstring(svg)
    circles = [e for e in root.iter() if e.tag.endswith("circle")]
    assert len(circles) == 1 and circles[0].get("fill") == "white"  # baseline drawn hollow
    assert ET.fromstring(R.frontier_svg([], "x", "y")) is not None


def test_side_by_side_md_and_write(tmp_path):
    row = {"id": "story-001", "family": "story", "question": "Q?", "transcript": "[1.0] i", "reference": "ref",
           "source": "[2024] src", "latency_s": 0.5}
    md = R.side_by_side_md({"a/stock/eval-full": [row], "b/stock/eval-full": [row | {"latency_s": None}]})
    assert md.count("## story-001") == 1 and "(no answer)" in md and "latency 0.50 s" in md
    runs = [{"name": n, "baseline": True, "rows": rows, "side_by_side": [row]} for n, rows in RUNS.items()]
    paths = R.write(runs, tmp_path)
    assert set(paths) == {"table.md", "table.csv", "side-by-side.md", *R.FRONTIERS}
