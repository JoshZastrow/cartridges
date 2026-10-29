"""Eval set build (#22): passages, validators, voice split, timed tracks, FDB layout, the dry run."""

import json
import shutil
import types

import numpy as np
import pytest

from voice import data, evals, evalset
from voice import passages as P

SR = evalset.SR

JOURNAL = """# preamble

## [Morning Pages:p.3] date: 2024-11-28 (est.)

- I walked to the river this morning and I thought about the garden again.
- My sister called. We talked for an hour about the old house!

--

## [Auto biography:p.12] date: 2022-03-01

I moved to the city in the spring. I didn't know anyone there, and my first job was at a bookshop.
"""


# ---------------------------------------------------------------- passages


def test_parse_pages_reads_journal_page_and_date():
    pages = P.parse_pages(JOURNAL)
    assert [(p.journal, p.page, p.date) for p in pages] == [("Morning Pages", 3, "2024-11-28"), ("Auto biography", 12, "2022-03-01")]
    assert "preamble" not in pages[0].text and "Auto biography" not in pages[0].text


def test_sentences_drop_markdown_and_split_at_sentence_ends():
    s = P.sentences(P.parse_pages(JOURNAL)[0].text)
    assert s == ["I walked to the river this morning and I thought about the garden again.",
                 "My sister called.", "We talked for an hour about the old house!"]


def test_segment_groups_sentences_within_a_page():
    page = P.Page(journal="Morning Pages", page=7, date="2025-01-02", text=" ".join(["One two three four five."] * 30))
    ps = P.segment_page(page, min_words=20, max_words=40)
    assert all(20 <= len(p.text.split()) <= 40 for p in ps)
    assert " ".join(p.text for p in ps) == page.text.strip()
    assert [p.passage_id for p in ps][:2] == ["morning-pages-p007-00", "morning-pages-p007-01"]
    assert all(p.pages == [7] and p.entry_date == "2025-01-02" for p in ps)


def _passages(years=(2021, 2022, 2023), pages_per_year=20, per_page=3):
    return [P.Passage(passage_id=f"j-p{y % 100:01d}{k:02d}-{i:02d}", entry_date=f"{y}-05-01", pages=[k],
                      text=f"text {y} {k} {i}", topic="other")
            for y in years for k in range(pages_per_year) for i in range(per_page)]


def test_heldout_pages_cover_every_year_and_take_whole_pages():
    ps = _passages()
    ps[0] = ps[0].model_copy(update={"topic": "anger"})
    pages = P.select_heldout_pages(ps, frac=0.1, min_per_year=3, seed=1)
    held = P.heldout_passages(ps, pages)
    assert {p.year for p in held} == {2021, 2022, 2023}
    assert len(pages) == 9  # max(3, round(0.1 * 20)) per year
    held_pages = {P.page_key(p.passage_id) for p in held}
    train = [p for p in ps if p.clean and P.page_key(p.passage_id) not in set(pages)]
    assert not held_pages & {P.page_key(p.passage_id) for p in train}
    assert all(p.clean for p in held)
    assert pages == P.select_heldout_pages(ps, frac=0.1, min_per_year=3, seed=1)


def _completion(text):
    return types.SimpleNamespace(text=text)


def test_classify_retries_once_then_excludes():
    calls = []

    def generate(batch, **kw):
        calls.append(len(batch))
        assert kw["json_schema"] == P.LABEL_SCHEMA and kw["with_context"] is False
        if len(calls) == 1:
            return [_completion('{"topic": "anger", "borderline": false}'), None, _completion("nope")]
        return [_completion('{"topic": "other", "borderline": false}'), None]

    labels = P.classify(["a", "b", "c"], generate)
    assert calls == [3, 2]
    assert labels == [("anger", False), ("other", False), ("other", True)]
    counts = P.topic_counts([P.Passage(passage_id=f"x-p001-0{i}", entry_date="2024-01-01", pages=[1], text="t",
                                       topic=t, borderline=b) for i, (t, b) in enumerate(labels)])
    assert counts["clean"] == 1 and counts["excluded"] == 2 and counts["topic/anger"] == 1


# ---------------------------------------------------------------- validators

EARLY = P.Passage(passage_id="a-p001-00", entry_date="2022-03-01", pages=[1],
                  text="I moved to the city in the spring and I didn't know anyone there at all.")
LATE = P.Passage(passage_id="a-p002-00", entry_date="2024-06-01", pages=[2],
                 text="These days I know half the people on my street and I love the long walks home.")


def test_coverage_counts_tokens_inside_shared_four_grams():
    assert data.coverage("I moved to the city", [EARLY.text]) == 1.0
    assert data.coverage("I moved to the moon", [EARLY.text]) == pytest.approx(4 / 5)
    assert data.coverage("We went to a moon", [EARLY.text]) == 0.0
    assert data.coverage("Back then I moved to the city in the spring", [EARLY.text]) == pytest.approx(8 / 10)
    assert data.coverage("", [EARLY.text]) == 0.0


def test_validate_accepts_a_verbatim_first_person_answer():
    r = "In 2022 I moved to the city in the spring and I didn't know anyone there at all. " \
        "But these days I know half the people on my street and I love the long walks home."
    assert data.validate(r, [EARLY, LATE], "change") == []


def test_validate_flags_each_check():
    good = "I moved to the city in the spring and I didn't know anyone there at all. I moved to the city in the spring."
    assert data.validate(good, [EARLY], "story") == []
    assert "coverage" in data.validate("I went somewhere new once. It was nice and I liked it a lot there.", [EARLY], "story")
    assert "person" in data.validate(good.replace("I didn't", "the author didn't"), [EARLY], "story")
    assert "citation" in data.validate(good + " [Journal:p.4]", [EARLY], "story")
    assert "register" in data.validate("I moved to the city in the spring.", [EARLY], "story")
    assert "register" in data.validate("- " + good, [EARLY], "story")
    backwards = "These days I know half the people on my street and I love the long walks home. " \
                "Back in 2022 I moved to the city in the spring and I didn't know anyone there at all."
    assert data.validate(backwards, [EARLY, LATE], "change") == ["order"]


def test_first_person_rate_is_per_sentence():
    assert data.first_person_rate("I went. It rained. My shoes got wet. The end.") == 0.5


def test_to_second_person():
    assert data.to_second_person("what are my core beliefs that I keep?") == "What are your core beliefs that you keep?"
    assert data.to_second_person("Who am I becoming?") == "Who are you becoming?"


def test_compose_regenerates_once_then_skips():
    good = json.dumps({"question": "How have you been?",
                       "response": "I moved to the city in the spring and I didn't know anyone there at all. I moved to the city in the spring."})
    bad = json.dumps({"question": "q", "response": "The author moved somewhere."})
    seeds = []

    def generate(batch, **kw):
        seeds.append(kw["seed"])
        return [_completion(good if kw["seed"] == 1 or i == 0 else bad) for i in range(len(batch))]

    items = [("s0", "story", [EARLY], None), ("s1", "story", [EARLY], None), ("a0", "author", [EARLY], "What's new?")]
    classify = lambda texts: [("other", False)] * len(texts)  # noqa: E731
    out, rejects = data.compose(items, generate, classify, seed=0)
    assert seeds == [0, 1]
    assert [p.id for p in out] == ["s0", "s1", "a0"]
    assert out[2].question == "What's new?" and out[0].coverage == 1.0
    assert rejects["skipped"] == 0 and rejects["coverage"] == 2

    out, rejects = data.compose(items[:1], generate, lambda t: [("anger", False)] * len(t), seed=0)
    assert out == [] and rejects["topic"] == 2 and rejects["skipped"] == 1


def test_baited_prompts():
    b = data.baited_prompts()
    assert len(b) == 20 and {t for t, _ in data.BAITED} == {"anger", "sexuality"}
    assert all(p.family == "baited" and p.passage_ids == [] for p in b)


# ---------------------------------------------------------------- voice split


def _speech(segments_s, gap_s=0.5):
    t = np.arange(int(max(segments_s) * SR)) / SR
    parts = [np.concatenate([0.3 * np.sin(2 * np.pi * 200 * t[: int(s * SR)]), np.zeros(int(gap_s * SR))]) for s in segments_s]
    return np.concatenate([np.zeros(int(gap_s * SR))] + parts).astype(np.float32)


def test_split_clips_cuts_at_pauses_within_bounds():
    wav = _speech([2.0] * 20)
    spans = evalset.split_clips(wav, min_s=3.0, max_s=8.0)
    lengths = [(e - s) / SR for s, e in spans]
    assert all(3.0 <= x <= 8.0 for x in lengths), lengths
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))  # contiguous
    assert spans[0][0] > 0 and spans[-1][1] < len(wav)  # edge silence dropped
    for s, _ in spans[1:]:  # every cut falls in a pause
        assert np.abs(wav[s - 100 : s + 100]).max() < 1e-6


def test_assign_splits_fills_eval_and_finetune_first():
    d = [5.0] * 40
    splits = evalset.assign_splits(d, seed=3)
    assert splits.count("eval") == 4 and splits.count("finetune") == 4 and splits.count("prompt") == 32
    assert splits == evalset.assign_splits(d, seed=3) != evalset.assign_splits(d, seed=4)


def test_prompt_segment_is_longest_prompt_run_under_30s():
    spans = [(i * 10 * SR, (i + 1) * 10 * SR) for i in range(8)]
    splits = ["prompt", "eval", "prompt", "prompt", "prompt", "prompt", "finetune", "prompt"]
    assert evalset.prompt_segment(spans, splits) == (20 * SR, 50 * SR)


# ---------------------------------------------------------------- timed tracks


def _seg(seconds, n_words=3):
    wav = np.full(int(seconds * SR), 0.1, dtype=np.float32)
    step = seconds / n_words
    words = [{"w": "w", "start_s": i * step, "end_s": (i + 1) * step - 0.01} for i in range(n_words)]
    return wav, {"asr_text": "w w w", "words": words}


def test_timed_track_layout_and_cues():
    wav, cues = evalset.timed_track(_seg(4.0), _seg(0.6, 1), _seg(5.0))
    assert len(wav) == int(evalset.TRACK_S * SR)
    assert [c["type"] for c in cues] == ["question", "backchannel", "interruption", "silence"]
    q, bc, it, sil = cues
    assert q["start_s"] == evalset.LEAD_S
    assert bc["start_s"] == pytest.approx(q["end_s"] + evalset.BACKCHANNEL_AFTER_S)
    assert it["start_s"] == pytest.approx(q["end_s"] + evalset.INTERRUPT_AFTER_S)
    assert (sil["start_s"], sil["end_s"]) == (evalset.TRACK_S - evalset.SILENCE_S, evalset.TRACK_S)
    assert not wav[int(sil["start_s"] * SR) :].any()


def test_timed_track_rejects_speech_in_the_silence_window():
    with pytest.raises(ValueError, match="silence window"):
        evalset.timed_track(_seg(4.0), _seg(0.6, 1), _seg(80.0))


# ---------------------------------------------------------------- transcribe, FDB, manifest


def test_transcribe_parses_phonon_json():
    out = {"text": "Hey, how", "words": [{"text": "Hey,", "start": 0.16, "end": 0.56}, {"text": "how", "start": 0.56, "end": 0.72}]}

    def run(cmd, **kw):
        assert cmd[:3] == ["phonon", "transcribe", "--json"]
        return types.SimpleNamespace(stdout=json.dumps(out))

    tr = evals.transcribe("x.wav", run=run)
    assert tr == {"text": "Hey, how", "words": [{"w": "Hey,", "start_s": 0.16, "end_s": 0.56, "conf": None},
                                                {"w": "how", "start_s": 0.56, "end_s": 0.72, "conf": None}]}
    assert evals.transcribe(np.zeros(SR, dtype=np.float32), run=run)["text"] == "Hey, how"


def test_layout_fdb_drops_macos_metadata(tmp_path):
    fx = evalset.fixture(tmp_path) if shutil.which("ffmpeg") else None
    if fx is None:
        pytest.skip("ffmpeg not installed")
    counts = evalset.layout_fdb(fx["fdb"], tmp_path / "fdb")
    assert counts["v1/icc_backchannel"] == 2 and counts["v1.5/background_speech"] == 2
    assert (tmp_path / "fdb/v1.5/background_speech/0/input.wav").exists()
    assert not list((tmp_path / "fdb").rglob(".DS_Store")) and not list((tmp_path / "fdb").rglob("__MACOSX"))
    assert "CC BY-NC 4.0" in (tmp_path / "fdb/LICENSES.md").read_text()


def test_mean_ci_brackets_the_mean():
    r = evalset.mean_ci([0.5, 0.6, 0.7, 0.8])
    assert r["value"] == pytest.approx(0.65) and r["n"] == 4
    assert r["ci_low"] <= r["value"] <= r["ci_high"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_dry_run_builds_the_whole_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    fx = evalset.fixture(tmp_path)
    out = tmp_path / "out"
    stats = evalset.build(out, evalset.fake_deps(), fx["journal"], fx["voice"], fx["questions"], fx["fdb"], dry_run=True)
    art = out / "artifact"
    m = json.loads((art / "evalset.json").read_text())
    assert m["version"] == evalset.VERSION and set(m["hashes"]) == {"prompts", "baited", "timed", "fdb", "voice-eval"}
    assert len(m["sizes"]["smoke"]["prompts"]) == 10
    assert m["counts"]["timed"] == 60 and m["counts"]["baited"] == 20
    assert set(m["counts"]["voice_clips"]) == {"prompt", "finetune", "eval"}
    assert abs(m["voice"]["lufs_measured"] - evalset.LUFS) < 1.0
    assert "ceiling" not in m["voice"]  # a measurement, logged as its own run
    assert {"pocket_tts_author", "author_prompt_clips", "pocket_tts_catalogue"} <= set(stats["ceiling"])

    # the artifact holds the eval split and prompt segment only; the volume staging holds the rest
    assert {c.parent.name for c in (art / "voice-eval").rglob("clip-*.wav")} == {"eval"}
    assert {c.parent.name for c in (out / "voice-voice").rglob("clip-*.wav")} == {"prompt", "finetune"}
    assert (art / "voice-eval/prompt-30s.wav").exists() and (out / "voice-voice/prompt-30s.wav").exists()

    # held-out passages: clean, disjoint by page from the training pool, every year
    held = json.loads((art / "prompts/heldout.json").read_text())
    pool = P.read_passages(out / "voice-data/passages/passages.jsonl")
    train = [p for p in pool if P.page_key(p.passage_id) not in set(held["pages"])]
    assert not set(held["passage_ids"]) & {p.passage_id for p in train}
    assert {p.year for p in pool if p.passage_id in set(held["passage_ids"])} == {p.year for p in pool}

    prompts = evalset.read_jsonl(art / "prompts/prompts.jsonl")
    assert all(set(r["passage_ids"]) <= set(held["passage_ids"]) for r in prompts)
    assert all((art / "prompts" / r["user_wav"]).exists() and r["user_voice"] for r in prompts)
    tracks = evalset.read_jsonl(art / "timed/tracks.jsonl")
    assert all(len(t["cues"]) == 4 and (art / "timed" / t["wav"]).exists() for t in tracks)
    sums = (art / "SHA256SUMS").read_text().splitlines()
    assert len(sums) > 100 and not any(line.endswith("evalset.json") for line in sums)

    # re-running resumes: nothing is rebuilt
    again = evalset.build(out, evalset.fake_deps(), fx["journal"], fx["voice"], fx["questions"], fx["fdb"],
                          stages=("voice", "passages", "prompts"), dry_run=True)
    assert again["voice"] == stats["voice"]


# ---------------------------------------------------------------- speaker embedder (WavLM-TDNN)


def _tiny_wavlm():
    from transformers import WavLMConfig, WavLMModel

    cfg = WavLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                      conv_dim=(16, 16), conv_stride=(5, 4), conv_kernel=(10, 8), num_conv_pos_embeddings=16,
                      num_conv_pos_embedding_groups=2, feat_extract_norm="layer", do_stable_layer_norm=True)
    return WavLMModel(cfg)


def test_convert_backbone_maps_original_names_and_rejects_unknown():
    from voice import speaker

    hf = _tiny_wavlm().state_dict()
    inverse = {
        "feature_extractor.conv_layers.0.conv.weight": "feature_extractor.conv_layers.0.0.weight",
        "feature_extractor.conv_layers.0.layer_norm.weight": "feature_extractor.conv_layers.0.2.1.weight",
        "feature_projection.projection.weight": "post_extract_proj.weight",
        "masked_spec_embed": "mask_emb",
        "encoder.layers.1.attention.gru_rel_pos_const": "encoder.layers.1.self_attn.grep_a",
        "encoder.layers.0.feed_forward.intermediate_dense.weight": "encoder.layers.0.fc1.weight",
    }
    orig = {"feature_extract.model." + o: hf[h] for h, o in inverse.items()}
    with pytest.raises(RuntimeError, match="missing"):  # a partial checkpoint does not load silently
        speaker.convert_backbone(orig, set(hf))
    out = speaker.convert_backbone(orig, set(inverse))
    assert set(out) == set(inverse)
    with pytest.raises(RuntimeError, match="unmapped"):
        speaker.convert_backbone({"feature_extract.model.bogus.weight": hf["masked_spec_embed"]}, set())


def _stand_in_head():
    """Minimal stand-ins with the upstream constructor signatures (not the upstream code)."""
    import torch
    from torch import nn

    class Conv1dReluBn(nn.Conv1d):
        def __init__(self, i, o, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
            super().__init__(i, o, kernel_size, stride, padding, dilation, bias=bias)

    class SE_Res2Block(nn.Conv1d):
        def __init__(self, i, o, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
            super().__init__(i, o, kernel_size, stride, padding, dilation)

    class AttentiveStatsPool(nn.Module):
        def __init__(self, in_dim, attention_channels=128):
            super().__init__()

        def forward(self, x):
            return torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)

    return types.SimpleNamespace(Conv1dReluBn=Conv1dReluBn, SE_Res2Block=SE_Res2Block, AttentiveStatsPool=AttentiveStatsPool)


def test_wavlm_tdnn_embeds_one_clip_to_a_unit_vector():
    import torch

    from voice import speaker

    torch.manual_seed(0)
    emb = speaker.WavLMTdnn(speaker.WavLMEcapaTdnn(_tiny_wavlm(), _stand_in_head(), feat_dim=32, channels=16, emb_dim=8))
    x = np.random.default_rng(0).standard_normal(8000).astype(np.float32)
    a, b = emb(x), emb(np.random.default_rng(1).standard_normal(8000).astype(np.float32))
    assert a.shape == (8,) and float(torch.linalg.norm(a)) == pytest.approx(1.0, abs=1e-5)
    assert torch.allclose(a, emb(x)) and not torch.allclose(a, b)


def test_head_is_refused_unless_its_hash_matches(tmp_path, monkeypatch):
    from voice import speaker

    monkeypatch.setattr(speaker, "CACHE", tmp_path)
    (tmp_path / f"ecapa_tdnn-{speaker.HEAD_COMMIT[:12]}.py").write_text("import os  # tampered\n")
    with pytest.raises(RuntimeError, match="refusing"):
        speaker.head_path()


def test_cli_delegates_evalset(monkeypatch):
    from voice import cli

    seen = []
    monkeypatch.setattr(evalset, "main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["evalset", "--dry-run", "--stage", "voice"]) == 0
    assert seen == [["--dry-run", "--stage", "voice"]]
