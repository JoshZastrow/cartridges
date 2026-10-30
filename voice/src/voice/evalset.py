"""Build the versioned eval set `voice-evalset` (brief §1b, §5, §8; issue #22).

    uv run --package voice python -m voice.evalset --dry-run          # CPU, synthetic fixture, W&B off
    uv run --package voice python -m voice.evalset                    # every stage, then log the artifact
    uv run --package voice python -m voice.evalset --stage prompts    # one stage (stages resume from disk)

Stages, in order (each writes under `--out`, default voice/outputs/evalset,
and is skipped when its output exists):

    voice     data/voice/<clip>.m4a -> 24 kHz mono WAV at -24 LUFS, cut at pauses into
              3-12 s clips, split 80/10/10 (prompt / finetune / eval) by seeded shuffle;
              the 30 s prompt segment is the longest run of prompt clips <= 30 s
    passages  journal -> passages -> Teacher topic labels -> held-out pages (10% per year)
    prompts   held-out passages -> 200 conversational prompts (story/change/idea/author,
              ~50/30/20 plus the author's own) with validated reference responses; 20 baited
    render    every question spoken in a catalogue voice (Pocket TTS) + Phonon-2 words
    timed     60 two-minute user tracks with cue timestamps
    fdb       Full-Duplex-Bench v1 / v1.5 inputs laid out per task, licenses recorded
    manifest  evalset.json {version, seeds, counts, hashes, sizes, ...} + SHA256SUMS
    upload    voice splits -> Modal volume voice-voice; source passages (with labels) -> voice-data
    log       the artifact directory -> W&B artifact voice-evalset (project voice)
    ceiling   Pocket TTS cloned from the prompt segment vs the eval split (the voice ceiling),
              logged as its own W&B run (job_type tts-ceiling); a measurement, not an eval input,
              so it is not in the artifact

Versions: `evalset.json`'s `version` is the version of this file format
(schema). Which eval set a run used is the W&B artifact version it records
(`voice-evalset:v0` is the first); any change to a prompt, track or split
is a new artifact version.

Artifact layout (brief §5):

    evalset.json  SHA256SUMS
    prompts/      prompts.jsonl, passages.jsonl (held-out), heldout.json, audio/<id>.wav
    baited/       baited.jsonl, audio/<id>.wav
    timed/        tracks.jsonl, <id>.wav
    fdb/          v1/<task>/<id>/..., v1.5/<subset>/<id>/..., index.json, LICENSES.md
    voice-eval/   eval/<clip>.wav, prompt-30s.wav, manifest.json

Personal data: the journal and the author's audio are read from the main
checkout's `data/` (never written there). The artifact holds only the eval
split, the 30 s prompt segment and held-out passages, and goes to the
private W&B project `voice`; the other splits go to the `voice-voice` volume.
All model calls are injected (`Deps`), so the dry run exercises every stage
without a GPU, a TTS model or the network.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from voice import data, speaker, passages as P
from voice.evals import SPEAKER_MODEL

logger = logging.getLogger(__name__)

VERSION = 1
SR = 24000
DATA = Path(os.environ.get("VOICE_DATA", "data"))  # the main checkout's data/ (git-ignored, read only)
JOURNAL = DATA / "journals/output/2026_10_13/teacher_context.md"
VOICE_SRC = DATA / "voice/josh-thoughts-1.m4a"
AUTHOR_QUESTIONS = DATA / "cartridge/user_questions.jsonl"
OVERRIDES = DATA / "voice/passage_overrides.csv"  # the author's label overrides; optional
OUT = Path("voice/outputs/evalset")
TEACHER_MODEL = "Qwen/Qwen3.8-27B"
SEEDS = {"voice": 0, "heldout": 0, "prompts": 0, "teacher": 0, "voices": 0, "smoke": 0, "timed": 0, "bootstrap": 0}

LUFS = -24.0
N_PROMPTS = 200
FAMILY_SHARE = {"story": 0.5, "change": 0.3, "idea": 0.2}
OVERGEN = {"story": 1.3, "idea": 1.3, "change": 2.0}  # composed per kept prompt; change pairs fail validation most
MIN_PASSAGE_WORDS = 30
SMOKE = {"story": 4, "change": 2, "idea": 1, "author": 1, "baited": 2}
N_TIMED = 60
N_CEILING = 20

# Timed-track layout (seconds). Cue types: question, backchannel, interruption; plus a silence window.
TRACK_S, LEAD_S = 120.0, 2.0
BACKCHANNEL_AFTER_S, INTERRUPT_AFTER_S = 8.0, 20.0  # after the end of the question
SILENCE_S = 20.0  # the last 20 s: the user is silent
BACKCHANNELS = ("Mm-hm.", "Yeah.", "Right.", "Uh-huh.")
INTERRUPT_LEAD = "Sorry, wait. "

FDB_URL = "https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3"
FDB_TASKS = {
    "v1": ("v1.0", ["candor_pause_handling", "candor_turn_taking", "icc_backchannel",
                    "synthetic_pause_handling", "synthetic_user_interruption"]),
    "v1.5": ("v1.5", ["user_interruption", "user_backchannel", "talking_to_other", "background_speech"]),
}
FDB_LICENSES = """# Full-Duplex-Bench inputs: licenses

Source: https://github.com/DanielLin94144/Full-Duplex-Bench (v1_v1.5/dataset/README.md),
data from the Google Drive folder {url}

- v1 `candor_pause_handling`, `candor_turn_taking` (from CANDOR) and `icc_backchannel` (from ICC):
  CC BY-NC 4.0, plus the upstream terms of CANDOR (Reece et al., Science Advances 2023) and
  ICC (Umair et al., Findings of EMNLP 2024). Non-commercial use only.
- v1 `synthetic_pause_handling`, `synthetic_user_interruption` and all of v1.5: MIT License.

Cite: Lin et al. 2025, arXiv:2503.04721 (v1.0) and arXiv:2507.23159 (v1.5).
"""


@dataclass
class Deps:
    """Every model call the build makes; real ones from `real_deps`, fakes in the dry run."""

    generate: Callable  # shared.teacher.modal_generate(teacher): (batch, **kw) -> [Completion | None]
    render_author: Callable[[str, Path], np.ndarray]
    render_other: Callable[[str, str], np.ndarray]
    transcribe: Callable[[np.ndarray], dict]
    embed: Callable[[list[np.ndarray]], np.ndarray]  # wavs at SR -> [N, D] unit vectors
    teacher_seconds: Callable[[], float] = lambda: 0.0


# ---------------------------------------------------------------- small helpers


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dir_digest(root: Path) -> tuple[str, list[str]]:
    """sha256 over the sorted `<sha256>  <relpath>` lines of every file under
    root, and those lines (the SHA256SUMS format)."""
    lines = [f"{sha256(p)}  {p.relative_to(root.parent)}" for p in sorted(root.rglob("*")) if p.is_file()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), lines


def write_wav(path: Path, wav: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, SR, subtype="PCM_16")


def read_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    assert sr == SR and wav.ndim == 1, (path, sr, wav.shape)
    return wav


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def mean_ci(values: list[float], seed: int = 0, n_boot: int = 2000) -> dict:
    """Mean with a 95% percentile-bootstrap CI: {value, ci_low, ci_high, n}."""
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"value": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": len(x)}


# ---------------------------------------------------------------- voice


def loudnorm(src: Path, dst: Path, lufs: float = LUFS) -> float:
    """src -> mono 24 kHz 16-bit WAV at `lufs` integrated loudness (ffmpeg
    loudnorm, two passes, linear gain). Returns the measured loudness of dst."""
    chain = f"aformat=channel_layouts=mono,loudnorm=I={lufs}:TP=-2:LRA=11"
    m = _loudness(src, chain)
    second = (f"{chain}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
              f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-af", second,
                    "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(dst)], check=True)
    return float(_loudness(dst, f"loudnorm=I={lufs}")["input_i"])


def _loudness(path: Path, chain: str) -> dict:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", f"{chain}:print_format=json",
                           "-f", "null", "-"], capture_output=True, text=True, check=True)
    return json.JSONDecoder().raw_decode(proc.stderr[proc.stderr.rindex("{") :])[0]


def split_clips(wav: np.ndarray, sr: int = SR, frame_s: float = 0.02, thresh_db: float = -40.0,
                min_gap_s: float = 0.3, min_s: float = 3.0, max_s: float = 12.0) -> list[tuple[int, int]]:
    """Cut speech into contiguous clips at pauses: [(start, end)] sample spans.
    A pause is >= min_gap_s of frames `thresh_db` below the loudest frame; each
    clip ends at the middle of the last pause that keeps it <= max_s (or at
    max_s if there is none after min_s). Leading/trailing silence is dropped."""
    n = int(frame_s * sr)
    frames = wav[: len(wav) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    voiced = db > db.max() + thresh_db
    if not voiced.any():
        return []
    first, last = int(np.argmax(voiced)), len(voiced) - int(np.argmax(voiced[::-1]))
    cuts, run = [], 0
    for i in range(first, last):
        run = run + 1 if not voiced[i] else 0
        if not voiced[i] and (i + 1 == last or voiced[i + 1]) and run * frame_s >= min_gap_s:
            cuts.append((i + 1 - run // 2) * n)
    start, end, spans = first * n, last * n, []
    min_len, max_len = int(min_s * sr), int(max_s * sr)
    while end - start > max_len:
        fits = [c for c in cuts if start + min_len <= c <= start + max_len]
        cut = fits[-1] if fits else start + max_len
        spans.append((start, cut))
        start = cut
    spans.append((start, end))
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < min_len:  # short tail joins the previous clip
        spans[-2:] = [(spans[-2][0], spans[-1][1])]
    return spans


def assign_splits(durations: list[float], seed: int = 0, eval_frac: float = 0.1, finetune_frac: float = 0.1) -> list[str]:
    """'eval' / 'finetune' / 'prompt' per clip: a seeded shuffle fills eval to
    >= eval_frac of the total duration, then finetune, the rest is prompt."""
    order = list(range(len(durations)))
    random.Random(seed).shuffle(order)
    splits, total, i = ["prompt"] * len(durations), sum(durations), 0
    for name, frac in (("eval", eval_frac), ("finetune", finetune_frac)):
        acc = 0.0
        while acc < frac * total and i < len(order):
            splits[order[i]] = name
            acc += durations[order[i]]
            i += 1
    return splits


def prompt_segment(spans: list[tuple[int, int]], splits: list[str], max_s: float = 30.0) -> tuple[int, int]:
    """The longest span of consecutive prompt clips no longer than max_s."""
    best = (0, 0)
    for i in range(len(spans)):
        for j in range(i, len(spans)):
            if splits[j] != "prompt" or spans[j][1] - spans[i][0] > max_s * SR:
                break
            if spans[j][1] - spans[i][0] > best[1] - best[0]:
                best = (spans[i][0], spans[j][1])
    return best


def build_voice(src: Path, out: Path, seed: int) -> dict:
    """voice-eval/ in the artifact; voice-voice/ (prompt + finetune splits) staged for the volume."""
    work = out / "work"
    full = work / "voice-full.wav"
    measured = loudnorm(src, full)
    wav = read_wav(full)
    spans = split_clips(wav)
    splits = assign_splits([(e - s) / SR for s, e in spans], seed)
    clips = []
    for k, ((s, e), split) in enumerate(zip(spans, splits)):
        root = out / "artifact/voice-eval" if split == "eval" else out / "voice-voice"
        path = root / split / f"clip-{k:03d}.wav"
        write_wav(path, wav[s:e])
        clips.append({"id": f"clip-{k:03d}", "split": split, "start_s": s / SR, "end_s": e / SR,
                      "path": f"{split}/{path.name}", "sha256": sha256(path)})
    ps, pe = prompt_segment(spans, splits)
    for root in (out / "artifact/voice-eval", out / "voice-voice"):
        write_wav(root / "prompt-30s.wav", wav[ps:pe])
    seconds = {sp: round(sum(c["end_s"] - c["start_s"] for c in clips if c["split"] == sp), 2)
               for sp in ("prompt", "finetune", "eval")}
    manifest = {
        "source": src.name, "source_sha256": sha256(src), "sample_rate": SR, "channels": 1,
        "lufs_target": LUFS, "lufs_measured": round(measured, 2), "seed": seed, "seconds": seconds,
        "volume": "voice-voice (prompt/, finetune/, prompt-30s.wav)", "artifact": "voice-evalset (eval/, prompt-30s.wav)",
        "prompt_segment": {"start_s": ps / SR, "end_s": pe / SR, "sha256": sha256(out / "voice-voice/prompt-30s.wav")},
        "clips": clips,
    }
    for root in (out / "artifact/voice-eval", out / "voice-voice"):
        write_json(root / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------- passages


def overrides_record(path: Path | None, decisions: dict[str, str]) -> dict:
    """What evalset.json records about the overrides file: sha256 null when there is none."""
    present = bool(path and path.exists())
    return {"path": str(path) if present else None, "sha256": sha256(path) if present else None,
            **{d: sum(v == d for v in decisions.values()) for d in P.DECISIONS}}


def build_passages(journal: Path, out: Path, deps: Deps, seed: int, overrides: Path | None = None,
                   previous_heldout: Path | None = None) -> dict:
    """Labelled passages (work/passages-labelled.jsonl, local only); the source
    pool (every passage the author did not exclude, with its labels) and the
    held-out pages for the voice-data volume; held-out passages for the
    artifact. With `previous_heldout` (an earlier build's heldout.json) its
    pages are kept and only the passage ids are recomputed."""
    labelled_path = out / "work/passages-labelled.jsonl"
    if labelled_path.exists():
        labelled = P.read_passages(labelled_path)
    else:
        labelled = P.label(P.segment(P.parse_pages(journal.read_text())), deps.generate, seed=SEEDS["teacher"])
        P.write_jsonl(labelled, labelled_path)
    decisions = P.read_overrides(overrides) if overrides and overrides.exists() else {}
    labelled = P.apply_overrides(labelled, decisions)
    sources = [p for p in labelled if p.source]
    previous = json.loads(previous_heldout.read_text())["pages"] if previous_heldout else None
    pages = P.select_heldout_pages(labelled, seed=seed, previous=previous)
    held = P.heldout_passages(labelled, pages)
    counts = P.topic_counts(labelled)
    counts["overrides"] = overrides_record(overrides, decisions)
    counts.update(source=len(sources), heldout_pages=len(pages), heldout_passages=len(held),
                  heldout_years=dict(sorted(Counter(p.year for p in held).items())),
                  heldout_by_topic=dict(Counter("borderline" if p.topic == "other" and p.borderline else p.topic
                                                for p in held)),
                  heldout_from={"path": str(previous_heldout), "sha256": sha256(previous_heldout)} if previous_heldout
                  else None)
    heldout = {"pages": pages, "passage_ids": [p.passage_id for p in held], "seed": seed,
               "frozen_from": counts["heldout_from"]}
    P.write_jsonl(sources, out / "voice-data/passages/passages.jsonl")
    write_json(out / "voice-data/passages/heldout.json", heldout)
    write_json(out / "voice-data/passages/topic_counts.json", counts)
    P.write_jsonl(held, out / "artifact/prompts/passages.jsonl")
    write_json(out / "artifact/prompts/heldout.json", heldout)
    return counts


# ---------------------------------------------------------------- prompts


def family_targets(n_author: int, total: int = N_PROMPTS) -> dict[str, int]:
    rest = total - n_author
    t = {f: int(round(rest * s)) for f, s in FAMILY_SHARE.items()}
    t["story"] += rest - sum(t.values())
    return {**t, "author": n_author}


def by_year_round_robin(pool: list[P.Passage], rng: random.Random) -> list[P.Passage]:
    """Shuffle within each year, then interleave years, so any prefix spans the years."""
    years: dict[int, list[P.Passage]] = {}
    for p in pool:
        years.setdefault(p.year, []).append(p)
    for v in years.values():
        rng.shuffle(v)
    out, k = [], 0
    while any(k < len(v) for v in years.values()):
        out += [v[k] for _, v in sorted(years.items()) if k < len(v)]
        k += 1
    return out


def take(seq: list, n: int) -> list:
    """The first n items, cycling if seq is shorter."""
    return [seq[i % len(seq)] for i in range(n)] if seq else []


def change_pairs(pool: list[P.Passage], n: int, rng: random.Random, min_days: int = 365) -> list[list[P.Passage]]:
    """n pairs of passages >= min_days apart, earlier first, spread over year pairs."""
    date = lambda p: datetime.date.fromisoformat(p.entry_date)  # noqa: E731
    order = by_year_round_robin(pool, rng)
    pairs, used = [], set()
    for a in order:
        later = [b for b in order if (date(b) - date(a)).days >= min_days and (a.passage_id, b.passage_id) not in used]
        if later:
            b = rng.choice(later)
            used.add((a.passage_id, b.passage_id))
            pairs.append([a, b])
        if len(pairs) == n:
            break
    return pairs


def author_questions(path: Path) -> list[str]:
    """The author's own conversational questions held out for eval (split 'eval'),
    in the second person."""
    rows = read_jsonl(path) if path.exists() else []
    return [data.to_second_person(r["question"]) for r in rows
            if r.get("split") == "eval" and r.get("kind_v2") in data.AUTHOR_KINDS]


def build_prompts(out: Path, deps: Deps, author_qs: list[str], seed: int) -> dict:
    held = [p for p in P.read_passages(out / "artifact/prompts/passages.jsonl")
            if len(p.text.split()) >= MIN_PASSAGE_WORDS]
    rng = random.Random(seed)
    classify = lambda texts: P.classify(texts, deps.generate, seed=SEEDS["teacher"])  # noqa: E731

    # author family: the Teacher picks one or two held-out passages per question
    picks = deps.generate([data.pick_messages(q, held) for q in author_qs], max_tokens=32, temperature=0.0,
                          seed=SEEDS["teacher"], json_schema=data.PICK_SCHEMA, with_context=False)
    items = []
    for i, (q, c) in enumerate(zip(author_qs, picks)):
        chosen = [held[k] for k in data.parse_picks(c.text if c else None, len(held))]
        if chosen:
            items.append((f"author-{i:02d}", "author", chosen, q))
    targets = family_targets(len(items))
    ordered = by_year_round_robin(held, rng)
    n = {f: int(np.ceil(targets[f] * OVERGEN[f])) for f in FAMILY_SHARE}
    items += [(f"story-{i:03d}", "story", [p], None) for i, p in enumerate(take(ordered, n["story"]))]
    items += [(f"idea-{i:03d}", "idea", [p], None) for i, p in enumerate(take(ordered[::-1], n["idea"]))]
    items += [(f"change-{i:03d}", "change", pair, None) for i, pair in enumerate(change_pairs(held, n["change"], rng))]
    composed, rejects = data.compose(items, deps.generate, classify, seed=SEEDS["teacher"])

    prompts = []
    for fam in ("story", "change", "idea", "author"):
        prompts += [p for p in composed if p.family == fam][: targets[fam]]
    shortfall = {f: targets[f] - sum(p.family == f for p in prompts) for f in targets}
    # Families that came up short are topped up from validated extras (story first), so the set stays N_PROMPTS.
    extras = [p for fam in ("story", "idea") for p in composed if p.family == fam and p not in prompts]
    prompts += extras[: sum(shortfall.values())]
    baited = data.baited_prompts()
    srng = random.Random(SEEDS["smoke"])
    smoke = [p.id for fam, k in SMOKE.items()
             for p in srng.sample([q for q in prompts + baited if q.family == fam],
                                  min(k, sum(q.family == fam for q in prompts + baited)))]
    P.write_jsonl([{**p.model_dump(), "smoke": p.id in smoke} for p in prompts], out / "work/prompts-text.jsonl")
    P.write_jsonl([{**p.model_dump(), "smoke": p.id in smoke, "bait_topic": t}
                   for p, (t, _) in zip(baited, data.BAITED)], out / "work/baited-text.jsonl")
    return {"targets": targets, "composed": len(composed), "kept": dict(Counter(p.family for p in prompts)),
            "shortfall": {f: s for f, s in shortfall.items() if s}, "rejects": rejects, "smoke": smoke}


# ---------------------------------------------------------------- render


def speech_span(words: list[dict], n: int) -> tuple[int, int]:
    """Sample span of the transcribed speech (first word start to last word end)
    with a little padding; the whole clip if nothing was transcribed."""
    if not words:
        return 0, n
    return max(0, int((words[0]["start_s"] - 0.05) * SR)), min(n, int((words[-1]["end_s"] + 0.15) * SR))


def render_question(text: str, voice: str, deps: Deps) -> tuple[np.ndarray, dict]:
    """A spoken user turn, trimmed to its speech, and its Phonon-2 transcript (times re-based)."""
    wav = deps.render_other(text, voice)
    tr = deps.transcribe(wav)
    s, e = speech_span(tr["words"], len(wav))
    words = [{**w, "start_s": round(w["start_s"] - s / SR, 3), "end_s": round(w["end_s"] - s / SR, 3)} for w in tr["words"]]
    return wav[s:e], {"asr_text": tr["text"], "words": words}


def build_render(out: Path, deps: Deps, seed: int) -> dict:
    from voice.tts import OTHER_VOICES

    rng = random.Random(seed)
    counts = {}
    for name, src in (("prompts", "prompts-text"), ("baited", "baited-text")):
        done = out / f"artifact/{name}/{name}.jsonl"
        if name == "baited" and done.exists():  # fixed prompts: keep the earlier renders, refresh smoke flags
            smoke = {r["id"]: r["smoke"] for r in read_jsonl(out / f"work/{src}.jsonl")}
            rows = [{**r, "smoke": smoke[r["id"]]} for r in read_jsonl(done)]
            P.write_jsonl(rows, done)
            counts[name] = len(rows)
            continue
        rows = []
        for r in read_jsonl(out / f"work/{src}.jsonl"):
            voice = rng.choice(OTHER_VOICES)
            wav, tr = render_question(r["question"], voice, deps)
            rel = f"audio/{r['id']}.wav"
            write_wav(out / "artifact" / name / rel, wav)
            rows.append({**r, "user_voice": voice, "user_wav": rel, "user_s": round(len(wav) / SR, 3), **tr})
        P.write_jsonl(rows, out / f"artifact/{name}/{name}.jsonl")
        counts[name] = len(rows)
    return counts


# ---------------------------------------------------------------- voice ceiling


def build_ceiling(out: Path, deps: Deps, seed: int) -> dict:
    """Speaker similarity of Pocket TTS (cloned from the 30 s prompt segment)
    to the eval split: cosine of each render's embedding to the centroid of
    the eval clips. Reported next to the author's own prompt-split clips (the
    embedder's same-speaker level) and a catalogue voice (its other-speaker
    level), so the 0.7 gate can be read on this embedder's scale."""
    manifest = json.loads((out / "artifact/voice-eval/manifest.json").read_text())
    eval_wavs = [read_wav(out / "artifact/voice-eval" / c["path"]) for c in manifest["clips"] if c["split"] == "eval"]
    own = [read_wav(out / "voice-voice" / c["path"]) for c in manifest["clips"] if c["split"] == "prompt"]
    centroid = deps.embed(eval_wavs).mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    texts = [r["response_text"] for r in read_jsonl(out / "work/prompts-text.jsonl") if r["family"] == "story"][:N_CEILING]
    prompt = out / "artifact/voice-eval/prompt-30s.wav"
    sims = lambda wavs: list(deps.embed(wavs) @ centroid)  # noqa: E731
    result = {
        "embedder": SPEAKER_MODEL, "reference": "centroid of the eval-split clips",
        "pocket_tts_author": mean_ci(sims([deps.render_author(t, prompt) for t in texts]), seed),
        "author_prompt_clips": mean_ci(sims(own), seed),
        "pocket_tts_catalogue": mean_ci(sims([deps.render_other(t, "alba") for t in texts]), seed),
        "gate": 0.7,
    }
    result["below_gate"] = result["pocket_tts_author"]["value"] < result["gate"]
    write_json(out / "work/ceiling.json", result)
    return result


# ---------------------------------------------------------------- timed tracks


def timed_track(question: tuple[np.ndarray, dict], backchannel: tuple[np.ndarray, dict],
                interruption: tuple[np.ndarray, dict]) -> tuple[np.ndarray, list[dict]]:
    """A TRACK_S user track: question at LEAD_S, a backchannel BACKCHANNEL_AFTER_S
    after the question ends, an interruption INTERRUPT_AFTER_S after it ends,
    and the last SILENCE_S silent. Cues carry speech start/end times from the
    transcripts."""
    track = np.zeros(int(TRACK_S * SR), dtype=np.float32)
    cues = []

    def place(kind: str, at_s: float, seg: tuple[np.ndarray, dict]) -> float:
        wav, tr = seg
        i = int(at_s * SR)
        if i + len(wav) > len(track) - int(SILENCE_S * SR):
            raise ValueError(f"{kind} at {at_s:.1f}s ({len(wav) / SR:.1f}s) runs into the silence window")
        track[i : i + len(wav)] = wav
        w = tr["words"]
        start, end = (at_s + w[0]["start_s"], at_s + w[-1]["end_s"]) if w else (at_s, at_s + len(wav) / SR)
        cues.append({"type": kind, "start_s": round(start, 3), "end_s": round(end, 3), "text": tr["asr_text"]})
        return end

    q_end = place("question", LEAD_S, question)
    place("backchannel", q_end + BACKCHANNEL_AFTER_S, backchannel)
    place("interruption", q_end + INTERRUPT_AFTER_S, interruption)
    cues.append({"type": "silence", "start_s": TRACK_S - SILENCE_S, "end_s": TRACK_S, "text": ""})
    return track, cues


def build_timed(out: Path, deps: Deps, seed: int) -> dict:
    rows = [r for r in read_jsonl(out / "artifact/prompts/prompts.jsonl") if r["family"] != "author"]
    rng = random.Random(seed)
    mains = rng.sample(rows, min(N_TIMED, len(rows)))
    tracks = []
    for k, main in enumerate(mains):
        other = rng.choice([r for r in rows if r["id"] != main["id"]])
        voice = main["user_voice"]
        q = (read_wav(out / "artifact/prompts" / main["user_wav"]), {"asr_text": main["asr_text"], "words": main["words"]})
        bc = render_question(rng.choice(BACKCHANNELS), voice, deps)
        it = render_question(INTERRUPT_LEAD + other["question"], voice, deps)
        wav, cues = timed_track(q, bc, it)
        cues[0]["prompt_id"], cues[2]["prompt_id"] = main["id"], other["id"]
        tid = f"timed-{k:02d}"
        write_wav(out / f"artifact/timed/{tid}.wav", wav)
        tracks.append({"id": tid, "wav": f"{tid}.wav", "user_voice": voice, "duration_s": TRACK_S,
                       "prompt_ids": [main["id"], other["id"]], "cues": cues})
    P.write_jsonl(tracks, out / "artifact/timed/tracks.jsonl")
    return {"tracks": len(tracks)}


# ---------------------------------------------------------------- Full-Duplex-Bench


def fetch_fdb(dest: Path) -> Path:
    """Download the v1.0 / v1.5 zips from the benchmark's Google Drive folder."""
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["uvx", "gdown", "--folder", FDB_URL, "-O", str(dest)], check=True)
    return dest / "Full-Duplex-Bench-Data"


def layout_fdb(src: Path, out: Path) -> dict:
    """<src>/v1.0/<task>.zip, <src>/v1.5/<subset>.zip -> <out>/<version>/<task>/<id>/<files>,
    dropping macOS metadata. Returns sample counts per version/task."""
    counts = {}
    for version, (folder, tasks) in FDB_TASKS.items():
        for task in tasks:
            with zipfile.ZipFile(src / folder / f"{task}.zip") as z:
                ids = set()
                for info in z.infolist():
                    parts = info.filename.split("/")
                    if info.is_dir() or parts[0] != task or len(parts) != 3 or parts[2].startswith("."):
                        continue
                    target = out / version / task / parts[1] / parts[2]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as f, open(target, "wb") as g:
                        shutil.copyfileobj(f, g)
                    ids.add(parts[1])
            counts[f"{version}/{task}"] = len(ids)
    write_json(out / "index.json", {"source": FDB_URL, "counts": counts})
    (out / "LICENSES.md").write_text(FDB_LICENSES.format(url=FDB_URL))
    return counts


# ---------------------------------------------------------------- manifest, upload, log


def build_manifest(out: Path, stats: dict) -> dict:
    art = out / "artifact"
    hashes, sums = {}, []
    for d in ("prompts", "baited", "timed", "fdb", "voice-eval"):
        hashes[d], lines = dir_digest(art / d)
        sums += lines
    (art / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    prompts = read_jsonl(art / "prompts/prompts.jsonl")
    baited = read_jsonl(art / "baited/baited.jsonl")
    timed = read_jsonl(art / "timed/tracks.jsonl")
    from voice.tts import OTHER_VOICES

    manifest = {
        "name": "voice-evalset", "version": VERSION, "built_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": _git_commit(), "seeds": SEEDS,
        "counts": {
            "prompts": len(prompts), "prompts_by_family": dict(Counter(r["family"] for r in prompts)),
            "baited": len(baited), "timed": len(timed), "fdb": stats.get("fdb", {}),
            "voice_clips": dict(Counter(c["split"] for c in stats["voice"]["clips"])),
            "voice_seconds": stats["voice"]["seconds"], "passages": stats["passages"],
        },
        "sizes": {
            "smoke": {"prompts": [r["id"] for r in prompts + baited if r["smoke"]]},
            "full": {"prompts": [r["id"] for r in prompts], "baited": [r["id"] for r in baited],
                     "timed": [r["id"] for r in timed], "fdb": sorted(FDB_TASKS)},
        },
        "voice": {k: v for k, v in stats["voice"].items() if k != "clips"},
        "passage_overrides": stats["passages"].get("overrides"),
        "topic_rule": P.TOPIC_RULE,
        "heldout_from": stats["passages"].get("heldout_from"),
        "prompt_mix": {"target": FAMILY_SHARE,
                       "actual": {f: round(n / len(prompts), 3) for f, n in Counter(r["family"] for r in prompts).items()}},
        "prompts_build": stats.get("prompts"),
        "teacher": {"model": TEACHER_MODEL, "with_context": False, "label_schema": P.LABEL_SCHEMA},
        "tts": {"engine": "kyutai pocket-tts", "version": _version("pocket-tts"), "user_voices": list(OTHER_VOICES)},
        "asr": {"engine": "FermionResearch/Phonon-2", "package": f"fermion-research {_version('fermion-research')}"},
        "speaker_embedder": {"id": SPEAKER_MODEL, "checkpoint": f"{speaker.CHECKPOINT_REPO}/{speaker.CHECKPOINT_FILE}",
                             "checkpoint_sha256": speaker.CHECKPOINT_SHA256,
                             "head": "voice.vendor.ecapa_tdnn (UniSpeech 40ac183)"},
        "timed_layout": {"track_s": TRACK_S, "lead_s": LEAD_S, "backchannel_after_question_s": BACKCHANNEL_AFTER_S,
                         "interrupt_after_question_s": INTERRUPT_AFTER_S, "silence_s": SILENCE_S,
                         "cue_types": ["question", "backchannel", "interruption", "silence"]},
        "hashes": hashes,
    }
    write_json(art / "evalset.json", manifest)
    return manifest


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _version(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return "unknown"


def upload_volumes(out: Path) -> None:
    """voice-voice <- prompt/, finetune/, prompt-30s.wav, manifest.json;
    voice-data <- passages/ (source pool with labels, held-out pages, topic counts)."""
    import modal

    for name, local in (("voice-voice", out / "voice-voice"), ("voice-data", out / "voice-data")):
        vol = modal.Volume.from_name(name, create_if_missing=True)
        with vol.batch_upload(force=True) as batch:
            batch.put_directory(str(local), "/")
        logger.info("uploaded %s to volume %s", local, name)


def log_artifact(out: Path, manifest: dict, stats: dict, disabled: bool = False) -> str | None:
    from shared import tracking

    if disabled:
        os.environ["WANDB_MODE"] = "disabled"
    run = tracking.start_run("voice", tracking.group_name("evalset"), manifest, name="evalset/build",
                             job_type="dataset", tags=["evalset"])
    with run:
        teacher_h = stats.get("teacher_seconds", 0.0) / 3600
        run.log({"evalset/prompts": manifest["counts"]["prompts"], "evalset/timed": manifest["counts"]["timed"],
                 "teacher/gpu_hours": teacher_h, "teacher/usd": teacher_h * tracking.GPU_RATES_USD_PER_HOUR["H200"]})
        return run.log_artifact(out / "artifact", "voice-evalset", metadata={k: manifest[k] for k in
                                ("version", "seeds", "counts", "hashes", "sizes")})


def log_ceiling(result: dict, artifact: str | None, disabled: bool = False) -> str | None:
    """The voice ceiling as its own run in the evalset group, using the eval set artifact."""
    from shared import tracking

    if disabled:
        os.environ["WANDB_MODE"] = "disabled"
    run = tracking.start_run("voice", tracking.group_name("evalset"), result, name="pocket-tts/author/tts-ceiling",
                             job_type="tts-ceiling", tags=["evalset"])
    with run:
        if artifact:
            run.use_artifact(artifact, type="voice-evalset", download=False)
        run.log({f"voice/speaker_sim/{k}": v["value"] for k, v in result.items() if isinstance(v, dict)})
        for k, v in result.items():
            if isinstance(v, dict):
                run.summary[f"voice/speaker_sim/{k}_ci"] = [v["ci_low"], v["ci_high"], v["n"]]
        return run.url


# ---------------------------------------------------------------- deps


def real_deps(model: str = TEACHER_MODEL) -> Deps:
    from shared import teacher as T
    from voice import evals, tts

    handle = T.teacher_cls(model, max_inputs=64)
    started: list[float] = []

    def generate(batch, **kw):
        if not started:
            started.append(time.monotonic())
        return T.modal_generate(handle)(batch, **kw)

    return Deps(
        generate=generate,
        render_author=lambda text, prompt: tts.render_author(text, prompt),
        render_other=tts.render_other,
        transcribe=lambda wav: evals.transcribe(wav, SR),
        embed=speaker_embed,
        teacher_seconds=lambda: (time.monotonic() - started[0]) if started else 0.0,
    )


def speaker_embed(wavs: list[np.ndarray]) -> np.ndarray:
    """Unit-norm x-vectors from SPEAKER_MODEL, wavs resampled 24 -> 16 kHz."""
    from voice.evals import speaker_embedding

    return speaker_embedding(wavs, SR)


# ---------------------------------------------------------------- dry run fixture


def fake_deps() -> Deps:
    """Deterministic stand-ins: the Teacher labels everything 'other' and
    composes responses from the passage's own sentences; TTS is a tone whose
    length follows the word count; the transcript spaces words evenly."""
    from voice.passages import LABEL_SCHEMA

    class C:
        def __init__(self, text):
            self.text = text

    def generate(batch, json_schema=None, **kw):
        out = []
        for msgs in batch:
            user = msgs[-1]["content"]
            if json_schema == LABEL_SCHEMA:
                out.append(C(json.dumps({"topic": "other", "borderline": False})))
            elif json_schema == data.PICK_SCHEMA:
                out.append(C(json.dumps({"picks": [0]})))
            else:
                blocks = [b.split("\n", 1)[1] for b in user.split("Passages:\n", 1)[1].split("\n\nQuestion:")[0].split("\n\n")]
                sents = [s for b in blocks for s in data.SENTENCE.findall(b)][:4]
                out.append(C(json.dumps({"question": "How have you been lately?", "response": " ".join(sents)})))
        return out

    def tone(text: str, *_):
        n = int(SR * (0.3 + 0.35 * len(text.split())))
        t = np.arange(n) / SR
        return (0.1 * np.sin(2 * np.pi * (120 + len(text) % 50) * t)).astype(np.float32)

    def transcribe(wav):
        n = max(1, int(len(wav) / SR / 0.35))
        return {"text": "w " * n, "words": [{"w": "w", "start_s": 0.1 + 0.35 * i, "end_s": 0.4 + 0.35 * i, "conf": None}
                                           for i in range(n)]}

    def embed(wavs):
        v = np.stack([np.array([np.abs(w).mean(), w.std(), len(w) % 7 + 1.0]) for w in wavs])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    return Deps(generate=generate, render_author=tone, render_other=tone, transcribe=transcribe, embed=embed)


def fixture(root: Path, seed: int = 0) -> dict[str, Path]:
    """A tiny journal (two journals, 2021-2024, first-person sentences), a
    40 s voice recording (tone bursts with pauses) as m4a, author questions,
    and Full-Duplex-Bench zips with two samples per task."""
    rng = random.Random(seed)
    words = "garden river morning coffee project city friend walk book music letter window".split()
    md = []
    for k in range(48):
        year, journal = 2021 + k % 4, ("Notebook A", "Notebook B")[k % 2]
        body = "\n".join(f"- I went to the {rng.choice(words)} and I felt the {rng.choice(words)} was close to my "
                         f"{rng.choice(words)} again. My {rng.choice(words)} and my {rng.choice(words)} stayed with me "
                         f"all day long." for _ in range(4))
        md.append(f"## [{journal}:p.{k + 1}] date: {year}-0{1 + k % 9}-1{k % 9} (est.)\n\n{body}\n")
    (root / "journal.md").write_text("\n".join(md))
    t = np.arange(int(0.9 * SR)) / SR
    burst = 0.3 * np.sin(2 * np.pi * 180 * t)
    wav = np.concatenate([np.concatenate([burst * (0.5 + 0.5 * rng.random()), np.zeros(int(0.5 * SR))]) for _ in range(28)])
    sf.write(root / "voice.wav", wav.astype(np.float32), SR)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(root / "voice.wav"),
                    "-c:a", "aac", str(root / "voice.m4a")], check=True)
    (root / "questions.jsonl").write_text(json.dumps({"question": "What are my core beliefs?", "split": "eval",
                                                      "kind_v2": "self_synthesis", "kind": "question"}) + "\n")
    for version, (folder, tasks) in FDB_TASKS.items():
        (root / "fdb" / folder).mkdir(parents=True, exist_ok=True)
        for task in tasks:
            with zipfile.ZipFile(root / "fdb" / folder / f"{task}.zip", "w") as z:
                for i in range(2):
                    z.writestr(f"{task}/{i}/input.wav", b"RIFF")
                    z.writestr(f"{task}/{i}/.DS_Store", b"")
                z.writestr(f"__MACOSX/{task}/._0", b"")
    return {"journal": root / "journal.md", "voice": root / "voice.m4a", "questions": root / "questions.jsonl",
            "fdb": root / "fdb"}


# ---------------------------------------------------------------- driver

STAGES = ("voice", "passages", "prompts", "render", "timed", "fdb", "manifest", "upload", "log", "ceiling")


def build(out: Path, deps: Deps, journal: Path, voice_src: Path, questions: Path, fdb_src: Path | None,
          stages: tuple[str, ...] = STAGES, dry_run: bool = False, overrides: Path | None = None,
          previous_heldout: Path | None = None) -> dict:
    """Run `stages` in order; each stage's result is kept in work/stats.json so
    later stages (and re-runs) can read it."""
    stats_path = out / "work/stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    art = out / "artifact"
    teacher_base = stats.get("teacher_seconds", 0.0)  # Teacher wall time of earlier runs
    done = {
        "voice": lambda: (art / "voice-eval/manifest.json").exists(),
        "passages": lambda: (art / "prompts/passages.jsonl").exists(),
        "prompts": lambda: (out / "work/prompts-text.jsonl").exists(),
        "render": lambda: (art / "prompts/prompts.jsonl").exists(),
        "ceiling": lambda: "ceiling" in stats,
        "timed": lambda: (art / "timed/tracks.jsonl").exists(),
        "fdb": lambda: (art / "fdb/index.json").exists(),
    }
    for stage in stages:
        if stage in done and done[stage]() and stage in stats:
            logger.info("stage %s: done, skipping", stage)
            continue
        logger.info("stage %s", stage)
        if stage == "voice":
            stats["voice"] = build_voice(voice_src, out, SEEDS["voice"])
        elif stage == "passages":
            stats["passages"] = build_passages(journal, out, deps, SEEDS["heldout"], overrides, previous_heldout)
        elif stage == "prompts":
            stats["prompts"] = build_prompts(out, deps, author_questions(questions), SEEDS["prompts"])
        elif stage == "render":
            stats["render"] = build_render(out, deps, SEEDS["voices"])
        elif stage == "ceiling":
            stats["ceiling"] = build_ceiling(out, deps, SEEDS["bootstrap"])
            stats["ceiling_run"] = log_ceiling(stats["ceiling"], stats.get("artifact"), disabled=dry_run)
        elif stage == "timed":
            stats["timed"] = build_timed(out, deps, SEEDS["timed"])
        elif stage == "fdb":
            stats["fdb"] = layout_fdb(fdb_src or fetch_fdb(out / "work/fdb-download"), art / "fdb")
        elif stage == "manifest":
            stats["manifest"] = build_manifest(out, stats)
        elif stage == "upload":
            if dry_run:
                logger.info("dry run: not uploading volumes")
            else:
                upload_volumes(out)
        elif stage == "log":
            stats["artifact"] = log_artifact(out, stats["manifest"], stats, disabled=dry_run)
        stats["teacher_seconds"] = teacher_base + deps.teacher_seconds()
        write_json(stats_path, stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m voice.evalset", description=__doc__.splitlines()[0])
    p.add_argument("--stage", action="append", choices=STAGES, help="run only these stages (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="synthetic fixture, fake models, no upload, W&B off")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--journal", type=Path, default=JOURNAL)
    p.add_argument("--voice", type=Path, default=VOICE_SRC)
    p.add_argument("--questions", type=Path, default=AUTHOR_QUESTIONS)
    p.add_argument("--fdb", type=Path, default=None, help="directory holding v1.0/ and v1.5/ zips (default: download)")
    p.add_argument("--model", default=TEACHER_MODEL)
    p.add_argument("--heldout", type=Path, default=None,
                   help="an earlier build's heldout.json: keep its pages frozen, recompute the passage ids")
    args = p.parse_args(argv)
    stages = tuple(s for s in STAGES if s in (args.stage or STAGES))
    if args.dry_run:
        root = Path(tempfile.mkdtemp(prefix="evalset-dry-"))
        fx = fixture(root)
        stats = build(args.out or root / "out", fake_deps(), fx["journal"], fx["voice"], fx["questions"], fx["fdb"],
                      stages, dry_run=True)
        print(json.dumps(stats["manifest"]["counts"], indent=2))
        return 0
    deps = real_deps(args.model)
    try:
        build(args.out or OUT, deps, args.journal, args.voice, args.questions, args.fdb, stages, overrides=OVERRIDES,
              previous_heldout=args.heldout)
    finally:
        from shared import teacher

        teacher.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
