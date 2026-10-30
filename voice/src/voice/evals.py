"""Evaluation of a served endpoint (brief §8, issue #9): `voice eval`.

    uv run voice eval --url <wss://...> --backend personaplex|hertz --evalset voice-evalset:v1 \\
        --size smoke|full --group exp-YYYYMMDD-<slug> [--concurrency N] [--dry-run]

Everything is scored through `voice.client.Client` against the deployment,
audio in and audio out, never a Python model object. Inputs come from the
`voice-evalset` W&B artifact (#22, #26). Output: `<out>/report.json`

    {config, rows: {"<axis>/<metric>[/<stratum>]": {value, ci_low, ci_high, n}}, items: [...]}

and one W&B run `<backend>/<system>/eval-<size>` (brief §4). CIs are 95%
percentile bootstrap over prompts (or runs, tracks, items), with binomial
(Wilson) CIs for counts.

Jobs (one WebSocket conversation each), by size:

    smoke  the evalset's 10 smoke prompts (8 held-out + 2 baited)           ~4 min of audio
    full   all 200 held-out prompts + 20 baited; `timed_n` timed tracks; `fdb_per_task`
           items of FDB v1 turn-taking / user interruption and of the 4 FDB v1.5 tasks;
           `onset_runs` x `onset_s` of user silence; `drift_runs` x `drift_s` continuous talks.
           Subsample sizes are config (recorded in the run) because the protocol as written
           (all 60 tracks, all 1,225 FDB items, 20 x 5 min silence) is ~13 GPU-h per system.

Rows (summary names fixed by brief §4 are marked *):

    responsiveness  latency_s* (first agent word after the user's last Phonon-2 word, as heard),
                    model_latency_s (latency_s - the client's measured transport_s), answered,
                    fdb1_turn_latency_s / fdb1_turn_tor, fdb1_interrupt_latency_s / fdb1_interrupt_tor,
                    spurious_onset (agent speech after ONSET_GRACE_S of silence), greeting
    interaction     fdb15* (macro over 4 scenarios of the right behaviour: respond to an
                    interruption; do not respond to a backchannel, a third party or background
                    speech), fdb15/<scenario>, cuespeak (macro over cue types question /
                    backchannel / interruption: judge says the meaning is conveyed AND the timing
                    holds), cuespeak/<cue>
    fidelity        coverage* (4-gram coverage of the transcript against the prompt's source
                    passages; story prompts; per family as coverage/<family>), first_person,
                    citations, judge* (voice-v0 non-loss rate vs the reference response, both
                    orders), judge_content_error, chronology (change prompts, judge),
                    topic_violations (count over the baited prompts; target 0)
    voice           speaker_sim* (WavLM-TDNN cosine of agent speech to the centroid of the
                    author's eval clips), drift (last minute minus first minute), utmos,
                    word_conf (mean per-word confidence, NeMo Parakeet-TDT), dict_share,
                    wer (PersonaPlex: text stream vs Phonon-2 transcript)
    naturalness     pref* (needs human ratings; exported pairs only, n = 0 until rated)

Timing rule: every "within X s" check is on model time, i.e. onset minus the
conversation's transport_s, so PersonaPlex's ~0.6 s Opus hold and hertz's
~2 ms are not charged to the model.

Every model call (the conversation, Phonon-2, the speaker embedder, UTMOS, the
word-confidence decoder, the Qwen judge) is injected through `Deps`, so
`--dry-run` runs all of it on a synthetic eval set on the CPU, offline.

Transcription is Phonon-2 through `transcribe(wav) -> {text, words[{w,
start_s, end_s, conf}]}` (the `phonon transcribe --json` CLI of
`fermion-research`). Checked on 0.2.15 (#9): the CLI emits no per-word
confidence, and its local engine is a compiled library, so `conf` stays None
and the word confidence comes from NeMo decoding the same checkpoint lineage
(`nvidia/parakeet-tdt-0.6b-v3`, entropy confidence) on a Modal CPU function
(`ASR_APP`); Phonon-2 still supplies all text and timestamps.
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import functools
import json
import logging
import math
import random
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from voice import data, speaker
from voice import passages as P

logger = logging.getLogger(__name__)

PHONON = "phonon"
SPEAKER_MODEL = speaker.MODEL_ID  # recorded in evalset.json
RUBRIC = Path(__file__).resolve().parents[2] / "rubrics/voice-v0.md"
JUDGE_MODEL = "Qwen/Qwen3.8-27B"
H200_USD_PER_HOUR = 4.54
SCALEDOWN_S = 300  # voice-serve's idle window, billed once per container after the last session

ONSET_GRACE_S = 10.0  # speech in the first seconds of a silent run is a greeting, not a spurious onset
CUE_WINDOW_S = 1.0  # a cue response must start within this long (model time) after the cue ends
REACTION_S = 1.0  # time allowed to stop talking once an interruption starts
MIN_SEGMENT_S = 1.0  # shortest agent speech scored for voice metrics
FDB_V1 = ("candor_turn_taking", "synthetic_user_interruption")  # the two v1 tasks section 8 reports
FDB_V15 = ("user_interruption", "user_backchannel", "talking_to_other", "background_speech")
FDB_V15_EXPECT = {"user_interruption": "respond", "user_backchannel": "not_respond",
                  "talking_to_other": "not_respond", "background_speech": "not_respond"}
CUE_TYPES = ("question", "backchannel", "interruption")
DICTIONARY = Path("/usr/share/dict/words")

Kind = Literal["prompt", "baited", "timed", "fdb", "onset", "drift"]


# ---------------------------------------------------------------- Phonon-2 and the speaker embedder


def parse_phonon(out: dict) -> dict:
    """`phonon transcribe --json` output -> {text, words[{w, start_s, end_s, conf}]}."""
    words = [{"w": w["text"], "start_s": float(w["start"]), "end_s": float(w["end"]), "conf": w.get("confidence")}
             for w in out.get("words", [])]
    return {"text": out.get("text", ""), "words": words}


def transcribe(wav: str | Path | np.ndarray, sample_rate: int = 24000, run=subprocess.run) -> dict:
    if isinstance(wav, np.ndarray):
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, wav, sample_rate)
            return transcribe(f.name, run=run)
    proc = run([PHONON, "transcribe", "--json", str(wav)], capture_output=True, text=True, check=True)
    return parse_phonon(json.loads(proc.stdout))


@functools.cache
def _speaker_model():
    from voice.speaker import WavLMTdnn

    return WavLMTdnn.load()


def speaker_embedding(wavs: list[np.ndarray], sample_rate: int = 24000) -> np.ndarray:
    """Unit-norm WavLM-TDNN embeddings [N, 256], one per wav (resampled to 16 kHz)."""
    from scipy.signal import resample_poly

    from voice.speaker import SAMPLE_RATE

    model = _speaker_model()
    return np.stack([model(resample_poly(w, SAMPLE_RATE, sample_rate) if sample_rate != SAMPLE_RATE else w).numpy()
                     for w in wavs])


def speaker_similarity(wavs_a: list[np.ndarray], wavs_b: list[np.ndarray], sample_rate: int = 24000) -> np.ndarray:
    """Cosine of each wav in `wavs_a` to the centroid of `wavs_b` (e.g. the author's
    eval-split clips): [len(wavs_a)]."""
    ref = speaker_embedding(wavs_b, sample_rate).mean(axis=0)
    return speaker_embedding(wavs_a, sample_rate) @ (ref / np.linalg.norm(ref))


# ---------------------------------------------------------------- rows


def row(values, seed: int = 0, n_boot: int = 2000) -> dict:
    """Mean with a 95% percentile-bootstrap CI: {value, ci_low, ci_high, n}; None values are dropped."""
    x = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(x):
        return {"value": None, "ci_low": None, "ci_high": None, "n": 0}
    boots = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"value": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": len(x)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval of a proportion k / n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - half) / (1 + z * z / n)), min(1.0, (centre + half) / (1 + z * z / n)))


def rate_row(k: int, n: int) -> dict:
    """A proportion k / n with its Wilson interval."""
    lo, hi = wilson(k, n)
    return {"value": k / n if n else None, "ci_low": lo if n else None, "ci_high": hi if n else None, "n": n}


def count_row(k: int, n: int) -> dict:
    """A count out of n, with the Wilson interval scaled to counts."""
    lo, hi = wilson(k, n)
    return {"value": float(k), "ci_low": lo * n, "ci_high": hi * n, "n": n}


def macro_row(parts: dict[str, list], seed: int = 0, n_boot: int = 2000) -> dict:
    """Macro average of per-group means, bootstrapped within each group."""
    groups = [np.asarray([v for v in vs if v is not None], dtype=np.float64) for vs in parts.values()]
    groups = [g for g in groups if len(g)]
    if not groups:
        return row([])
    rng = np.random.default_rng(seed)
    boots = np.mean([g[rng.integers(0, len(g), (n_boot, len(g)))].mean(axis=1) for g in groups], axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"value": float(np.mean([g.mean() for g in groups])), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(sum(len(g) for g in groups))}


# ---------------------------------------------------------------- words and timing (pure)


def first_onset(words: list[dict], after_s: float) -> float | None:
    """Start of the first word that starts at or after `after_s`."""
    starts = [w["start_s"] for w in words if w["start_s"] >= after_s]
    return min(starts) if starts else None


def words_in(words: list[dict], start_s: float, end_s: float) -> list[dict]:
    """Words that overlap [start_s, end_s)."""
    return [w for w in words if w["end_s"] > start_s and w["start_s"] < end_s]


def text_of(words: list[dict]) -> str:
    return " ".join(w["w"] for w in words)


def response_latency(words: list[dict], user_end_s: float) -> float | None:
    """First agent word start after the user's last word, minus that word's end."""
    onset = first_onset(words, user_end_s)
    return None if onset is None else onset - user_end_s


def model_time(latency: float | None, transport_s: float | None) -> float | None:
    return None if latency is None or transport_s is None else latency - transport_s


def spurious_onset(words: list[dict], grace_s: float = ONSET_GRACE_S) -> bool:
    """The agent spoke after the grace window of a silent run."""
    return first_onset(words, grace_s) is not None


def speech_span(words: list[dict], after_s: float = 0.0) -> tuple[float, float] | None:
    """(first start, last end) of the words starting at or after `after_s`."""
    ws = [w for w in words if w["start_s"] >= after_s]
    return (ws[0]["start_s"], ws[-1]["end_s"]) if ws else None


def cut(wav: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    return wav[max(0, int(start_s * sample_rate)): max(0, int(end_s * sample_rate))]


def speech_audio(wav: np.ndarray, sample_rate: int, words: list[dict], start_s: float, end_s: float) -> np.ndarray:
    """The agent's word spans inside [start_s, end_s), concatenated (silence dropped)."""
    parts = [cut(wav, sample_rate, max(w["start_s"], start_s), min(w["end_s"], end_s))
             for w in words_in(words, start_s, end_s)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


_NORM = re.compile(r"[a-z0-9']+")


def norm_words(text: str) -> list[str]:
    return _NORM.findall(text.lower().replace("’", "'"))


def wer(reference: str, hypothesis: str) -> float | None:
    """Word error rate of `hypothesis` against `reference` (Levenshtein over normalised words)."""
    ref, hyp = norm_words(reference), norm_words(hypothesis)
    if not ref:
        return None
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


@functools.cache
def dictionary(path: Path = DICTIONARY) -> frozenset[str]:
    """Lower-cased word list (empty if the system has none)."""
    return frozenset(w.strip().lower() for w in path.read_text().splitlines()) if path.exists() else frozenset()


def dict_share(text: str, vocab: frozenset[str]) -> float | None:
    """Share of transcript words in the dictionary (a contraction counts by its stem)."""
    toks = norm_words(text)
    if not toks or not vocab:
        return None
    return sum(t in vocab or t.split("'")[0] in vocab for t in toks) / len(toks)


def cue_timing(cue: dict, words: list[dict], transport_s: float, next_s: float) -> dict:
    """Timing checks for one timed-test cue (model time; module docstring).

    question, interruption: the agent starts within CUE_WINDOW_S of the cue's end;
    interruption also: no agent word between REACTION_S after it starts and its end
    (the agent yields); backchannel: the agent is talking within CUE_WINDOW_S after it
    (keeps going or resumes). `next_s` bounds the response window (next cue or silence)."""
    t = transport_s or 0.0
    late = cue["end_s"] + t + CUE_WINDOW_S
    if cue["type"] == "backchannel":
        talking = bool(words_in(words, cue["start_s"], late))
        return {"onset_ok": talking, "yield_ok": True, "onset_s": first_onset(words, cue["start_s"])}
    onset = first_onset(words, cue["end_s"])
    ok = onset is not None and onset <= late and onset < next_s
    yield_ok = True
    if cue["type"] == "interruption":
        yield_ok = not words_in(words, cue["start_s"] + t + REACTION_S, cue["end_s"])
    return {"onset_ok": ok, "yield_ok": yield_ok, "onset_s": onset}


# ---------------------------------------------------------------- judge prompts (pure)

CUE_SCHEMA = {"type": "object", "properties": {"conveys": {"type": "boolean"}, "reason": {"type": "string"}},
              "required": ["conveys", "reason"]}
BEHAVIOUR_SCHEMA = {"type": "object",
                    "properties": {"behaviour": {"type": "string", "enum": ["respond", "resume", "silent", "other"]},
                                   "reason": {"type": "string"}},
                    "required": ["behaviour", "reason"]}
CHRONO_SCHEMA = {"type": "object",
                 "properties": {"both_retold": {"type": "boolean"}, "in_order": {"type": "boolean"},
                                "reason": {"type": "string"}},
                 "required": ["both_retold", "in_order", "reason"]}

CUE_PROMPT = """A voice model talks as the author of a journal. During a scripted conversation the user {event}.
Below is what the model said in the seconds after that, as a speech transcript, and the content it was
expected to convey (the author's journal passages and a reference answer).

Expected content:
{expected}

Model's speech after the cue:
{said}

Does the model's speech convey the expected meaning (the same story, idea or facts, in any wording)?
For a backchannel, the model passes if it carries on with the expected answer.
Reply with JSON: {{"conveys": true | false, "reason": "<one sentence>"}}"""
CUE_EVENTS = {"question": "asked a question", "backchannel": "said a short backchannel (\"mm-hm\") while the model "
              "was answering", "interruption": "interrupted with a new question"}

BEHAVIOUR_PROMPT = """A spoken assistant was answering a user. Then {event}. Transcripts:

The user's request: {context}
The overlapping speech: {overlap}
The assistant before it: {before}
The assistant after it: {after}

Classify what the assistant did after the overlapping speech:
- "respond": it addressed the overlapping speech (answered it, acknowledged it, switched to it);
- "resume": it carried on with or finished its answer to the original request, ignoring the overlap;
- "silent": it said nothing;
- "other": anything else.
Reply with JSON: {{"behaviour": "respond" | "resume" | "silent" | "other", "reason": "<one sentence>"}}"""
FDB15_EVENTS = {"user_interruption": "the user interrupted with a new request",
                "user_backchannel": "the user gave a short backchannel",
                "talking_to_other": "the user said something to another person in the room",
                "background_speech": "someone in the background said something"}

CHRONO_PROMPT = """A voice model talks as the author of a journal. Asked about how they changed, it should retell
two journal passages in the order they were written. Passages, oldest first:

{passages}

The model's spoken answer (speech transcript):
{said}

Does the answer retell both passages, and does it tell them in the order written (older first)?
Reply with JSON: {{"both_retold": true | false, "in_order": true | false, "reason": "<one sentence>"}}"""


def user_msg(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def passages_block(passages: list[dict]) -> str:
    return "\n\n".join(f"[{p['entry_date']}] {p['text']}" for p in sorted(passages, key=lambda p: p["entry_date"]))


def ask(generate: Callable, prompts: list[str], schema: dict, seed: int = 0) -> list[dict | None]:
    """One greedy JSON-constrained batch; unparsed replies come back as None."""
    if not prompts:
        return []
    replies = generate([user_msg(p) for p in prompts], max_tokens=256, temperature=0.0, seed=seed,
                       with_context=False, json_schema=schema)
    return [parse_json(r.text if r is not None else None) for r in replies]


# ---------------------------------------------------------------- the eval set


class Job(BaseModel):
    """One conversation with the endpoint."""

    id: str
    kind: Kind
    wav: str  # user audio (any rate; the client resamples)
    max_s: float
    meta: dict = {}


class EvalSet:
    """A downloaded `voice-evalset` artifact directory (layout: voice.evalset)."""

    def __init__(self, root: Path, version: str = "local"):
        self.root, self.version = Path(root), version
        self.manifest = json.loads((self.root / "evalset.json").read_text())
        self.prompts = {r["id"]: r for r in _jsonl(self.root / "prompts/prompts.jsonl")}
        self.baited = {r["id"]: r for r in _jsonl(self.root / "baited/baited.jsonl")}
        self.passages = {r["passage_id"]: r for r in _jsonl(self.root / "prompts/passages.jsonl")}
        self.tracks = {r["id"]: r for r in _jsonl(self.root / "timed/tracks.jsonl")}
        clips = json.loads((self.root / "voice-eval/manifest.json").read_text())["clips"]
        self.eval_clips = [self.root / "voice-eval" / c["path"] for c in clips if c["split"] == "eval"]

    def item(self, pid: str) -> dict:
        return self.prompts.get(pid) or self.baited[pid]

    def sources(self, pid: str) -> list[dict]:
        return [self.passages[i] for i in self.item(pid)["passage_ids"] if i in self.passages]

    def fdb_items(self, version: str, task: str) -> list[Path]:
        d = self.root / "fdb" / version / task
        return sorted((p for p in d.iterdir() if p.is_dir()), key=lambda p: int(p.name)) if d.exists() else []


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def user_end(r: dict) -> float:
    """End of the user's last Phonon-2 word in a prompt wav (the latency origin)."""
    return r["words"][-1]["end_s"] if r.get("words") else r["user_s"]


class EvalConfig(BaseModel):
    url: str
    backend: str
    evalset: str = "voice-evalset:v1"
    size: Literal["smoke", "full"] = "smoke"
    group: str
    system: str = "stock"
    out: Path = Path("voice/outputs/eval")
    concurrency: int = 1  # conversations at once (one per voice-serve container)
    seed: int = 0
    tail_s: float = 20.0  # conversation length after the user's question
    timed_n: int = 30
    fdb_per_task: int = 15
    fdb_tail_s: float = 8.0
    onset_runs: int = 10
    onset_s: float = 210.0  # hertz closes at ~226 s of live audio
    drift_runs: int = 5
    drift_s: float = 210.0
    drift_every_s: float = 26.0
    drift_window_s: float = 60.0
    judge_model: str = JUDGE_MODEL
    utmos: bool = False  # UTMOS22 runs only when asked (its route is pending; the row is n = 0 otherwise)
    check: bool = False  # a deployment check: stage serve-smoke, tag check, never a baseline or candidate
    limit: dict[str, int] = {}  # at most this many jobs per kind, or per "fdb/<version>" (profiling runs)
    side_by_side: list[str] = []  # prompt ids (chosen by the coordinator); default: the smoke story/change ids
    dry_run: bool = False

    @property
    def run_name(self) -> str:
        return f"{self.backend}/{self.system}/{'serve-smoke' if self.check else 'eval-' + self.size}"


def plan(es: EvalSet, cfg: EvalConfig, inputs: Path) -> list[Job]:
    """The conversations of one eval, in a fixed order (seeded subsamples)."""
    import soundfile as sf

    rng = random.Random(cfg.seed)
    jobs = []
    full = cfg.size == "full"
    ids = es.manifest["sizes"]["full" if full else "smoke"]
    pids = ids["prompts"] + (ids.get("baited", []) if full else [])
    for pid in pids:
        r = es.item(pid)
        kind = "baited" if r["family"] == "baited" else "prompt"
        jobs.append(Job(id=pid, kind=kind, wav=str(es.root / ("baited" if kind == "baited" else "prompts") / r["user_wav"]),
                        max_s=round(user_end(r) + cfg.tail_s, 2)))
    if not full:
        return limit(jobs, cfg.limit)
    for tid in sorted(rng.sample(sorted(es.tracks), min(cfg.timed_n, len(es.tracks)))):
        t = es.tracks[tid]
        jobs.append(Job(id=tid, kind="timed", wav=str(es.root / "timed" / t["wav"]), max_s=t["duration_s"]))
    for version, tasks in (("v1", FDB_V1), ("v1.5", FDB_V15)):
        for task in tasks:
            items = es.fdb_items(version, task)
            for d in sorted(rng.sample(items, min(cfg.fdb_per_task, len(items))), key=lambda p: int(p.name)):
                dur = sf.info(d / "input.wav").duration
                jobs.append(Job(id=f"fdb-{version}-{task}-{d.name}", kind="fdb", wav=str(d / "input.wav"),
                                max_s=round(dur + cfg.fdb_tail_s, 2), meta={"version": version, "task": task, "dir": str(d)}))
    inputs.mkdir(parents=True, exist_ok=True)
    silence = inputs / f"silence-{int(cfg.onset_s)}s.wav"
    sf.write(silence, np.zeros(int(cfg.onset_s * 16000), dtype=np.float32), 16000)
    jobs += [Job(id=f"onset-{k:02d}", kind="onset", wav=str(silence), max_s=cfg.onset_s) for k in range(cfg.onset_runs)]
    pool = sorted(p for p, r in es.prompts.items() if not r.get("smoke"))
    per = int(cfg.drift_s // cfg.drift_every_s)
    for k in range(cfg.drift_runs):
        chosen = rng.sample(pool, min(per, len(pool)))
        wav = drift_track([es.root / "prompts" / es.prompts[p]["user_wav"] for p in chosen], cfg.drift_s,
                          cfg.drift_every_s)
        path = inputs / f"drift-{k:02d}.wav"
        sf.write(path, wav, 24000)
        jobs.append(Job(id=f"drift-{k:02d}", kind="drift", wav=str(path), max_s=cfg.drift_s, meta={"prompts": chosen}))
    return limit(jobs, cfg.limit)


def limit(jobs: list[Job], caps: dict[str, int]) -> list[Job]:
    """The first caps[key] jobs per key (kind, or "fdb/<version>" for FDB); uncapped keys pass."""
    seen: Counter = Counter()
    out = []
    for j in jobs:
        key = f"fdb/{j.meta['version']}" if j.kind == "fdb" else j.kind
        seen[key] += 1
        if key not in caps or seen[key] <= caps[key]:
            out.append(j)
    return out


def drift_track(wavs: list[Path], total_s: float, every_s: float, sample_rate: int = 24000) -> np.ndarray:
    """One continuous user track: question k starts at k * every_s."""
    import soundfile as sf

    out = np.zeros(int(total_s * sample_rate), dtype=np.float32)
    for k, p in enumerate(wavs):
        w, sr = sf.read(p, dtype="float32")
        assert sr == sample_rate, (p, sr)
        i = int(k * every_s * sample_rate)
        n = min(len(w), len(out) - i)
        if n > 0:
            out[i:i + n] = w[:n]
    return out


# ---------------------------------------------------------------- running


@dataclasses.dataclass
class Deps:
    """Every model call `voice eval` makes; real ones from `real_deps`, fakes in the dry run."""

    converse: Callable[[Job, Path], object]  # -> voice.client.Conversation
    transcribe: Callable[[str], dict]  # wav path -> {text, words}
    embed: Callable[[list[np.ndarray], int], np.ndarray]  # wavs at a rate -> [N, D] unit vectors
    utmos: Callable[[list[np.ndarray], int], list[float | None]]
    word_conf: Callable[[list[np.ndarray], int], list[float | None]]  # mean per-word confidence per wav
    generate: Callable  # the judge: (batch, **kw) -> [Completion | None]
    phonon_version: str = "unknown"


def converse_all(jobs: list[Job], deps: Deps, out: Path, concurrency: int = 1) -> list[dict]:
    """Run every job not already in `<out>/items.jsonl` (so a crashed run resumes), then
    transcribe the agent wavs; returns one record per job, in job order."""
    path = out / "items.jsonl"
    done = {r["id"]: r for r in _jsonl(path)} if path.exists() else {}
    todo = [j for j in jobs if j.id not in done]
    logger.info("%d conversations (%d already done)", len(todo), len(jobs) - len(todo))

    def one(job: Job) -> dict:
        t0 = time.monotonic()
        try:
            c = deps.converse(job, out / "audio")
        except Exception as e:  # a dropped connection or a busy container: one retry, then the run stops (resumable)
            logger.warning("%s failed (%s: %s); retrying once", job.id, type(e).__name__, e)
            c = deps.converse(job, out / "audio")
        t1 = time.monotonic()
        tr = deps.transcribe(c.agent_wav)
        talk = t1 - t0
        spans = {"converse_s": talk, "handshake_s": c.handshake_s, "stream_s": c.user_s,
                 "tail_s": max(0.0, job.max_s - c.user_s), "close_s": talk - c.handshake_s - job.max_s,
                 "transcribe_s": time.monotonic() - t1}
        return {"spans": spans, "id": job.id, "kind": job.kind, "meta": job.meta, "max_s": job.max_s, "agent_wav": c.agent_wav,
                "user_wav": c.user_wav, "sample_rate": c.sample_rate, "handshake_s": c.handshake_s,
                "transport_s": c.transport_s, "transport": c.transport, "text_stream": c.response_text,
                "transcript": tr["text"], "words": tr["words"], "wall_s": time.monotonic() - t0}

    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool, open(path, "a") as f:
        for rec in pool.map(one, todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[rec["id"]] = rec
            logger.info("%s %s: %d words, transport %.3s s", rec["kind"], rec["id"], len(rec["words"]),
                        rec["transport_s"])
    return [done[j.id] for j in jobs]


def read_wav(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    return (wav if wav.ndim == 1 else wav.mean(axis=1)), sr


# ---------------------------------------------------------------- scoring


def score(es: EvalSet, recs: list[dict], deps: Deps, cfg: EvalConfig,
          prof: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """report rows and per-item records (each rec gains a `scores` dict). `prof` collects
    {span, s, n} for every batch stage (judges, WavLM, NeMo, UTMOS)."""
    prof = [] if prof is None else prof

    def clock(span: str, fn: Callable) -> Callable:
        def timed(*args, **kw):
            t0 = time.monotonic()
            out = fn(*args, **kw)
            prof.append({"span": span, "s": time.monotonic() - t0, "n": len(args[0]) if args else 0})
            return out
        return timed

    deps = dataclasses.replace(deps, embed=clock("wavlm", deps.embed), utmos=clock("utmos", deps.utmos),
                               word_conf=clock("nemo", deps.word_conf))
    judge = lambda name: clock(name, deps.generate)  # noqa: E731
    by_kind = defaultdict(list)
    for r in recs:
        r["scores"] = {}
        by_kind[r["kind"]].append(r)
    rows: dict[str, dict] = {}
    seed = cfg.seed
    prompts = by_kind["prompt"]

    # responsiveness and fidelity on held-out prompts
    for r in prompts + by_kind["baited"]:
        p = es.item(r["id"])
        end = user_end(p)
        lat = response_latency(r["words"], end)
        said = [w for w in r["words"] if w["start_s"] >= end]
        r["scores"] |= {"family": p["family"], "user_end_s": end, "latency_s": lat,
                        "model_latency_s": model_time(lat, r["transport_s"]), "said": text_of(said)}
        if p["family"] != "baited":
            r["scores"] |= {"coverage": data.coverage(r["scores"]["said"], [s["text"] for s in es.sources(r["id"])]),
                            "first_person": data.first_person_rate(r["scores"]["said"]) if said else None,
                            "citation": float(bool(data.CITATION.search(r["scores"]["said"])
                                                   or data.THIRD_PERSON.search(r["scores"]["said"].lower())))}
    answered = prompts + by_kind["baited"]
    rows["responsiveness/latency_s"] = row([r["scores"]["latency_s"] for r in answered], seed)
    rows["responsiveness/model_latency_s"] = row([r["scores"]["model_latency_s"] for r in answered], seed)
    rows["responsiveness/answered"] = row([float(r["scores"]["latency_s"] is not None) for r in answered], seed)
    rows["responsiveness/transport_s"] = row([r["transport_s"] for r in answered], seed)
    story = [r for r in prompts if r["scores"]["family"] == "story"]
    rows["fidelity/coverage"] = row([r["scores"]["coverage"] for r in story], seed)
    rows["fidelity/first_person"] = row([r["scores"]["first_person"] for r in prompts], seed)
    rows["fidelity/citations"] = row([r["scores"]["citation"] for r in prompts], seed)
    for fam in sorted({r["scores"]["family"] for r in prompts}):
        sub = [r for r in prompts if r["scores"]["family"] == fam]
        rows[f"fidelity/coverage/{fam}"] = row([r["scores"]["coverage"] for r in sub], seed)
        rows[f"responsiveness/latency_s/{fam}"] = row([r["scores"]["latency_s"] for r in sub], seed)

    # voice-v0 judge, chronology, topic
    rows |= judge_rows(es, prompts, by_kind["baited"], dataclasses.replace(deps, generate=judge("judge_prompts")), cfg)

    # voice
    rows |= voice_rows(es, prompts + by_kind["baited"], by_kind["drift"], deps, cfg)

    if cfg.size == "full":
        rows |= timed_rows(es, by_kind["timed"], judge("judge_timed"), seed)
        rows |= fdb_rows(by_kind["fdb"], judge("judge_fdb"), seed)
        onset = [float(spurious_onset(r["words"])) for r in by_kind["onset"]]
        greet = [float(bool(words_in(r["words"], 0.0, ONSET_GRACE_S))) for r in by_kind["onset"]]
        rows["responsiveness/spurious_onset"] = rate_row(int(sum(onset)), len(onset))
        rows["responsiveness/greeting"] = row(greet, seed)
    rows["naturalness/pref"] = row([])
    return rows, recs


def judge_rows(es: EvalSet, prompts: list[dict], baited: list[dict], deps: Deps, cfg: EvalConfig) -> dict:
    from shared import judge as J

    rows = {}
    rubric = J.load_rubric(RUBRIC)
    tasks = [J.Task(task_id=r["id"], origin=r["scores"]["family"], kind=r["scores"]["family"],
                    question=es.item(r["id"])["question"], context=passages_block(es.sources(r["id"])))
             for r in prompts]
    if tasks:
        outcomes, summary, _ = J.judge([r["scores"]["said"] or "(silence)" for r in prompts], tasks,
                                       [es.item(r["id"])["response_text"] for r in prompts], deps.generate, rubric,
                                       seed=cfg.seed)
        for r, o in zip(prompts, outcomes, strict=True):
            r["scores"] |= {"judge_non_loss": float(o.winner != "y"), "judge_content_error": float(o.x_error),
                            "judge_parsed": o.parsed}
        rows["fidelity/judge"] = row([r["scores"]["judge_non_loss"] for r in prompts], cfg.seed)
        rows["fidelity/judge_content_error"] = row([r["scores"]["judge_content_error"] for r in prompts], cfg.seed)
        rows["fidelity/judge_inconsistency"] = {"value": summary["all"].get("inconsistency"), "ci_low": None,
                                                "ci_high": None, "n": summary["all"]["n"]}
        for fam in sorted({r["scores"]["family"] for r in prompts}):
            rows[f"fidelity/judge/{fam}"] = row([r["scores"]["judge_non_loss"] for r in prompts
                                                if r["scores"]["family"] == fam], cfg.seed)
    change = [r for r in prompts if r["scores"]["family"] == "change"]
    verdicts = ask(deps.generate, [CHRONO_PROMPT.format(passages=passages_block(es.sources(r["id"])),
                                                        said=r["scores"]["said"] or "(silence)") for r in change],
                   CHRONO_SCHEMA, cfg.seed)
    for r, v in zip(change, verdicts, strict=True):
        r["scores"]["chronology"] = None if v is None else float(bool(v.get("both_retold") and v.get("in_order")))
    rows["fidelity/chronology"] = row([r["scores"].get("chronology") for r in change], cfg.seed)
    spoken = [r for r in baited if r["scores"]["said"]]
    labels = P.classify([r["scores"]["said"] for r in spoken], deps.generate, seed=cfg.seed) if spoken else []
    for r, (topic, borderline) in zip(spoken, labels, strict=True):
        r["scores"] |= {"topic": topic, "borderline": borderline}
    rows["fidelity/topic_violations"] = count_row(sum(r["scores"].get("topic", "other") != "other" for r in baited),
                                                  len(baited))
    return rows


def voice_rows(es: EvalSet, answered: list[dict], drift: list[dict], deps: Deps, cfg: EvalConfig) -> dict:
    """Speaker similarity, UTMOS, word confidence, dictionary share, WER, drift."""
    rows = {}
    refs = [read_wav(str(p)) for p in es.eval_clips]
    ref = deps.embed([w for w, _ in refs], refs[0][1]).mean(axis=0)
    ref /= np.linalg.norm(ref)
    segs, keep = [], []
    for r in answered:
        wav, sr = read_wav(r["agent_wav"])
        seg = speech_audio(wav, sr, r["words"], r["scores"]["user_end_s"], math.inf)
        if len(seg) >= MIN_SEGMENT_S * sr:
            segs.append(seg)
            keep.append(r)
    sr = keep[0]["sample_rate"] if keep else 24000
    if keep:
        sims = deps.embed(segs, sr) @ ref
        mos = deps.utmos(segs, sr)
        conf = deps.word_conf(segs, sr)
        for r, s, m, c in zip(keep, sims, mos, conf, strict=True):
            r["scores"] |= {"speaker_sim": float(s), "utmos": m, "word_conf": c}
    vocab = dictionary()
    for r in answered:
        r["scores"]["dict_share"] = dict_share(r["scores"]["said"], vocab)
        if cfg.backend == "personaplex":
            r["scores"]["wer"] = wer(r["text_stream"], r["transcript"])
    rows["voice/speaker_sim"] = row([r["scores"].get("speaker_sim") for r in answered], cfg.seed)
    rows["voice/utmos"] = row([r["scores"].get("utmos") for r in answered], cfg.seed)
    rows["voice/word_conf"] = row([r["scores"].get("word_conf") for r in answered], cfg.seed)
    rows["voice/dict_share"] = row([r["scores"].get("dict_share") for r in answered], cfg.seed)
    if cfg.backend == "personaplex":
        rows["voice/wer"] = row([r["scores"].get("wer") for r in answered], cfg.seed)
    if drift:
        firsts, lasts, deltas = [], [], []
        for r in drift:
            wav, sr = read_wav(r["agent_wav"])
            w = cfg.drift_window_s
            a = speech_audio(wav, sr, r["words"], 0.0, w)
            b = speech_audio(wav, sr, r["words"], cfg.drift_s - w, cfg.drift_s)
            if min(len(a), len(b)) < MIN_SEGMENT_S * sr:
                r["scores"]["drift"] = None
                continue
            s_a, s_b = (deps.embed([a, b], sr) @ ref).tolist()
            r["scores"] |= {"sim_first": s_a, "sim_last": s_b, "drift": s_b - s_a}
            firsts.append(s_a)
            lasts.append(s_b)
            deltas.append(s_b - s_a)
        rows["voice/drift"] = row(deltas, cfg.seed)
        rows["voice/sim_first_min"] = row(firsts, cfg.seed)
        rows["voice/sim_last_min"] = row(lasts, cfg.seed)
    return rows


def timed_rows(es: EvalSet, tracks: list[dict], generate: Callable, seed: int = 0) -> dict:
    """CueSpeak-style timed test: per cue, judge (meaning) AND timing; macro over cue types."""
    cases = []
    for r in tracks:
        cues = es.tracks[r["id"]]["cues"]
        bounds = [c["start_s"] for c in cues[1:]] + [es.tracks[r["id"]]["duration_s"]]
        main = cues[0]["prompt_id"]
        for cue, next_s in zip(cues, bounds):
            if cue["type"] not in CUE_TYPES:
                continue
            pid = cue.get("prompt_id", main)
            timing = cue_timing(cue, r["words"], r["transport_s"] or 0.0, next_s)
            said = text_of([w for w in r["words"] if cue["end_s"] <= w["start_s"] < next_s]
                           if cue["type"] != "backchannel" else words_in(r["words"], cue["start_s"], next_s))
            expected = passages_block(es.sources(pid)) + "\n\nReference answer: " + es.item(pid)["response_text"]
            cases.append((r, cue["type"], timing, said, expected))
    verdicts = ask(generate, [CUE_PROMPT.format(event=CUE_EVENTS[t], expected=e, said=s or "(silence)")
                                   for _, t, _, s, e in cases], CUE_SCHEMA, seed)
    per_type = defaultdict(list)
    for (r, t, timing, said, _), v in zip(cases, verdicts, strict=True):
        conveys = bool(v and v.get("conveys")) and bool(said)
        ok = float(conveys and timing["onset_ok"] and timing["yield_ok"])
        r["scores"].setdefault("cues", []).append({"type": t, "conveys": conveys, **timing, "pass": ok})
        per_type[t].append(ok)
    rows = {"interaction/cuespeak": macro_row(per_type, seed)}
    rows |= {f"interaction/cuespeak/{t}": row(per_type[t], seed) for t in CUE_TYPES}
    return rows


def fdb_rows(items: list[dict], generate: Callable, seed: int = 0) -> dict:
    """FDB v1 turn-taking and interruption latency / take-over rate; FDB v1.5 behaviour rates."""
    rows, v15 = {}, []
    turn_lat, turn_tor, int_lat, int_tor = [], [], [], []
    for r in items:
        d, task = Path(r["meta"]["dir"]), r["meta"]["task"]
        if task == "candor_turn_taking":
            end = json.loads((d / "turn_taking.json").read_text())[-1]["timestamp"][1]
            lat = response_latency(r["words"], end)
            turn_lat.append(model_time(lat, r["transport_s"]))
            turn_tor.append(float(lat is not None))
        elif task == "synthetic_user_interruption":
            end = json.loads((d / "interrupt.json").read_text())[-1]["timestamp"][1]
            lat = response_latency(r["words"], end)
            int_lat.append(model_time(lat, r["transport_s"]))
            int_tor.append(float(lat is not None))
        else:
            m = json.loads((d / "metadata.json").read_text())
            start, end = m["timestamps"]
            overlap = m.get("current_turn_text") or m.get("backchannel_text") or m.get("background_text") or ""
            before = text_of([w for w in r["words"] if w["start_s"] < start])
            after = text_of([w for w in r["words"] if w["start_s"] >= end])
            v15.append((r, task, m.get("context_text", ""), overlap, before, after))
        r["scores"]["task"] = task
    rows["responsiveness/fdb1_turn_latency_s"] = row(turn_lat, seed)
    rows["responsiveness/fdb1_turn_tor"] = row(turn_tor, seed)
    rows["responsiveness/fdb1_interrupt_latency_s"] = row(int_lat, seed)
    rows["responsiveness/fdb1_interrupt_tor"] = row(int_tor, seed)
    asks = [i for i, (*_, after) in enumerate(v15) if after]
    verdicts = ask(generate, [BEHAVIOUR_PROMPT.format(event=FDB15_EVENTS[v15[i][1]], context=v15[i][2],
                                                           overlap=v15[i][3], before=v15[i][4] or "(silence)",
                                                           after=v15[i][5]) for i in asks], BEHAVIOUR_SCHEMA, seed)
    behaviour = dict.fromkeys(range(len(v15)), "silent")
    for i, v in zip(asks, verdicts, strict=True):
        behaviour[i] = (v or {}).get("behaviour", "other")
    per_task = defaultdict(list)
    for i, (r, task, *_) in enumerate(v15):
        responded = behaviour[i] == "respond"
        ok = responded if FDB_V15_EXPECT[task] == "respond" else (not responded and behaviour[i] != "other")
        r["scores"] |= {"behaviour": behaviour[i], "correct": float(ok)}
        per_task[task].append(float(ok))
    rows["interaction/fdb15"] = macro_row(per_task, seed)
    rows |= {f"interaction/fdb15/{t}": row(per_task[t], seed) for t in FDB_V15}
    return rows


# ---------------------------------------------------------------- cost, side-by-sides, naturalness pairs


def profile(recs: list[dict], batches: list[dict]) -> dict:
    """Per-kind mean and max of every per-conversation span, and per-item seconds of each batch stage."""
    items = defaultdict(lambda: defaultdict(list))
    for r in recs:
        for k, v in (r.get("spans") or {}).items():
            items[r["kind"]][k].append(v)
    stages = defaultdict(lambda: {"s": 0.0, "n": 0})
    for b in batches:
        stages[b["span"]]["s"] += b["s"]
        stages[b["span"]]["n"] += b["n"]
    return {"items": {kind: {k: {"mean": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)}
                             for k, v in spans.items()} for kind, spans in items.items()},
            "stages": {k: v | {"per_item_s": v["s"] / v["n"] if v["n"] else None} for k, v in stages.items()}}


def gpu_cost(recs: list[dict], concurrency: int) -> dict:
    """Billed endpoint time: every conversation's wall time (handshake included), plus
    one cold start and one idle scale-down window per container."""
    busy = sum(r["wall_s"] for r in recs)
    containers = max(1, min(concurrency, len(recs)))
    hours = (busy + containers * SCALEDOWN_S) / 3600
    return {"gpu_hours": hours, "usd": hours * H200_USD_PER_HOUR, "busy_s": busy, "containers": containers}


def default_side_by_side(es: EvalSet) -> list[str]:
    smoke = es.manifest["sizes"]["smoke"]["prompts"]
    return [p for p in smoke if es.item(p)["family"] in ("story", "change", "idea")][:5]


def naturalness_pairs(es: EvalSet, n: int = 30, seed: int = 0) -> list[str]:
    """The 30 held-out prompt ids every system renders for the A/B naturalness test (fixed by seed)."""
    pool = sorted(p for p, r in es.prompts.items() if r["family"] != "author")
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


# ---------------------------------------------------------------- W&B


SUMMARY = ("fidelity/coverage", "fidelity/judge", "interaction/fdb15", "responsiveness/latency_s",
           "voice/speaker_sim", "naturalness/pref")


def axis_tables(rows: dict) -> dict[str, list[list]]:
    tables = defaultdict(list)
    for name, r in sorted(rows.items()):
        axis, metric = name.split("/", 1)
        tables[axis].append([metric, r["value"], r["ci_low"], r["ci_high"], r["n"]])
    return dict(tables)


def log_wandb(cfg: EvalConfig, es: EvalSet, report: dict, recs: list[dict], out: Path) -> str | None:
    """One eval run per brief §4. Passage text goes only into tables of the private project."""
    import wandb

    from shared import tracking

    tag = "check" if cfg.check else "baseline" if cfg.system == "stock" else "candidate"
    with tracking.start_run("voice", cfg.group, report["config"], name=cfg.run_name, job_type="eval", gpu="CPU",
                            tags=[tag, cfg.backend]) as run:
        if not cfg.dry_run:
            run.run.use_artifact(f"conle-ai/voice/{cfg.evalset}" if "/" not in cfg.evalset else cfg.evalset)
        cols = ["metric", "value", "ci_low", "ci_high", "n"]
        run.log({f"tables/{axis}": wandb.Table(columns=cols, data=data) for axis, data in axis_tables(report["rows"]).items()})
        run.summary.update({k: report["rows"][k]["value"] for k in report["rows"]})
        run.summary.update({f"{k}_ci": [report["rows"][k]["ci_low"], report["rows"][k]["ci_high"]]
                            for k in report["rows"]})
        run.summary.update({f"cost/{k}": v for k, v in report["cost"].items()})
        by_id = {r["id"]: r for r in recs}
        sbs = [p for p in report["config"]["side_by_side"] if p in by_id]
        rows = []
        for pid in sbs:
            r, p = by_id[pid], es.item(pid)
            words = " ".join(f"[{w['start_s']:.1f}] {w['w']}" for w in r["words"])
            rows.append([pid, p["family"], p["question"], wandb.Audio(r["agent_wav"], sample_rate=r["sample_rate"]),
                         words, p["response_text"], passages_block(es.sources(pid)), r["scores"].get("latency_s")])
        run.log({"side_by_side": wandb.Table(columns=["id", "family", "question", "agent", "transcript", "reference",
                                                      "source", "latency_s"], data=rows)})
        prof = report.get("profile", {})
        run.log({"profile/items": wandb.Table(columns=["kind", "span", "mean_s", "max_s", "n"], data=[
            [kind, k, v["mean"], v["max"], v["n"]] for kind, spans in prof.get("items", {}).items() for k, v in spans.items()]),
            "profile/stages": wandb.Table(columns=["stage", "s", "n", "per_item_s"], data=[
                [k, v["s"], v["n"], v["per_item_s"]] for k, v in prof.get("stages", {}).items()])})
        item_cols = ["id", "kind", "transcript", "transport_s", "scores"]
        run.log({"items": wandb.Table(columns=item_cols, data=[[r["id"], r["kind"], r["transcript"], r["transport_s"],
                                                                 json.dumps(r["scores"])] for r in recs])})
        run.run.save(str(out / "report.json"), base_path=str(out), policy="now")
        return run.url


# ---------------------------------------------------------------- driver


def evaluate(cfg: EvalConfig, es: EvalSet, deps: Deps) -> dict:
    """Converse, transcribe, score; writes <out>/report.json and returns the report."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    jobs = plan(es, cfg, out / "inputs")
    recs = converse_all(jobs, deps, out, cfg.concurrency)
    talk_s = time.monotonic() - t0
    prof: list[dict] = []
    rows, recs = score(es, recs, deps, cfg, prof)
    side_by_side = cfg.side_by_side or default_side_by_side(es)
    rubric_text = RUBRIC.read_text()
    import hashlib

    config = {
        "backend": cfg.backend, "system": cfg.system, "url": cfg.url, "size": cfg.size, "group": cfg.group,
        "evalset": f"voice-evalset:{es.version}" if es.version != "local" else cfg.evalset,
        "evalset_built_at": es.manifest.get("built_at"), "evalset_git_commit": es.manifest.get("git_commit"),
        "seeds": {"eval": cfg.seed, **es.manifest.get("seeds", {})}, "judge_model": cfg.judge_model,
        "rubric": RUBRIC.stem, "rubric_hash": hashlib.sha256(rubric_text.encode()).hexdigest()[:12],
        "phonon": deps.phonon_version, "speaker_embedder": SPEAKER_MODEL, "concurrency": cfg.concurrency,
        "subsample": {k: getattr(cfg, k) for k in ("tail_s", "timed_n", "fdb_per_task", "onset_runs", "onset_s",
                                                   "drift_runs", "drift_s")},
        "jobs": dict(sorted(Counter(j.kind for j in jobs).items())),
        "side_by_side": side_by_side, "naturalness_pairs": naturalness_pairs(es, seed=cfg.seed),
        "dry_run": cfg.dry_run,
    }
    report = {"config": config, "rows": rows, "cost": gpu_cost(recs, cfg.concurrency) | {"wall_s": talk_s},
              "profile": profile(recs, prof),
              "items": [{k: r[k] for k in ("id", "kind", "transport_s", "scores")} for r in recs]}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    return report


def fetch_evalset(ref: str, root: Path) -> EvalSet:
    """Download `voice-evalset:vN` from W&B (project voice) into root/<version>."""
    import wandb

    name = ref if "/" in ref else f"conle-ai/voice/{ref}"
    art = wandb.Api().artifact(name)
    path = art.download(root=str(root / art.version))
    return EvalSet(Path(path), art.version)


def _version(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------- real deps


ASR_APP = "voice-asr-conf"
NEMO_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


def _nemo_app():
    import modal

    app = modal.App(ASR_APP)
    image = (modal.Image.debian_slim(python_version="3.13").apt_install("ffmpeg", "libsndfile1")
             .pip_install("nemo_toolkit[asr]==2.4.0", "soundfile")
             .pip_install("ml_dtypes==0.5.4"))  # onnx (via nemo) needs float4_e2m1fn, absent from the resolved ml_dtypes
    volume = modal.Volume.from_name("voice-data", create_if_missing=True)

    @app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes={"/data": volume},
                  env={"HF_HOME": "/data/hf"}, serialized=True)
    def word_confidence(wavs: list[bytes]) -> list[float | None]:
        """Mean per-word confidence (NeMo entropy confidence, Tsallis, min-aggregated) per wav."""
        import io
        import os
        import tempfile

        import soundfile as sf
        from nemo.collections.asr.models import ASRModel
        from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig, ConfidenceMethodConfig
        from omegaconf import OmegaConf, open_dict

        model = ASRModel.from_pretrained(NEMO_MODEL, map_location="cpu").eval()
        conf = ConfidenceConfig(preserve_word_confidence=True, preserve_token_confidence=True, aggregation="min",
                                exclude_blank=True, method_cfg=ConfidenceMethodConfig(
                                    name="entropy", entropy_type="tsallis", alpha=0.5, entropy_norm="exp"))
        decoding = model.cfg.decoding  # the model's own TDT decoding (durations kept), plus confidence
        with open_dict(decoding):
            decoding.strategy = "greedy_batch"
            decoding.preserve_alignments = True
            decoding.confidence_cfg = OmegaConf.structured(conf)
        model.change_decoding_strategy(decoding)
        paths = []
        with tempfile.TemporaryDirectory() as d:
            for i, b in enumerate(wavs):
                wav, sr = sf.read(io.BytesIO(b), dtype="float32")
                p = os.path.join(d, f"{i}.wav")
                sf.write(p, wav, sr)
                paths.append(p)
            hyps = model.transcribe(paths, batch_size=8, return_hypotheses=True)
        hyps = hyps[0] if isinstance(hyps, tuple) else hyps
        out = []
        for h in hyps:
            wc = [float(c) for c in (getattr(h, "word_confidence", None) or [])]
            out.append(sum(wc) / len(wc) if wc else None)
        return out

    return app, word_confidence


def nemo_word_conf(wavs: list[np.ndarray], sample_rate: int) -> list[float | None]:
    """Run `word_confidence` on Modal (CPU) for wavs at `sample_rate` (resampled to 16 kHz there by NeMo's loader)."""
    import io

    import soundfile as sf
    from scipy.signal import resample_poly

    blobs = []
    for w in wavs:
        buf = io.BytesIO()
        sf.write(buf, resample_poly(w, 16000, sample_rate) if sample_rate != 16000 else w, 16000, format="WAV")
        blobs.append(buf.getvalue())
    app, fn = _nemo_app()
    with app.run():
        return fn.remote(blobs)


@functools.cache
def _utmos_model():
    import torch

    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()


def utmos(wavs: list[np.ndarray], sample_rate: int) -> list[float]:
    """UTMOS22 strong (SpeechMOS) per wav, on the CPU."""
    import torch

    model = _utmos_model()
    with torch.no_grad():
        return [float(model(torch.from_numpy(w).float()[None], sample_rate)) for w in wavs]


def real_deps(cfg: EvalConfig) -> Deps:
    from shared import teacher as T

    from voice.client import Client

    handle = T.teacher_cls(cfg.judge_model, max_inputs=64)

    def converse(job: Job, audio: Path):
        return Client(cfg.url, backend=cfg.backend, out_dir=audio).converse(job.wav, max_s=job.max_s, id=job.id)

    return Deps(converse=converse, transcribe=lambda wav: transcribe(wav),
                embed=lambda wavs, sr: speaker_embedding(wavs, sr),
                utmos=utmos if cfg.utmos else (lambda wavs, sr: [None] * len(wavs)), word_conf=nemo_word_conf,
                generate=T.modal_generate(handle), phonon_version=f"fermion-research {_version('fermion-research')}")


# ---------------------------------------------------------------- dry run


def fake_deps(es: EvalSet, latency_s: float = 0.5) -> Deps:
    """Offline stand-ins: the 'endpoint' answers each prompt with its reference response
    (a tone, one word per 0.3 s) `latency_s` after the user's last word, and stays
    quiet on silence; the transcript is exactly what it 'said'; judges answer in-schema."""
    import soundfile as sf

    from voice.client import Conversation

    said: dict[str, list[dict]] = {}

    def converse(job: Job, audio: Path):
        sr = 24000
        audio.mkdir(parents=True, exist_ok=True)
        agent = np.zeros(int(job.max_s * sr), dtype=np.float32)
        words = []
        if job.kind in ("prompt", "baited"):
            p = es.item(job.id)
            t = user_end(p) + latency_s
            for w in p["response_text"].split():
                words.append({"w": w, "start_s": round(t, 3), "end_s": round(t + 0.25, 3), "conf": None})
                t += 0.3
        elif job.kind in ("timed", "drift", "fdb"):
            t = 3.0
            while t + 0.3 < job.max_s:
                words.append({"w": "garden", "start_s": t, "end_s": t + 0.25, "conf": None})
                t += 0.3
        for w in words:
            i, j = int(w["start_s"] * sr), int(w["end_s"] * sr)
            agent[i:j] = 0.1 * np.sin(2 * np.pi * 150 * np.arange(j - i) / sr)
        path = audio / f"{job.id}.agent.wav"
        sf.write(path, agent, sr)
        said[str(path)] = words
        return Conversation(id=job.id, user_wav=job.wav, agent_wav=str(path), sample_rate=sr, max_s=job.max_s,
                            response_text=" ".join(w["w"] for w in words), transport_s=0.0,
                            transport={"send_hold_s": 0.0, "server_decode_hold_s": 0.0, "client_decode_hold_s": 0.0})

    class C:
        def __init__(self, text):
            self.text = text

    def generate(batch, json_schema=None, **kw):
        reply = {id(CUE_SCHEMA): {"conveys": True, "reason": "dry run"},
                 id(BEHAVIOUR_SCHEMA): {"behaviour": "resume", "reason": "dry run"},
                 id(CHRONO_SCHEMA): {"both_retold": True, "in_order": True, "reason": "dry run"}}
        if json_schema == P.LABEL_SCHEMA:
            return [C(json.dumps({"topic": "other", "borderline": False})) for _ in batch]
        if id(json_schema) in reply:
            return [C(json.dumps(reply[id(json_schema)])) for _ in batch]
        return [C(json.dumps({"winner": "tie", "factual_errors": "none", "reason": "dry run"})) for _ in batch]

    def embed(wavs, sr):
        v = np.stack([np.array([np.abs(w).mean() + 1e-3, w.std() + 1e-3, 1.0]) for w in wavs])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    return Deps(converse=converse, transcribe=lambda wav: {"text": text_of(said.get(str(wav), [])),
                                                           "words": said.get(str(wav), [])},
                embed=embed, utmos=lambda wavs, sr: [3.0] * len(wavs), word_conf=lambda wavs, sr: [0.9] * len(wavs),
                generate=generate, phonon_version="dry-run")


def dry_run_evalset(root: Path) -> EvalSet:
    """The synthetic eval set of `python -m voice.evalset --dry-run`, plus FDB items with
    valid audio and metadata for the tasks scored here."""
    import soundfile as sf

    from voice import evalset

    fx = evalset.fixture(root)
    evalset.build(root / "out", evalset.fake_deps(), fx["journal"], fx["voice"], fx["questions"], fx["fdb"],
                  stages=("voice", "passages", "prompts", "render", "timed", "fdb", "manifest"), dry_run=True)
    art = root / "out/artifact"
    meta = {"candor_turn_taking": ("turn_taking.json", [{"text": "x", "timestamp": [3.0, 3.3]}]),
            "synthetic_user_interruption": ("interrupt.json", [{"context": "x", "interrupt": "y",
                                                                 "timestamp": [2.0, 3.0]}])}
    for version, tasks in (("v1", FDB_V1), ("v1.5", FDB_V15)):
        for task in tasks:
            for d in (art / "fdb" / version / task).iterdir():
                sf.write(d / "input.wav", np.zeros(16000 * 6, dtype=np.float32), 16000)
                name, obj = meta.get(task, ("metadata.json", {"context_text": "x", "current_turn_text": "y",
                                                              "timestamps": [2.0, 3.0]}))
                (d / name).write_text(json.dumps(obj))
    return EvalSet(art, "dry-run")
