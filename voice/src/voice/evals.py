"""Evaluation measures (brief §8). #9 owns this module; #22 ships `transcribe`
and the speaker embedder used for the Pocket TTS ceiling.

    transcribe(wav) -> {text, words: [{w, start_s, end_s, conf}]}
    speaker_embedding(wavs, sample_rate) -> [N, 256] unit vectors (WavLM-TDNN, voice.speaker)
    speaker_similarity(wavs_a, wavs_b, sample_rate) -> [len(wavs_a)] cosine to the centroid of wavs_b

Every transcript and word timestamp in the epic comes from Phonon-2
(`FermionResearch/Phonon-2`) through `transcribe`: the `phonon transcribe
--json` CLI of the `fermion-research` package (MLX on Apple silicon, its CPU
engine elsewhere). `wav` is a path, or a float array at `sample_rate` that is
written to a temporary file first. The packaged CLI (0.2.15) emits no
per-word confidence, so `conf` is None until #9 adds it (brief §4).
"""

from __future__ import annotations

import functools
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from voice import speaker

PHONON = "phonon"
SPEAKER_MODEL = speaker.MODEL_ID  # recorded in evalset.json


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
