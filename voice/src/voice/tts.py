"""Speech rendering with Kyutai Pocket TTS (brief §1b, §4).

    render_author(text, prompt=AUTHOR_PROMPT) -> np.ndarray [samples] float32 at 24 kHz
    render_other(text, voice_id)             -> np.ndarray [samples] float32 at 24 kHz

`render_author` clones the author's voice from the 30 s prompt segment of the
eval set (`voice-evalset:/voice-eval/prompt-30s.wav`, also on the
`voice-voice` volume); every issue renders agent-side speech through it so
every issue teaches the same voice. `render_other` speaks user-channel text
in one of Pocket TTS's catalogue voices (OTHER_VOICES: English voice
donations and VCTK speakers; the Expresso voice is left out for its
non-commercial license). The model and the voice states are loaded once per
process and cached.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
AUTHOR_PROMPT = Path(os.environ.get("VOICE_AUTHOR_PROMPT", "voice/outputs/evalset/artifact/voice-eval/prompt-30s.wav"))
OTHER_VOICES = ("alba", "marius", "javert", "jean", "anna", "vera", "fantine", "charles", "paul", "eponine",
                "azelma", "george", "mary", "jane", "michael", "eve")


@functools.cache
def model():
    from pocket_tts import TTSModel

    m = TTSModel.load_model()
    assert m.sample_rate == SAMPLE_RATE, m.sample_rate
    return m


@functools.cache
def _voice_state(prompt: str):
    return model().get_state_for_audio_prompt(prompt, truncate=True)


def _render(state, text: str) -> np.ndarray:
    return model().generate_audio(state, text, copy_state=True).numpy().astype(np.float32).reshape(-1)


def render_author(text: str, prompt: Path | str = AUTHOR_PROMPT) -> np.ndarray:
    """`text` in the author's voice, cloned from `prompt` (a wav path)."""
    return _render(_voice_state(str(prompt)), text)


def render_other(text: str, voice_id: str) -> np.ndarray:
    """`text` in a catalogue voice (one of OTHER_VOICES)."""
    if voice_id not in OTHER_VOICES:
        raise ValueError(f"unknown voice {voice_id!r}; one of {OTHER_VOICES}")
    return _render(_voice_state(voice_id), text)
