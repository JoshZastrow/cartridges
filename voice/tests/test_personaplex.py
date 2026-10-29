"""backends/personaplex.py on CPU: registration, spec, text stream layout. The
weights-backed methods are checked on Modal (#20 summary)."""

import pytest
from voice.backends.personaplex import EPAD, PAD, STREAMS, PersonaPlexBackend, chat_path, text_stream
from voice.backends.registry import PersonaPlexConfig

from voice import backends


def test_registered_with_17_streams_and_no_weights_loaded():
    b = backends.get("personaplex", {"checkpoint": "nvidia/personaplex-7b-v1"})
    assert isinstance(b, PersonaPlexBackend)
    assert [s.name for s in b.spec.streams] == STREAMS and len(STREAMS) == 17
    assert b.spec.streams[0].kind == "text" and {s.role for s in b.spec.streams[9:]} == {"user"}
    assert b.spec.sample_rate == 24000 and b.spec.frame_rate_hz == 12.5 and b.spec.cache_groups == ["temporal"]


def test_cartridge_half_is_left_to_issue_8():
    b = PersonaPlexBackend(PersonaPlexConfig(checkpoint="x"))
    for call in (lambda: b.build_prefix(None, "", ""), lambda: b.cartridge_from_prefill(None),
                 lambda: b.forward(None, None), lambda: b.sft_loss(None, None, None), lambda: b.distill_loss(None, None)):
        with pytest.raises(NotImplementedError, match="#8"):
            call()
    with pytest.raises(NotImplementedError, match="#12"):
        b.serve(object(), 8998)


def test_text_stream_places_tokens_at_word_starts_with_epad_before():
    # 12.5 Hz: 0.0 s -> frame 0, 0.4 s -> frame 5
    out = text_stream([([11, 12], 0.0, 0.3), ([21], 0.4, 0.6)], 9).tolist()
    assert out == [11, 12, PAD, PAD, EPAD, 21, PAD, PAD, PAD]


def test_text_stream_keeps_leftover_tokens_when_words_crowd():
    out = text_stream([([1, 2, 4], 0.0, 0.1), ([5], 0.08, 0.2)], 6).tolist()
    assert out == [1, 2, 4, 5, PAD, PAD]


def test_chat_path_carries_the_system():
    assert chat_path("NATM1.pt", "You are Al.") == "/api/chat?voice_prompt=NATM1.pt&text_prompt=You+are+Al."
