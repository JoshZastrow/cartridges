"""backends/registry.py: register, get, the discriminated config."""

import pytest
from voice.backends.fake import FakeBackend
from voice.backends.registry import FakeConfig, HertzConfig, PersonaPlexConfig, parse_config, register

from voice import backends


def test_get_builds_registered_backend_from_model_dict_or_defaults():
    assert isinstance(backends.get("fake"), FakeBackend)
    assert backends.get("fake", {"p": 6}).config.p == 6
    assert backends.get("fake", FakeConfig(seed=3)).config.seed == 3


def test_get_rejects_unknown_and_mismatched():
    with pytest.raises(KeyError, match="unknown backend"):
        backends.get("nope")
    with pytest.raises(ValueError, match="not 'fake'"):
        backends.get("fake", PersonaPlexConfig(checkpoint="nvidia/personaplex-7b-v1"))


def test_config_is_discriminated_on_name():
    pp = parse_config({"name": "personaplex", "checkpoint": "nvidia/personaplex-7b-v1"})
    h = parse_config({"name": "hertz", "checkpoint": "/vol/x.pt"})
    assert isinstance(pp, PersonaPlexConfig) and pp.p == 1024 and isinstance(h, HertzConfig) and h.p == 512
    with pytest.raises(ValueError):
        parse_config({"name": "personaplex"})  # checkpoint is required
    with pytest.raises(ValueError):
        parse_config({"name": "moshi"})


def test_register_refuses_a_second_class_for_a_name():
    with pytest.raises(ValueError, match="already registered"):
        register("fake")(type("Other", (), {}))
