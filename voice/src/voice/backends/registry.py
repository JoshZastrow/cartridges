"""Backend factory: `@register(name)` on a backend class, `get(name, config)`
to build one. Nothing outside `backends/` imports a backend module directly.

`BackendConfig` is a union discriminated on `name`; each backend's config
subclasses `BackendConfigBase` with its own defaults. "fake" is the CPU test
double that `voice train --dry-run` and the tests use.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class BackendConfigBase(BaseModel):
    checkpoint: str  # HF id or volume path
    p: int = 1024  # prefix frames
    device: str = "cuda"


class PersonaPlexConfig(BackendConfigBase):
    name: Literal["personaplex"] = "personaplex"


class HertzConfig(BackendConfigBase):
    name: Literal["hertz"] = "hertz"
    p: int = 512  # 64 s at 8 Hz


class FakeConfig(BackendConfigBase):
    name: Literal["fake"] = "fake"
    checkpoint: str = ""  # no weights
    p: int = 4
    device: str = "cpu"
    vocab: int = 8
    dim: int = 16
    seed: int = 0


BackendConfig = Annotated[PersonaPlexConfig | HertzConfig | FakeConfig, Field(discriminator="name")]
NAMES = ("personaplex", "hertz", "fake")

_REGISTRY: dict[str, type] = {}


def parse_config(data: dict) -> BackendConfig:
    return TypeAdapter(BackendConfig).validate_python(data)


def register(name: str):
    """Class decorator: make `get(name, ...)` build this backend."""

    def wrap(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"backend {name!r} is already registered to {_REGISTRY[name].__name__}")
        _REGISTRY[name] = cls
        return cls

    return wrap


def get(name: str, config: BackendConfigBase | dict | None = None):
    """The backend registered as `name`, built from `config` (a config model, a
    dict for `parse_config`, or None for that backend's defaults)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(_REGISTRY)}")
    if config is None:
        config = {"name": name}
    if isinstance(config, dict):
        config = parse_config({"name": name, **config})
    if config.name != name:
        raise ValueError(f"config is for backend {config.name!r}, not {name!r}")
    return _REGISTRY[name](config)


def registered() -> list[str]:
    return sorted(_REGISTRY)
