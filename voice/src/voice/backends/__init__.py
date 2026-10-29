"""Duplex backends behind one protocol; see base.py and brief section 4b.

    backend = backends.get(cfg.name, cfg)

Importing this package registers every backend module listed below.
"""

from voice.backends import fake, personaplex  # noqa: F401  (register "fake", "personaplex")
from voice.backends.registry import BackendConfig, get, parse_config, register, registered

__all__ = ["BackendConfig", "get", "parse_config", "register", "registered"]
