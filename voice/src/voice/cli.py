"""`voice` command line: encode | train | eval | report | deploy, each with --backend.

    uv run voice train --config sft.toml [--dry-run]

Only `train` is implemented here (#19); the rest are stubs naming the issue
that ships them.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib

from voice.backends.registry import NAMES

STUBS = {
    "encode": "#10 (data) and #8/#18 (backend encoders)",
    "eval": "#9",
    "report": "#9",
    "deploy": "#20/#21 (stock) and #12 (cartridge)",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voice", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("encode", "train", "eval", "report", "deploy"):
        s = sub.add_parser(name)
        s.add_argument("--backend", choices=NAMES, default=None)
        if name == "train":
            s.add_argument("--config", help="TOML file of voice.train.TrainConfig")
            s.add_argument("--dry-run", action="store_true", help="CPU, FakeBackend, tiny fixture, W&B disabled")
    return p


def train_command(args) -> int:
    from voice.train import TrainConfig, dry_run, launch

    data = {}
    if args.config:
        with open(args.config, "rb") as f:
            data = tomllib.load(f)
    if args.backend:
        data["backend"] = {**data.get("backend", {}), "name": args.backend}
    cfg = TrainConfig.model_validate(data)
    if args.dry_run:
        dry_run(cfg)
        return 0
    if cfg.backend.name == "fake":
        print("the fake backend only runs with --dry-run", file=sys.stderr)
        return 2
    print(launch(cfg))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parser().parse_args(argv)
    if args.command == "train":
        return train_command(args)
    print(f"voice {args.command}: not implemented yet; ships with {STUBS[args.command]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
