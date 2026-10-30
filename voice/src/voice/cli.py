"""`voice` command line: encode | train | eval | report | deploy | serve-local, each with --backend; evalset.

    uv run voice train --config sft.toml [--dry-run]
    uv run voice deploy --backend personaplex --system stock     # Modal app voice-serve; prints the WebSocket URL
    uv run voice serve-local --backend personaplex [--port 8998] # the same server on this machine's GPU
    uv run voice evalset [--dry-run] [--stage ...]               # build voice-evalset (python -m voice.evalset)

`train` (#19), `deploy` and `serve-local` (#20), `evalset` (#22) are implemented; the rest are
stubs naming the issue that ships them.
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
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voice", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("encode", "train", "eval", "report", "deploy", "serve-local"):
        s = sub.add_parser(name)
        s.add_argument("--backend", choices=NAMES, default=None, required=name in ("deploy", "serve-local"))
        if name == "train":
            s.add_argument("--config", help="TOML file of voice.train.TrainConfig")
            s.add_argument("--dry-run", action="store_true", help="CPU, FakeBackend, tiny fixture, W&B disabled")
        if name == "deploy":
            s.add_argument("--system", default="stock", help="stock, or a cartridge artifact (#12)")
            s.add_argument("--dry-run", action="store_true", help="no Modal calls; print the URL shape")
        if name == "serve-local":
            s.add_argument("--port", type=int, default=8998)
    sub.add_parser("evalset", add_help=False, help="build the eval set; arguments as `python -m voice.evalset`")
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
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["evalset"]:
        from voice import evalset

        return evalset.main(argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parser().parse_args(argv)
    if args.command == "train":
        return train_command(args)
    if args.command in ("deploy", "serve-local"):
        from voice.deploy import FUNCTIONS, deploy, serve_local

        if args.backend not in FUNCTIONS:
            print(f"voice {args.command} --backend {args.backend}: not implemented yet; served: {sorted(FUNCTIONS)}",
                  file=sys.stderr)
            return 2
        if args.command == "serve-local":
            return serve_local(args.backend, args.port)
        try:
            print(deploy(args.backend, args.system, args.dry_run))
        except NotImplementedError as e:
            print(f"voice deploy: {e}", file=sys.stderr)
            return 2
        return 0
    print(f"voice {args.command}: not implemented yet; ships with {STUBS[args.command]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
