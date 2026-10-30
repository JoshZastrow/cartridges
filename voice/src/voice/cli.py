"""`voice` command line: encode | train | eval | report | deploy | serve-local, each with --backend; evalset.

    uv run voice train --config sft.toml [--dry-run]
    uv run voice deploy --backend personaplex|hertz --system stock  # Modal app voice-serve; prints the WebSocket URL
    uv run voice serve-local --backend personaplex [--port 8998] # the same server on this machine's GPU
    uv run voice evalset [--dry-run] [--stage ...]               # build voice-evalset (python -m voice.evalset)
    uv run voice eval --url <wss://...> --backend <name> --evalset voice-evalset:v1 --size smoke|full --group exp-...
    uv run voice report --group exp-... [--stage eval-full]      # table, frontier plots, side-by-sides from W&B

`train` (#19), `deploy` and `serve-local` (#20; hertz deploy #21), `evalset` (#22), `eval` and `report` (#9) are
implemented; the rest are stubs naming the issue that ships them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from pathlib import Path

from voice.backends.registry import NAMES

STUBS = {
    "encode": "#10 (data) and #8/#18 (backend encoders)",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voice", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("encode", "train", "eval", "report", "deploy", "serve-local"):
        s = sub.add_parser(name)
        if name == "report":
            s.add_argument("--group", required=True, help="exp-YYYYMMDD-<slug>")
            s.add_argument("--stage", default="eval-full")
            s.add_argument("--out", type=Path, default=None)
            continue
        s.add_argument("--backend", choices=NAMES, default=None, required=name in ("deploy", "serve-local", "eval"))
        if name == "train":
            s.add_argument("--config", help="TOML file of voice.train.TrainConfig")
            s.add_argument("--dry-run", action="store_true", help="CPU, FakeBackend, tiny fixture, W&B disabled")
        if name == "deploy":
            s.add_argument("--system", default="stock", help="stock, or a cartridge artifact (#12)")
            s.add_argument("--dry-run", action="store_true", help="no Modal calls; print the URL shape")
        if name == "serve-local":
            s.add_argument("--port", type=int, default=8998)
        if name == "eval":
            s.add_argument("--url", help="the endpoint's WebSocket URL (names the system; required unless --dry-run)")
            s.add_argument("--evalset", default="voice-evalset:v1")
            s.add_argument("--size", choices=("smoke", "full"), default="smoke")
            s.add_argument("--group", required=True, help="exp-YYYYMMDD-<slug>")
            s.add_argument("--system", default="stock", help="stock, or the cartridge artifact being served")
            s.add_argument("--out", type=Path, default=None, help="default voice/outputs/eval/<group>/<backend>-<system>-<size>")
            s.add_argument("--concurrency", type=int, default=1, help="conversations at once (voice-serve containers)")
            s.add_argument("--side-by-side", nargs="*", default=[], help="prompt ids for the side-by-sides")
            for knob in ("timed_n", "fdb_per_task", "onset_runs", "drift_runs"):
                s.add_argument("--" + knob.replace("_", "-"), type=int, default=None)
            s.add_argument("--dry-run", action="store_true", help="synthetic eval set, fake endpoint and models, W&B off")
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


def eval_command(args) -> int:
    import os
    import tempfile

    from voice import evals

    out = args.out or Path("voice/outputs/eval") / args.group / f"{args.backend}-{args.system}-{args.size}"
    knobs = {k: getattr(args, k) for k in ("timed_n", "fdb_per_task", "onset_runs", "drift_runs")
             if getattr(args, k) is not None}
    if args.dry_run:
        os.environ["WANDB_MODE"] = "disabled"
        root = Path(tempfile.mkdtemp(prefix="voice-eval-dry-"))
        es = evals.dry_run_evalset(root)
        cfg = evals.EvalConfig(url=args.url or "ws://dry-run", backend=args.backend, evalset=args.evalset,
                               size=args.size, group=args.group, system=args.system, out=root / "eval",
                               concurrency=args.concurrency, side_by_side=args.side_by_side, dry_run=True,
                               **({"timed_n": 3, "fdb_per_task": 2, "onset_runs": 2, "drift_runs": 1} | knobs))
        report = evals.evaluate(cfg, es, evals.fake_deps(es))
    else:
        if not args.url:
            print("voice eval: --url is required (or --dry-run)", file=sys.stderr)
            return 2
        from shared import teacher

        cfg = evals.EvalConfig(url=args.url, backend=args.backend, evalset=args.evalset, size=args.size,
                               group=args.group, system=args.system, out=out, concurrency=args.concurrency,
                               side_by_side=args.side_by_side, **knobs)
        es = evals.fetch_evalset(args.evalset, Path("voice/outputs/evalsets"))
        try:
            report = evals.evaluate(cfg, es, evals.real_deps(cfg))
        finally:
            teacher.stop()
    recs = [json.loads(line) for line in (cfg.out / "items.jsonl").read_text().splitlines()]
    scores = {i["id"]: i["scores"] for i in report["items"]}
    print(evals.log_wandb(cfg, es, report, [r | {"scores": scores[r["id"]]} for r in recs], cfg.out))
    for name, r in sorted(report["rows"].items()):
        print(f"{name:45s} {r['value']!s:>22} [{r['ci_low']}, {r['ci_high']}] n={r['n']}")
    print(f"cost: {report['cost']}  report: {cfg.out / 'report.json'}")
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
    if args.command == "eval":
        return eval_command(args)
    if args.command == "report":
        from voice import report

        return report.main(args.group, args.stage, args.out)
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
