"""The question-generator eval as one process (cli.tasks_qgen_eval): probe the
Teacher, seed, synthesize, top up short kinds, make the answer pairs, and stop
the Teacher whatever happens. Every step is injected, so tests use fakes.

The phases run strictly one after another and must stay that way: the probe,
seed and synthesis are with-context calls, and a single with_context=False
request (the pairs step's no-journal answers, the chat classifier) in the same
vLLM batch turns cascade attention off for the whole batch.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence

from cartridges.cartridge.tasks.types import Kind

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 35 * 60  # capacity queue (22 min last time) + cold start
PROBE_LOG_EVERY_S = 60


def probe(spawn: Callable[[], object], timeout_s: float, log_every_s: float):
    """Spawn one Teacher call and wait for its result, logging every
    log_every_s. On the deadline (TimeoutError) or any other error the call is
    cancelled, so no queued input is left to start a container later."""
    call = spawn()
    logger.info("teacher probe: call %s (cancel by hand: modal.FunctionCall.from_id(id).cancel())",
                getattr(call, "object_id", "?"))
    start = time.monotonic()
    try:
        while True:
            left = timeout_s - (time.monotonic() - start)
            try:
                result = call.get(timeout=max(0.0, min(log_every_s, left)))
                break
            except TimeoutError:
                waited = time.monotonic() - start
                if waited >= timeout_s:
                    raise TimeoutError(f"teacher probe: no answer after {waited:.0f} s") from None
                logger.info("teacher probe: waiting, %.0f s", waited)
    except BaseException:
        try:
            call.cancel()
        except BaseException as e:  # e.g. a second Ctrl-C: the input stays queued
            logger.error("teacher probe: cancel of call %s failed (%r); cancel it by id before stopping containers",
                         getattr(call, "object_id", "?"), e)
        raise
    logger.info("teacher probe: answered after %.0f s", time.monotonic() - start)
    return result


def topup_mix(shortfall: Mapping[Kind, int]) -> dict[Kind, float] | None:
    """The short kinds, weighted by how short they are; None if none is."""
    total = sum(shortfall.values())
    return {k: v / total for k, v in shortfall.items() if v} if total else None


def qgen_eval(
    *,
    spawn_probe: Callable[[], object],
    seed: Callable[[], dict] | None,
    synthesize: Callable[[int, int, Mapping[Kind, float] | None], dict],
    shortfall: Callable[[Sequence[str]], Mapping[Kind, int]],
    pairs: Callable[[Sequence[str]], dict],
    stop: Callable[[], None],
    classify: Callable[[], dict] | None = None,
    run_ids: Sequence[str] = (),
    n: int = 150,
    topup_n: int = 50,
    probe_timeout_s: float = PROBE_TIMEOUT_S,
    probe_log_every_s: float = PROBE_LOG_EVERY_S,
) -> dict:
    """Run the steps in order, each to completion before the next, and return
    their reports: probe, classify (no-context, optional), seed (optional),
    synthesize n (skipped if n=0), one top-up of topup_n tasks (seed 1) for the
    kinds short over `run_ids` plus the new run, pairs. spawn_probe() starts one
    Teacher call (a modal FunctionCall: .get(timeout), .cancel()).
    synthesize(n, seed, kind_mix) returns a report with "run_id". stop() (the
    session's end; it must not stop other sessions' containers) runs
    even if a step fails."""
    report: dict = {}
    run_ids = list(run_ids)

    def done(step: str, value) -> None:
        report[step] = value
        logger.info("%s: %s", step, json.dumps(value, default=str))

    try:
        done("probe", probe(spawn_probe, probe_timeout_s, probe_log_every_s))
        if classify:
            done("classify", classify())
        if seed:
            done("seed", seed())
        report["synthesize"] = []
        if n:
            report["synthesize"].append(synthesize(n, 0, None))
            run_ids.append(report["synthesize"][-1]["run_id"])
            done("synthesize", report["synthesize"])
        short = shortfall(run_ids)
        done("shortfall", dict(short))
        if mix := topup_mix(short):
            report["synthesize"].append(synthesize(topup_n, 1, mix))
            run_ids.append(report["synthesize"][-1]["run_id"])
            done("topup", report["synthesize"][-1])
        done("pairs", pairs(run_ids))
    finally:
        stop()
        logger.info("session ended")
    return report
