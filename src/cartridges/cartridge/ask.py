"""Answer a question with base model + cartridge, no timeline in context.

`uv run cartridges cartridge-ask "<question>" --cartridge <ref>` (cli.py) runs
`ask` on a Modal GPU. <ref> is one of:
- a local directory with cartridge.safetensors + cartridge.json (sent along),
- a path on the cartridges-cartridge-cache volume (e.g. /vol/cartridges/<group>),
- a W&B artifact (e.g. cartridge-<run_id>, or entity/project/name:alias).
The base model is the one recorded in cartridge.json. The prompt is
label.answer_messages (the citation reminder, then the question), the format the
teacher labels were generated with, so answers cite pages as [journal:p.N].
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from cartridges.cartridge.cache import CONFIG, WEIGHTS
from cartridges.cartridge.train_distill import VOLUME_PATH, image, secrets, volume

app = modal.App("cartridges-ask", image=image)
GPU = "A100-80GB"


def prompt_ids(tokenizer, question: str) -> list[int]:
    """The student's prompt: the shared task-answer format (label.answer_messages, the
    citation reminder then the question) that the teacher labels were generated
    with, rendered without the context block the cartridge stands in for."""
    from cartridges.cartridge.tasks.label import answer_messages
    from cartridges.cartridge.teacher import render

    return render(tokenizer, answer_messages(question))[1]


@app.function(gpu=GPU, timeout=20 * 60, volumes={VOLUME_PATH: volume}, secrets=secrets)
def ask(question: str, cartridge: str = "", files: dict[str, bytes] | None = None, max_new_tokens: int = 256) -> dict:
    """`files` (a local cartridge's two files) wins over `cartridge` (volume path or W&B ref)."""
    import tempfile

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cartridges.cartridge.cache import Cartridge
    from cartridges.cartridge.evals import parse_citations, strip_think
    from cartridges.cartridge import tracking

    if files:
        path = Path(tempfile.mkdtemp())
        for name, data in files.items():
            (path / name).write_bytes(data)
    elif cartridge.startswith(VOLUME_PATH):
        volume.reload()
        path = Path(cartridge)
    else:
        path = tracking.fetch_artifact(cartridge, type="cartridge")
    cart = Cartridge.load(path, device="cuda")
    model_id = cart.meta["base_model"]
    # One eval run per ask, in the cartridge's experiment group, so spend and lineage show in W&B.
    group = cart.meta.get("group") or tracking.group_name("ask")
    config = {"question": question, "cartridge": cartridge or "(local)", "max_new_tokens": max_new_tokens, **cart.meta}
    with tracking.init_run("eval", model_id, group, config=config, p=cart.p, gpu=GPU, role="ask") as run:
        if not files and not cartridge.startswith(VOLUME_PATH):
            run.use_artifact(cartridge, type="cartridge")  # lineage; already in the local W&B cache
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=cart.dtype_, device_map="cuda").eval()

        prompt = torch.tensor([prompt_ids(tok, question)], device="cuda")
        t0 = time.monotonic()
        new = cart.generate(model, prompt, max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id)
        seconds = time.monotonic() - t0
        answer = strip_think(tok.decode(new[0], skip_special_tokens=True))
        out = {
            "question": question,
            "answer": answer,
            "citations": parse_citations(answer),
            "cartridge": cartridge or "(local)",
            "base_model": model_id,
            "p": cart.p,
            "cartridge_mb": round(cart.nbytes() / 2**20, 1),
            "generate_s": round(seconds, 2),
            "new_tokens": new.shape[1],
        }
        run.summary.update(out)
        return {**out, "wandb_run": run.name, "cost_usd": round(run.cost()["cost/usd"], 4)}


def _local_files(cartridge: str) -> dict[str, bytes] | None:
    local = Path(cartridge)
    if local.is_dir() and (local / WEIGHTS).exists():
        return {name: (local / name).read_bytes() for name in (WEIGHTS, CONFIG)}
    return None


def run(question: str, cartridge: str, max_new_tokens: int = 256) -> dict:
    """Local side: ship a local cartridge's files, else pass the reference through."""
    files = _local_files(cartridge)
    with app.run():
        return ask.remote(question, "" if files else cartridge, files, max_new_tokens)


@app.local_entrypoint()
def main(question: str, cartridge: str, max_new_tokens: int = 256) -> None:
    files = _local_files(cartridge)
    print(json.dumps(ask.remote(question, "" if files else cartridge, files, max_new_tokens), indent=2))
