"""CartridgeServer: base model + cartridge as a deployed Modal class, for chat.

`ask.py` answers one question per GPU job (load the model, answer, exit); a
chat needs the model to stay loaded between turns, so this is a class with the
cartridge and base model loaded in `@enter`, one container per cartridge
(the `cartridge` parameter is the container pool's identity), idle 5 minutes
before scaling down.

The prompt is `teacher.render(tokenizer, messages)[1]`: the student's
prompt_ids, the chat-templated conversation without the context block, which
the cartridge stands in for. Decoding is `Cartridge.generate` (HF forward,
batch 1): greedy unless a temperature is given, then nucleus sampling with the
teacher's top_p.

    uv run modal deploy -m cartridges.cartridge.serve_cartridge
    out = cartridge_cls("/vol/cartridges/exp-20261013-smoke/grpo").generate.remote(messages, max_tokens=128)
"""

# No `from __future__ import annotations`: Modal reads the modal.parameter
# annotations at class creation and needs real types.
import time

import modal

from cartridges.cartridge.train_distill import VOLUME_PATH, image, secrets, volume

MINUTES = 60
APP_NAME = "cartridges-cartridge-serve"
SMOKE_CARTRIDGE = f"{VOLUME_PATH}/cartridges/exp-20261013-smoke/grpo"
# L4 (24 GB): the 4B base in bf16 is ~8 GB, a p=2048 cartridge ~0.2 GB, and
# batch-1 decoding of a few thousand tokens needs little KV. Bigger bases need
# with_options(gpu=...).
GPU = "L4"

app = modal.App(APP_NAME, image=image)


def finish(new_ids: list[int], eos_ids: list[int]) -> tuple[list[int], str]:
    """(completion ids, finish_reason) from Cartridge.generate's row for one
    prompt, which pads with EOS after the first EOS: the tokens before the first
    EOS and "stop", or every token and "length" when none was generated."""
    for i, token in enumerate(new_ids):
        if token in eos_ids:
            return new_ids[:i], "stop"
    return list(new_ids), "length"


@app.cls(
    gpu=GPU,
    max_containers=1,  # one container per cartridge pool; a second would serve nothing while the first starts
    scaledown_window=5 * MINUTES,
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
    timeout=30 * MINUTES,  # per call
    startup_timeout=20 * MINUTES,
    retries=0,  # a failed call fails; callers bound waits with .get(timeout=...)
)
class CartridgeServer:
    """Chat generation with a cartridge on the volume (a path under /vol/cartridges)."""

    cartridge: str = modal.parameter(default=SMOKE_CARTRIDGE)

    @modal.enter()
    def start(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from cartridges.cartridge import tracking
        from cartridges.cartridge.cache import Cartridge

        volume.reload()
        self.cart = Cartridge.load(self.cartridge, device="cuda")
        model_id = self.cart.meta["base_model"]
        # One eval run per container in the cartridge's experiment group, like
        # ask.py, so the serving GPU's cost shows next to the training cost.
        group = self.cart.meta.get("group") or tracking.group_name("cartridge-serve")
        self.run = tracking.init_run(
            "eval", model_id, group, config={"cartridge": self.cartridge, **self.cart.meta},
            p=self.cart.p, gpu=GPU, role="cartridge-serve",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.cart.dtype_, device_map="cuda").eval()
        eos = self.tokenizer.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
        self.run.log_gpu_memory()
        self.torch = torch

    @modal.method()
    def generate(self, messages: list[dict], max_tokens: int = 512, temperature: float | None = None) -> dict:
        """One answer to the conversation (which ends with a user turn):
        {text, finish_reason, prompt_tokens, completion_tokens}. Greedy when
        temperature is None or 0; otherwise nucleus sampling at that temperature
        with the teacher's top_p. prompt_tokens counts the conversation only,
        not the cartridge."""
        from cartridges.cartridge.evals import strip_think
        from cartridges.cartridge.teacher import SAMPLING, render

        _, prompt_ids = render(self.tokenizer, messages)
        prompt = self.torch.tensor([prompt_ids], device="cuda")
        t0 = time.perf_counter()
        new = self.cart.generate(
            self.model, prompt, max_new_tokens=max_tokens, temperature=temperature or 0.0,
            top_p=SAMPLING["top_p"] if temperature else 1.0, eos_token_id=self.eos_ids or None,
        )
        latency = time.perf_counter() - t0
        ids, finish_reason = finish(new[0].tolist(), self.eos_ids)
        self.run.log({"gen/latency_s": latency, "gen/prompt_tokens": len(prompt_ids), "gen/output_tokens": len(ids)})
        return {
            "text": strip_think(self.tokenizer.decode(ids, skip_special_tokens=True)),
            "finish_reason": finish_reason,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(ids),
        }

    @modal.exit()
    def stop(self):
        self.run.finish()


def cartridge_cls(cartridge: str = SMOKE_CARTRIDGE, **options):
    """A handle on the deployed CartridgeServer for `cartridge` (one container
    pool per path); `options` go to with_options (e.g. gpu for a bigger base)."""
    cls = modal.Cls.from_name(APP_NAME, "CartridgeServer")
    return (cls.with_options(**options) if options else cls)(cartridge=cartridge)
