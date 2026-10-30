"""hertz-dev parity: the served load path against the saved torch 2.5.1 reference. One H200 container (~7 min, ~$0.5).

    uv run --package voice python voice/scripts/hertz_parity.py [--reference 2.5.1.pt]

Loads through `voice.deploy_hertz.loaded_backend()`, so it checks whatever
`HertzBackend.load()` applies (hertz_sdpa, the MD5 stamp, the fast build), on
the stock prompt `voice-voice:/prompt-30s.wav` (238 frames):

- `latents`: the prompt's codec posterior means, both channels;
- `logits_user` / `logits_agent`: the LM teacher-forced over all positions,
  no cache, both heads;
- `generated`: seeded generation, `manual_seed(0)`, temps (0.8, (0.4, 0.1)),
  32 frames through `next_latent`;
- the unsynced live step, 400 steps on silence, as the server runs it.

The reference `voice-data:/hertz-parity/<name>` holds the same procedure's
outputs from upstream's pinned torch 2.5.1+cu124 (`2.5.1.pt`; `2.14.1.pt` is
2.14 unpatched). Pass criteria are the bf16 floor measured on #21 (torch 2.14
+ hertz_sdpa: latents 0.031, logits 1 ulp at |176|, KL 0.022 / 0.023 nats,
argmax agreement 0.845 / 0.853, generation within 0.016 for frames 0-7 and
divergent from frame 8 as sampling forks at near-ties, live step 62.5 ms).
Exits 1 if any check fails.

Written by impl-21 on #21; committed on #27 so later load changes and #18 re-run it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import modal

from voice import deploy_hertz as dh

FRAMES = 32
TEMPS = (0.8, (0.4, 0.1))
EXACT_FRAMES = 8  # generated frames that stay at the bf16 floor before sampling forks
LIMITS = {"latents_max_abs_diff": 0.05, "logits_max_ulps": 1.0, "kl_mean": 0.03, "argmax_agree_min": 0.80,
          "generated_max_abs_diff": 0.02}

app = modal.App("voice-hertz-parity", image=dh.image)


@app.function(gpu=dh.GPU, volumes=dh.volumes, secrets=[modal.Secret.from_name("HF_TOKEN")], timeout=2400)
def check(reference: str) -> dict:
    import torch

    from voice.backends.hertz import CHUNK
    from voice.backends.hertz import TEMPS as LIVE_TEMPS

    b = dh.loaded_backend()
    m = b.model
    lat = torch.cat([b.latents(b.prompt[0]), b.latents(b.prompt[1])], -1)[None]
    with b._ctx():
        m.deinit_cache()
        l1, l2 = (x.float()[0].cpu() for x in m.forward(lat.to(torch.bfloat16)))
        m.init_cache(bsize=1, device="cuda", dtype=torch.bfloat16, length=2048)
        torch.manual_seed(0)
        nxt, gen = m.next_latent(lat.to(torch.bfloat16), TEMPS), []
        for _ in range(FRAMES):
            gen.append(nxt.float()[0, 0].cpu())
            nxt = m.next_latent(nxt, TEMPS)
        m.deinit_cache()
    ref = torch.load(Path(dh.DATA) / "hertz-parity" / dh.safe_name(reference), weights_only=False)  # our own file
    st = {"torch": str(torch.__version__), "reference": reference,
          "latents_max_abs_diff": float((ref["latents"] - lat[0].cpu()).abs().max())}
    for k, v in (("logits_user", l1), ("logits_agent", l2)):
        d = (ref[k] - v).abs()
        ulp = 2.0 ** (torch.floor(torch.log2(ref[k].abs().max())) - 7)  # bf16 spacing at the largest logit
        kl = torch.nn.functional.kl_div(v.log_softmax(-1), ref[k].log_softmax(-1), log_target=True, reduction="none")
        st[k] = {"max_abs_diff": float(d.max()), "max_ulps": float(d.max() / ulp), "mean_abs_diff": float(d.mean()),
                 "argmax_agree": float((ref[k].argmax(-1) == v.argmax(-1)).float().mean()),
                 "kl_mean": float(kl.sum(-1).mean())}
    g = (ref["generated"] - torch.stack(gen)).abs().amax(-1)
    st["generated_max_abs_diff_per_frame"] = [round(float(x), 3) for x in g]

    silence = torch.zeros(1, 1, CHUNK, device="cuda")
    with b._ctx():
        m.init_cache(bsize=1, device="cuda", dtype=torch.bfloat16, length=2048)
        nxt = m.next_audio_from_audio(b._tensor(b.prompt)[None], temps=LIVE_TEMPS)
        for _ in range(8):
            nxt = m.next_audio_from_audio(torch.cat([silence.to(nxt.dtype), nxt], 1), temps=LIVE_TEMPS)
        torch.cuda.synchronize()
        t0, n = time.perf_counter(), 400
        for _ in range(n):
            nxt = m.next_audio_from_audio(torch.cat([silence.to(nxt.dtype), nxt], 1), temps=LIVE_TEMPS)
        nxt.float().cpu()
        st["live_step_ms_unsynced"] = round(1000 * (time.perf_counter() - t0) / n, 1)
        m.deinit_cache()
    st["load_s"] = round(b.load_s, 1)
    st["max_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(st, flush=True)
    return st


def failures(st: dict) -> list[str]:
    out = []
    if st["latents_max_abs_diff"] > LIMITS["latents_max_abs_diff"]:
        out.append(f"latents {st['latents_max_abs_diff']:.3f}")
    for k in ("logits_user", "logits_agent"):
        s = st[k]
        if s["max_ulps"] > LIMITS["logits_max_ulps"]:
            out.append(f"{k} max diff {s['max_ulps']:.1f} bf16 ulps")
        if s["kl_mean"] > LIMITS["kl_mean"]:
            out.append(f"{k} KL {s['kl_mean']:.3f}")
        if s["argmax_agree"] < LIMITS["argmax_agree_min"]:
            out.append(f"{k} argmax agreement {s['argmax_agree']:.3f}")
    worst = max(st["generated_max_abs_diff_per_frame"][:EXACT_FRAMES])
    if worst > LIMITS["generated_max_abs_diff"]:
        out.append(f"generated frames 0-{EXACT_FRAMES - 1} differ by {worst:.3f}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reference", default="2.5.1.pt", help="file name under voice-data:/hertz-parity/")
    args = p.parse_args(argv)
    with modal.enable_output(), app.run():
        st = check.remote(args.reference)
    bad = failures(st)
    print(json.dumps({**st, "pass": not bad, "failures": bad}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
