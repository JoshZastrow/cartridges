# voice

The duplex speech cartridge: a trained KV prefix for a frozen full-duplex
speech model (PersonaPlex, hertz-dev) that answers from the author's journal in
the author's voice. A uv workspace member, isolated from the `cartridges`
package: nothing here imports `cartridges.*`.

- Plan and research: [`docs/plans/duplex-speech-cartridge.md`](../docs/plans/duplex-speech-cartridge.md)
- Agent brief, contracts and protocol: [`docs/plans/duplex-speech-cartridge-agents.md`](../docs/plans/duplex-speech-cartridge-agents.md) (sections 4, 4b, 5)
- Epic: #7

## Layout

```
src/voice/
  backends/base.py      BackendSpec, StreamSpec, Frames, Cartridge (named KV groups), Outputs,
                        LossWeights, Targets, DuplexBackend protocol
  backends/registry.py  register(name), get(name, config), BackendConfig (discriminated on name)
  backends/fake.py      FakeBackend: two tiny discrete streams, CPU test double
  backends/personaplex.py  PersonaPlex inference half (#20); cartridge half is #8
  patches/              upstream patches applied at import (personaplex_voice_prompt: voice_prompt check;
                        hertz_sdpa, hertz_md5_stamp, hertz_fast_build: see "Cold starts")
  train.py              generic trainer; Modal app "voice-train" (gpu="H200")
  deploy.py             Modal app "voice-serve": a backend's live server behind a WebSocket URL
  client.py             offline client: user wav in real time -> agent wav + timestamps (Conversation)
  cli.py                `voice encode|train|eval|report|deploy|serve-local --backend ...`
scripts/
  hertz_parity.py       hertz load path vs the saved torch 2.5.1 reference (one H200 container)
```

Shared components (Teacher client, W&B tracking, judge, target shards,
optimizer) live in the sibling workspace member [`shared/`](../shared).

## Use

```
uv sync --all-packages
uv run --package voice pytest voice/tests shared/tests
uv run voice train --dry-run                    # CPU, FakeBackend, W&B disabled
uv run voice train --config sft.toml            # Modal, gpu="H200"
uv run voice deploy --backend personaplex --system stock   # prints the wss:// URL
uv run python -m voice.client smoke --url <wss-url> --prompts <dir of wavs>
```

## Serving and auth

`voice-serve` requires Modal proxy auth: a connection without a token is refused
(401) before it reaches a GPU. Create a workspace proxy auth token in the Modal
dashboard (Settings > Proxy Auth Tokens) and export it before using the client:

```
export MODAL_PROXY_TOKEN_ID=wk-...
export MODAL_PROXY_TOKEN_SECRET=ws-...
```

`voice.client` sends them as the `Modal-Key` / `Modal-Secret` headers only to
our workspace's hosts, `<MODAL_WORKSPACE>--*.modal.run` (`MODAL_WORKSPACE`
defaults to `joshzastrow`), and refuses to connect to them when either is
unset. Any other host gets no token.

## Cold starts

A cold start loads the weights from the HF cache on the `voice-data` Volume
(v1; cold reads 1.2-1.7 GB/s on H200). Measured on #27:

| backend | container start to listening | client's first handshake (cold, through the proxy) |
|---|---|---|
| hertz, before #27 | ~352 s | 412 s |
| hertz, after | 31-39 s (91 s on the first start after a checkpoint changes) | 28 s |
| personaplex (load path unchanged) | not timed separately (cold reads 1.6 GB/s; server ready 19 s warm) | 37 s (#20), 35 s (#27) |

hertz-dev's load runs three patches in `HertzBackend.load()`:

- `hertz_md5_stamp`: upstream MD5s every checkpoint on every start. The
  patch hashes each file once and writes `<file>.md5stamp` (path, size,
  mtime_ns, md5) next to it. Later starts trust the stamp while size and
  mtime_ns match; any change re-hashes, and a mismatch deletes the file and
  downloads it once more, then fails. The stamp is an integrity check against
  truncated or corrupted downloads, not a security control. It does not catch
  bit-rot that leaves size and mtime unchanged. `deploy_hertz.loaded_backend`
  commits the Volume so stamps persist.
- `hertz_fast_build`: upstream random-initialises the fp32 model on the CPU
  (293 s) before loading the checkpoints over it. The patch builds with
  parameters on meta and assigns the checkpoint tensors; a weight that no
  checkpoint holds raises `MissingWeights` instead of keeping a random init.
- `hertz_sdpa`: flash/efficient attention on torch 2.14 (#21).

After changing the hertz load path, re-run
`uv run --package voice python voice/scripts/hertz_parity.py` (exits 1 on a
parity failure) and `uv run python -m voice.deploy_hertz --profile` (one cold
start, timed per stage; `--group exp-...` logs it to W&B).

A backend is a new module under `backends/` with `@register("name")`, a config
class in `registry.py`, and an import in `backends/__init__.py`.

## Data

Audio of the author lives under `data/voice/` (git-ignored) and on the Modal
Volumes `voice-voice` and `voice-data`. It is personal data: never commit it,
upload it to a public W&B artifact, or paste it into an issue.
