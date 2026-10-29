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
  patches/              fork patches applied at import (personaplex_voice_prompt: voice_prompt check)
  train.py              generic trainer; Modal app "voice-train" (gpu="H200")
  deploy.py             Modal app "voice-serve": a backend's live server behind a WebSocket URL
  client.py             offline client: user wav in real time -> agent wav + timestamps (Conversation)
  cli.py                `voice encode|train|eval|report|deploy|serve-local --backend ...`
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

`voice.client` sends them as the `Modal-Key` / `Modal-Secret` headers to any
`*.modal.run` URL and refuses to connect when either is unset.

A backend is a new module under `backends/` with `@register("name")`, a config
class in `registry.py`, and an import in `backends/__init__.py`.

## Data

Audio of the author lives under `data/voice/` (git-ignored) and on the Modal
Volumes `voice-voice` and `voice-data`. It is personal data: never commit it,
upload it to a public W&B artifact, or paste it into an issue.
