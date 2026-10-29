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
  train.py              generic trainer; Modal app "voice-train" (gpu="H200")
  cli.py                `voice encode|train|eval|report|deploy --backend ...`
```

Shared components (Teacher client, W&B tracking, judge, target shards,
optimizer) live in the sibling workspace member [`shared/`](../shared).

## Use

```
uv sync --all-packages
uv run --package voice pytest voice/tests shared/tests
uv run voice train --dry-run                    # CPU, FakeBackend, W&B disabled
uv run voice train --config sft.toml            # Modal, gpu="H200"
```

A backend is a new module under `backends/` with `@register("name")`, a config
class in `registry.py`, and an import in `backends/__init__.py`.

## Data

Audio of the author lives under `data/voice/` (git-ignored) and on the Modal
Volumes `voice-voice` and `voice-data`. It is personal data: never commit it,
upload it to a public W&B artifact, or paste it into an issue.
