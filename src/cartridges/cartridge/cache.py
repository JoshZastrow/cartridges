"""The Cartridge: a trainable cache for a frozen hybrid qwen3_5 model.

qwen3_5 interleaves `full_attention` layers (a KV cache) with Gated DeltaNet
`linear_attention` layers (a conv state of the last `linear_conv_kernel_dim`
inputs and a recurrent state [heads, k_dim, v_dim]). A cartridge is, per layer:

- full_attention: keys/values for `p` prefix tokens, shape [1, kv_heads, p, head_dim],
  post-RoPE at positions 0..p-1, so the model's own tokens start at position p;
- linear_attention: the conv state [1, conv_dim, kernel] and the recurrent state
  [1, v_heads, k_dim, v_dim] that the layer reads before the first real token.

`init_from_text` prefills the first `p` tokens of the corpus and keeps the
resulting cache (the paper's initialization), so a fresh cartridge reproduces
the logits of prefilling those tokens exactly. As in HazyResearch/cartridges,
the first `num_frozen_tokens` KV positions (the attention sink) are not trained.

`as_past_key_values` hands the HF forward a `DynamicCache` whose layers already
hold the cartridge tensors. Gradients reach them because the attention layers
`torch.cat` the cached keys with the new ones, and the linear layers concatenate
the conv state with the new inputs and pass the recurrent state as `initial_state`
to the gated-delta-rule kernel (the torch fallback and fla's chunk kernel both
differentiate through it). HF's linear layer `copy_`s new states into its buffers
in place, which would clobber tensors autograd saved, so `_CartridgeLinearLayer`
replaces them instead.

Artifact: a directory with `cartridge.safetensors` and `cartridge.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, LinearAttentionLayer

WEIGHTS = "cartridge.safetensors"
CONFIG = "cartridge.json"


class _ReadOnlyKVLayer(DynamicLayer):
    """An attention cache layer that returns prefix + new keys/values without storing them."""

    def update(self, key_states, value_states, *args, **kwargs):
        return torch.cat([self.keys, key_states], dim=-2), torch.cat([self.values, value_states], dim=-2)


class _CartridgeLinearLayer(LinearAttentionLayer):
    """A linear-attention cache layer that starts from given states and never writes in place.

    With `read_only`, updates are not stored, so a forward can be replayed (gradient checkpointing).
    """

    def __init__(self, conv_state: torch.Tensor, recurrent_state: torch.Tensor, read_only: bool = False):
        super().__init__(number_of_states=1)
        self.read_only = read_only
        self.dtype, self.device = conv_state.dtype, conv_state.device
        self.conv_kernel_size[0] = conv_state.shape[-1]
        self.conv_states[0] = conv_state
        self.recurrent_states[0] = recurrent_state
        self.is_conv_states_initialized[0] = True
        self.is_recurrent_states_initialized[0] = True
        self.has_previous_state[0] = True

    def update_conv_state(self, conv_states, state_idx=0, **kwargs):
        full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
        if self.read_only:
            return full
        # Contiguous: the single-token decode kernel updates this buffer in place.
        self.conv_states[state_idx] = full[..., -self.conv_kernel_size[state_idx] :].contiguous()
        return full

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        if not self.read_only:
            self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states


class CartridgeCache(DynamicCache):
    """A DynamicCache built from pre-filled layers (see `Cartridge.as_past_key_values`)."""

    def __init__(self, layers: list):
        Cache.__init__(self, layers=layers)


class Cartridge(nn.Module):
    """Trainable KV prefix (attention layers) + initial conv/recurrent state (linear layers)."""

    def __init__(
        self,
        layer_types: list[str],
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        conv_states: list[torch.Tensor],
        recurrent_states: list[torch.Tensor],
        num_frozen_tokens: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        meta: dict | None = None,
    ):
        super().__init__()
        self.layer_types = list(layer_types)
        self.attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        self.linear_layers = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
        assert len(self.attn_layers) + len(self.linear_layers) == len(layer_types), layer_types
        assert len(keys) == len(values) == len(self.attn_layers)
        assert len(conv_states) == len(recurrent_states) == len(self.linear_layers)
        self.p = keys[0].shape[2]
        self.num_frozen_tokens = num_frozen_tokens
        self.dtype_ = dtype  # the model's dtype; parameters are kept in fp32
        self.meta = dict(meta or {})

        def param(t):
            return nn.Parameter(t.detach().float().contiguous().clone())

        n = num_frozen_tokens
        for name, kv in (("keys", keys), ("values", values)):
            self.register_buffer(f"frozen_{name}", torch.stack([t[:, :, :n].float() for t in kv]), persistent=True)
            setattr(self, f"trainable_{name}", nn.ParameterList([param(t[:, :, n:]) for t in kv]))
        self.conv_states = nn.ParameterList([param(t) for t in conv_states])
        self.recurrent_states = nn.ParameterList([param(t) for t in recurrent_states])

    # ---- construction -------------------------------------------------------

    @classmethod
    @torch.no_grad()
    def from_prefill(cls, model, input_ids: torch.Tensor, num_frozen_tokens: int = 1, meta: dict | None = None):
        """Prefill `input_ids` ([1, p]) through `model` and keep the resulting cache."""
        config = model.config.get_text_config(decoder=True)
        cache = DynamicCache(config=model.config)
        model(input_ids=input_ids.to(model.device), past_key_values=cache, use_cache=True)
        layers = cache.layers
        types = list(config.layer_types)
        return cls(
            types,
            keys=[layers[i].keys for i, t in enumerate(types) if t == "full_attention"],
            values=[layers[i].values for i, t in enumerate(types) if t == "full_attention"],
            conv_states=[layers[i].conv_states[0] for i, t in enumerate(types) if t == "linear_attention"],
            recurrent_states=[layers[i].recurrent_states[0] for i, t in enumerate(types) if t == "linear_attention"],
            num_frozen_tokens=num_frozen_tokens,
            dtype=model.dtype,
            meta={"base_model": getattr(model.config, "_name_or_path", ""), **(meta or {})},
        ).to(model.device)

    @classmethod
    def init_from_text(cls, model, tokenizer, text: str, p: int, num_frozen_tokens: int = 1, meta: dict | None = None):
        """The paper's initialization on the teacher's own prefix: the cache of the first
        `p` tokens of teacher.context_block(text) (system header, instructions, corpus
        start), so the cartridge stands where the teacher's context block stands and
        the student's prompt_ids (teacher.render) follow it at position p."""
        from cartridges.cartridge.teacher import render

        prefix_ids, _ = render(tokenizer, [{"role": "user", "content": "."}], text)
        if len(prefix_ids) < p:
            raise ValueError(f"context block has {len(prefix_ids)} tokens, fewer than p={p}")
        ids = torch.tensor([prefix_ids[:p]])
        return cls.from_prefill(model, ids, num_frozen_tokens=num_frozen_tokens, meta=meta)

    # ---- use with the HF forward ---------------------------------------------

    def as_past_key_values(self, batch_size: int = 1, read_only: bool = True) -> CartridgeCache:
        """A fresh hybrid cache holding the cartridge; differentiable w.r.t. the cartridge.

        `read_only` (training): layers return cartridge + new states without storing them,
        so the forward can be replayed under gradient checkpointing (set the decoder
        layers' `_can_checkpoint_with_cache = True`). Pass False to continue the
        sequence from `output.past_key_values` (generation).
        """
        dtype = self.dtype_
        attn = iter(range(len(self.attn_layers)))
        lin = iter(range(len(self.linear_layers)))
        layers = []
        for t in self.layer_types:
            if t == "full_attention":
                j = next(attn)
                layer = _ReadOnlyKVLayer() if read_only else DynamicLayer()
                kv = []
                for frozen, trainable in ((self.frozen_keys, self.trainable_keys), (self.frozen_values, self.trainable_values)):
                    full = torch.cat([frozen[j], trainable[j]], dim=2).to(dtype)
                    kv.append(full.expand(batch_size, -1, -1, -1))
                layer.lazy_initialization(*kv)
                layer.keys, layer.values = kv
            else:
                j = next(lin)
                conv = self.conv_states[j].to(dtype).expand(batch_size, -1, -1).clone()
                rec = self.recurrent_states[j].expand(batch_size, -1, -1, -1)
                layer = _CartridgeLinearLayer(conv, rec, read_only)
            layers.append(layer)
        return CartridgeCache(layers)

    def forward(
        self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, read_only: bool = True, **kwargs
    ):
        """Run `model` on `input_ids` ([B, L], right-padded) after the cartridge.

        `attention_mask` covers `input_ids` only; the cartridge columns are added here.
        With `read_only=False`, `output.past_key_values` continues the sequence.
        """
        b, n = input_ids.shape
        position_ids = torch.arange(self.p, self.p + n, device=input_ids.device).expand(b, -1)
        if attention_mask is not None:
            prefix = attention_mask.new_ones(b, self.p)
            attention_mask = torch.cat([prefix, attention_mask], dim=1)
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self.as_past_key_values(b, read_only),
            use_cache=True,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        model,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_token_id: int | list[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample continuations of `input_ids` ([B, L], same prompt length, no padding).

        Returns [B, T] new tokens; rows that hit EOS are padded with the EOS id after it.
        Unpadded prompts keep padding out of the linear layers' recurrent state.
        """
        eos = torch.tensor([eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or []), device=input_ids.device)
        out = self(model, input_ids, read_only=False)
        cache, pos = out.past_key_values, self.p + input_ids.shape[1]
        logits = out.logits[:, -1].float()
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        tokens = []
        for _ in range(max_new_tokens):
            nxt = sample(logits, temperature, top_p, generator)
            if eos.numel():
                nxt = torch.where(done, eos[0], nxt)
                done |= torch.isin(nxt, eos)
            tokens.append(nxt)
            if done.all():
                break
            position_ids = torch.full((input_ids.shape[0], 1), pos, device=input_ids.device)
            out = model(input_ids=nxt[:, None], position_ids=position_ids, past_key_values=cache, use_cache=True)
            cache, pos = out.past_key_values, pos + 1
            logits = out.logits[:, -1].float()
        return torch.stack(tokens, dim=1) if tokens else input_ids.new_zeros(input_ids.shape[0], 0)

    # ---- artifact -------------------------------------------------------------

    def config_dict(self) -> dict:
        return {
            "p": self.p,
            "num_frozen_tokens": self.num_frozen_tokens,
            "layer_types": self.layer_types,
            "dtype": str(self.dtype_).removeprefix("torch."),
            **self.meta,
        }

    def save(self, path: str | Path) -> Path:
        """Write `cartridge.safetensors` and `cartridge.json` into directory `path`."""
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(tensors, str(path / WEIGHTS))
        (path / CONFIG).write_text(json.dumps(self.config_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "Cartridge":
        from safetensors.torch import load_file

        path = Path(path)
        cfg = json.loads((path / CONFIG).read_text())
        sd = load_file(str(path / WEIGHTS), device=str(device or "cpu"))
        types = cfg.pop("layer_types")
        n_attn = sum(t == "full_attention" for t in types)
        n_lin = len(types) - n_attn
        frozen_k, frozen_v = sd["frozen_keys"], sd["frozen_values"]

        def kv(name, frozen):
            return [torch.cat([frozen[j], sd[f"trainable_{name}.{j}"]], dim=2) for j in range(n_attn)]

        num_frozen = cfg.pop("num_frozen_tokens")
        cfg.pop("p")
        return cls(
            types,
            keys=kv("keys", frozen_k),
            values=kv("values", frozen_v),
            conv_states=[sd[f"conv_states.{j}"] for j in range(n_lin)],
            recurrent_states=[sd[f"recurrent_states.{j}"] for j in range(n_lin)],
            num_frozen_tokens=num_frozen,
            dtype=getattr(torch, cfg.pop("dtype")),
            meta=cfg,
        )

    def nbytes(self) -> int:
        """Memory of the cache at the model's dtype (recurrent states stay fp32), for the report."""
        size = torch.finfo(self.dtype_).bits // 8
        kv = sum(t.numel() for t in (self.frozen_keys, self.frozen_values)) + sum(
            p.numel() for p in [*self.trainable_keys, *self.trainable_values, *self.conv_states]
        )
        return kv * size + sum(p.numel() * 4 for p in self.recurrent_states)


def sample(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Greedy when temperature == 0, else nucleus sampling from [B, V] logits."""
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, idx = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_probs)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
