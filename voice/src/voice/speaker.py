"""WavLM-TDNN speaker embedder: the brief §8 speaker-similarity model.

    WavLMTdnn.load() -> embedder;  embedder(wav16k [samples]) -> [256] unit vector

UniSpeech's speaker-verification checkpoint "WavLM large, fix pre-train: No"
(`wavlm_large_finetune.pth`, the SIM model of seed-tts-eval, VALL-E and
PersonaPlex), run with transformers instead of s3prl: an HF `WavLMModel`
with the checkpoint's backbone remapped onto it, plus the ECAPA-TDNN head over
a softmax-weighted sum of its 25 hidden states. Same-speaker cosines on
VoxCeleb are ~0.6-0.7; different speakers ~0 (not the scale of
`microsoft/wavlm-base-plus-sv`).

Sources:
- Vendored here: the backbone key remap and the HF-WavLM wiring
  (`convert_backbone`, `WavLMEcapaTdnn`) from huggingface/open-tts-leaderboard
  `transformers/score_similarity.py` at 1645e63016f0556e69cbd4850c95d7cb95943877
  (Apache-2.0), reported bit-exact against upstream.
- Not vendored: the ECAPA-TDNN head classes (Res2Conv1dReluBn, Conv1dReluBn,
  SE_Connect, SE_Res2Block, AttentiveStatsPool) from microsoft/UniSpeech
  `downstreams/speaker_verification/models/ecapa_tdnn.py` at commit
  40ac183e772abdcbf3a318eccffaa70c0cfd3a74, CC BY-SA 3.0 (Microsoft,
  UniSpeech authors). `load_head` downloads that file once into VOICE_CACHE,
  checks its sha256 (HEAD_SHA256) before every load, and executes only those
  five class definitions (not the file's imports of torchaudio and s3prl).
The checkpoint (CC BY-SA 3.0) is never committed; it is fetched from
a Hub mirror pinned by the sha256 of the official file (CHECKPOINT_SHA256; the
same digest on three independent mirrors), loaded with `weights_only=True`.
One clip per forward: instance norm and attentive pooling take time-axis
statistics, so padding would shift the embedding.
"""

from __future__ import annotations

import ast
import hashlib
import os
import types
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_REPO = "bezzam/wavlm_large_finetune_seed_tts_eval"
CHECKPOINT_FILE = "wavlm_large_finetune.pth"
CHECKPOINT_SHA256 = "51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b"
BACKBONE_CONFIG = "microsoft/wavlm-large"  # config.json only; every weight comes from the checkpoint
MODEL_ID = f"wavlm-tdnn:{CHECKPOINT_FILE}@sha256:{CHECKPOINT_SHA256[:12]}"
SAMPLE_RATE = 16000
HEAD_COMMIT = "40ac183e772abdcbf3a318eccffaa70c0cfd3a74"
HEAD_URL = (f"https://raw.githubusercontent.com/microsoft/UniSpeech/{HEAD_COMMIT}"
            "/downstreams/speaker_verification/models/ecapa_tdnn.py")
HEAD_SHA256 = "9571bb524980c1447d64c4fadaa31a4772d0ca1e9864d523c091ed600ff134e6"
HEAD_CLASSES = ("Res2Conv1dReluBn", "Conv1dReluBn", "SE_Connect", "SE_Res2Block", "AttentiveStatsPool")
CACHE = Path(os.environ.get("VOICE_CACHE", Path.home() / ".cache/voice"))


def head_path() -> Path:
    """UniSpeech's ecapa_tdnn.py at HEAD_COMMIT in VOICE_CACHE (downloaded once), digest-checked."""
    path = CACHE / f"ecapa_tdnn-{HEAD_COMMIT[:12]}.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HEAD_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != HEAD_SHA256:
        raise RuntimeError(f"{path}: sha256 {digest} != {HEAD_SHA256}; refusing to execute it")
    return path


def load_head(path: Path | None = None) -> types.SimpleNamespace:
    """The HEAD_CLASSES of the verified upstream file, defined against torch, nn and F only."""
    tree = ast.parse((path or head_path()).read_text())
    body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in HEAD_CLASSES]
    assert {n.name for n in body} == set(HEAD_CLASSES), [n.name for n in body]
    scope = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path or HEAD_URL), "exec"), scope)  # noqa: S102
    return types.SimpleNamespace(**{name: scope[name] for name in HEAD_CLASSES})


class WavLMEcapaTdnn(nn.Module):
    """UniSpeech `ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')` over an HF `WavLMModel`;
    `head` holds the upstream head classes (`load_head`)."""

    def __init__(self, wavlm, head: types.SimpleNamespace, feat_dim=1024, channels=512, emb_dim=256):
        super().__init__()
        self.wavlm = wavlm
        self.feature_weight = nn.Parameter(torch.zeros(1 + wavlm.config.num_hidden_layers))
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.layer1 = head.Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = head.SE_Res2Block(channels, channels, 3, 1, 2, 2, scale=8, se_bottleneck_dim=128)
        self.layer3 = head.SE_Res2Block(channels, channels, 3, 1, 3, 3, scale=8, se_bottleneck_dim=128)
        self.layer4 = head.SE_Res2Block(channels, channels, 3, 1, 4, 4, scale=8, se_bottleneck_dim=128)
        self.conv = nn.Conv1d(channels * 3, 1536, kernel_size=1)
        self.pooling = head.AttentiveStatsPool(1536, attention_channels=128)
        self.bn = nn.BatchNorm1d(1536 * 2)
        self.linear = nn.Linear(1536 * 2, emb_dim)

    def forward(self, wav):
        hidden = torch.stack(self.wavlm(wav, output_hidden_states=True).hidden_states, dim=0)  # [25, B, T, C]
        x = (F.softmax(self.feature_weight, dim=-1).view(-1, 1, 1, 1) * hidden).sum(dim=0)
        x = self.instance_norm(x.transpose(1, 2) + 1e-6)  # [B, C, T]
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.relu(self.conv(torch.cat([out2, out3, out4], dim=1)))
        return self.linear(self.bn(self.pooling(out)))


def convert_backbone(orig: dict, target_keys: set[str]) -> dict:
    """Original-WavLM names (`feature_extract.model.*`) -> HF `WavLMModel` names; raises on any gap."""
    prefix = "feature_extract.model."
    parametrized = "encoder.pos_conv_embed.conv.parametrizations.weight.original0" in target_keys
    pos_g = "parametrizations.weight.original0" if parametrized else "weight_g"
    pos_v = "parametrizations.weight.original1" if parametrized else "weight_v"
    layer_map = (("self_attn.q_proj.", "attention.q_proj."), ("self_attn.k_proj.", "attention.k_proj."),
                 ("self_attn.v_proj.", "attention.v_proj."), ("self_attn.out_proj.", "attention.out_proj."),
                 ("self_attn.grep_linear.", "attention.gru_rel_pos_linear."),
                 ("self_attn.relative_attention_bias.", "attention.rel_attn_embed."),
                 ("self_attn_layer_norm.", "layer_norm."), ("final_layer_norm.", "final_layer_norm."),
                 ("fc1.", "feed_forward.intermediate_dense."), ("fc2.", "feed_forward.output_dense."))

    def convert(key: str) -> str | None:
        if key.startswith("feature_extractor.conv_layers."):
            layer_id, sub = key[len("feature_extractor.conv_layers.") :].split(".", 1)
            if sub == "0.weight":
                return f"feature_extractor.conv_layers.{layer_id}.conv.weight"
            if sub.startswith("2.1."):
                return f"feature_extractor.conv_layers.{layer_id}.layer_norm.{sub[4:]}"
            return None
        if key.startswith("layer_norm."):
            return "feature_projection." + key
        if key.startswith("post_extract_proj."):
            return "feature_projection.projection." + key[len("post_extract_proj.") :]
        if key == "mask_emb":
            return "masked_spec_embed"
        if key.startswith("encoder.pos_conv.0."):
            mapped = {"bias": "bias", "weight_g": pos_g, "weight_v": pos_v}.get(key[len("encoder.pos_conv.0.") :])
            return f"encoder.pos_conv_embed.conv.{mapped}" if mapped else None
        if key.startswith("encoder.layer_norm."):
            return key
        if key.startswith("encoder.layers."):
            layer_id, sub = key[len("encoder.layers.") :].split(".", 1)
            if sub == "self_attn.grep_a":
                return f"encoder.layers.{layer_id}.attention.gru_rel_pos_const"
            for src, dst in layer_map:
                if sub.startswith(src):
                    return f"encoder.layers.{layer_id}.{dst}{sub[len(src):]}"
        return None

    converted, unmapped = {}, []
    for key, value in orig.items():
        if not key.startswith(prefix):
            continue
        mapped = convert(key[len(prefix) :])
        if mapped is None:
            unmapped.append(key)
        else:
            converted[mapped] = value
    missing, unexpected = target_keys - converted.keys(), converted.keys() - target_keys
    if unmapped or missing or unexpected:
        raise RuntimeError(f"WavLM backbone remap: unmapped {sorted(unmapped)[:5]}, missing {sorted(missing)[:5]}, "
                           f"unexpected {sorted(unexpected)[:5]}")
    return converted


def checkpoint_path() -> str:
    """The checkpoint from the Hub cache (downloaded once), digest-checked."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CHECKPOINT_REPO, CHECKPOINT_FILE)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    if h.hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError(f"{CHECKPOINT_REPO}/{CHECKPOINT_FILE}: sha256 {h.hexdigest()} != {CHECKPOINT_SHA256}")
    return path


class WavLMTdnn:
    def __init__(self, model: WavLMEcapaTdnn):
        self.model = model.eval()

    @classmethod
    def load(cls, path: str | None = None) -> WavLMTdnn:
        from transformers import WavLMConfig, WavLMModel

        config = WavLMConfig.from_pretrained(BACKBONE_CONFIG)
        config.apply_spec_augment, config.layerdrop = False, 0.0
        backbone = WavLMModel(config)
        state = torch.load(path or checkpoint_path(), map_location="cpu", weights_only=True)["model"]
        backbone.load_state_dict(convert_backbone(state, set(backbone.state_dict())), strict=True)
        model = WavLMEcapaTdnn(backbone, load_head(), feat_dim=config.hidden_size)
        head = {k: v for k, v in state.items() if not k.startswith(("feature_extract.", "loss_calculator."))}
        missing, unexpected = model.load_state_dict(head, strict=False)
        stray = [k for k in missing if not k.startswith("wavlm.")]
        if stray or unexpected:
            raise RuntimeError(f"ECAPA head keys: missing {stray[:5]}, unexpected {list(unexpected)[:5]}")
        return cls(model)

    @torch.no_grad()
    def __call__(self, wav16k) -> torch.Tensor:
        wav = torch.as_tensor(wav16k, dtype=torch.float32).reshape(1, -1)
        wav = F.layer_norm(wav, wav.shape)  # WavLM-Large expects layer-normalised waveforms (s3prl does this)
        return F.normalize(self.model(wav), dim=-1)[0]
