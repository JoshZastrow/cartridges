"""How far a trained cartridge moved from its initial prefill, layer by layer.

1. Compute the init cartridge on Modal (one A100: load the base model, prefill the
   first p tokens of the teacher context block exactly as train_distill does, save):
   uv run modal run scripts/cartridge_drift.py::init
2. Pull both cartridges and build a self-contained HTML (Plotly via CDN):
   uv run modal volume get cartridges-cartridge-cache /cartridges/exp-20261016-train-27b/distill-best <dir>/trained
   uv run modal volume get cartridges-cartridge-cache /cartridges/exp-20261016-train-27b/init-prefill <dir>/init
   uv run python scripts/cartridge_drift.py build <dir>/init <dir>/trained docs/report/figures/cartridge-drift.html

The HTML holds only numbers (drift metrics, PCA coordinates, position indices),
never journal text or tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from cartridges.cartridge.train_distill import GPU, VOLUME_PATH, image, secrets, volume

MODEL_ID = "Qwen/Qwen3.8-27B"
P = 2048
INIT_DIR = f"{VOLUME_PATH}/cartridges/exp-20261016-train-27b/init-prefill"
STRIDE = 4  # PCA scatter: every 4th position; heatmap: mean over bins of 4

app = modal.App("cartridges-drift-init", image=image)


@app.function(gpu=GPU, timeout=12 * 60, volumes={VOLUME_PATH: volume}, secrets=secrets)
def compute_init(model_id: str = MODEL_ID, p: int = P) -> str:
    """The init cartridge of train_distill._distill: Cartridge.from_prefill on the
    first p tokens of the teacher context block, saved to INIT_DIR."""
    import torch

    from cartridges.cartridge import tracking
    from cartridges.cartridge.cache import Cartridge
    from cartridges.cartridge.teacher import render
    from cartridges.cartridge.train_distill import DistillConfig, _load_model

    with tracking.init_run("profile", model_id, tracking.group_name("cartridge-drift"),
                           {"check": "drift-init", "p": p}, p=p, gpu=GPU) as run:
        with run.timer("load_model"):
            model, tokenizer = _load_model(model_id)
        prefix_ids, _ = render(tokenizer, [{"role": "user", "content": "."}], Path(DistillConfig().init_text_path).read_text())
        cartridge = Cartridge.from_prefill(model, torch.tensor([prefix_ids[:p]]), meta={"source": "drift-init"})
        cartridge.save(INIT_DIR)
        volume.commit()
    return INIT_DIR


@app.local_entrypoint()
def init(model_id: str = MODEL_ID, p: int = P) -> None:
    from cartridges.cartridge import tracking

    tracking.check_budget(0.6)
    print(compute_init.remote(model_id, p))


# --- analysis (local, numpy) ---------------------------------------------------


def _load(path: Path):
    from safetensors.numpy import load_file

    return load_file(str(path / "cartridge.safetensors")), json.loads((path / "cartridge.json").read_text())


def _rel(a, b) -> float:
    import numpy as np

    return float(np.linalg.norm(b - a) / np.linalg.norm(a))


def _decompose(a, b) -> dict:
    """Split the change of a whole (flattened) tensor: relative norm change, angle
    between init and trained, and the share of |delta| perpendicular to init."""
    import numpy as np

    a, b = a.ravel(), b.ravel()
    d = b - a
    na, nb, nd = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(d)
    cos = float(a @ b / (na * nb))
    par = float(d @ a / na)
    return {"cos_flat": cos, "norm_change": float(nb / na - 1), "angle_deg": float(np.degrees(np.arccos(min(cos, 1.0)))),
            "perp_share": float(np.sqrt(max(nd**2 - par**2, 0.0)) / nd)}


def _cos(a, b) -> float:
    import numpy as np

    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def _pca_modes(a, b, idx) -> dict:
    """Three 2D views of one layer's init (a) and trained (b) vectors [positions, dim]:
    init: PCA fit on init only (both projected into it);
    joint: PCA fit on the stacked [init; trained], so the plane holds the shift;
    delta: uncentered PCA of the displacements b - a (their principal directions,
    including any shared shift), with the energy share of the top 10 directions."""
    import numpy as np

    def fit(x, mean):
        _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
        return vt[:2].T, s

    disp = b - a
    out = {}
    for mode, fit_on in (("init", a), ("joint", np.vstack([a, b]))):
        mean = fit_on.mean(0)
        basis, s = fit(fit_on, mean)
        out[mode] = {
            "init": np.round(((a - mean) @ basis)[idx], 3).T.tolist(),
            "trained": np.round(((b - mean) @ basis)[idx], 3).T.tolist(),
            "evr": np.round((s[:2] ** 2) / (s**2).sum(), 4).tolist(),
            "disp_in_plane": round(float(np.linalg.norm(disp @ basis) ** 2 / np.linalg.norm(disp) ** 2), 4),
        }
    _, s, vt = np.linalg.svd(disp, full_matrices=False)
    out["delta"] = {
        "xy": np.round((disp @ vt[:2].T)[idx], 3).T.tolist(),
        "energy10": np.round((s[:10] ** 2) / (s**2).sum(), 4).tolist(),
    }
    return out


def analyze(init_dir: Path, trained_dir: Path) -> dict:
    """Drift metrics and PCA projections. Attention metrics use the trainable
    positions 1..p-1 (position 0 is frozen and identical)."""
    import numpy as np

    si, cfg = _load(init_dir)
    st, cfg_t = _load(trained_dir)
    types = cfg["layer_types"]
    assert types == cfg_t["layer_types"], "layer types differ"
    attn = [i for i, t in enumerate(types) if t == "full_attention"]
    lin = [i for i, t in enumerate(types) if t == "linear_attention"]
    frozen_diff = max(_rel(si[k], st[k]) for k in ("frozen_keys", "frozen_values"))

    overview, heat, pca = [], {"keys": [], "values": []}, {}
    for j, layer in enumerate(attn):
        row = {"layer": layer, "type": "attention"}
        for kind in ("keys", "values"):
            # [1, heads, p-1, head_dim] -> [p-1, heads*head_dim]: one vector per position
            a = si[f"trainable_{kind}.{j}"][0].transpose(1, 0, 2).reshape(P - 1, -1).astype(np.float64)
            b = st[f"trainable_{kind}.{j}"][0].transpose(1, 0, 2).reshape(P - 1, -1).astype(np.float64)
            per_pos = np.linalg.norm(b - a, axis=1) / np.linalg.norm(a, axis=1)
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            row[f"{kind}_drift"] = _rel(a, b)
            row[f"{kind}_cos"] = float(cos.mean())
            row.update({f"{kind}_{k}": v for k, v in _decompose(a, b).items()})
            n_bins = len(per_pos) // STRIDE
            heat[kind].append(np.round(per_pos[: n_bins * STRIDE].reshape(n_bins, STRIDE).mean(1), 4).tolist())
            idx = np.arange(0, P - 1, STRIDE)
            pca[f"{layer}-{kind}"] = {"pos": (idx + 1).tolist(), **_pca_modes(a, b, idx)}  # position 0 is the frozen sink
        overview.append(row)
    deltanet_heads = []
    for j, layer in enumerate(lin):
        ci, ct = si[f"conv_states.{j}"].astype(np.float64), st[f"conv_states.{j}"].astype(np.float64)
        ri, rt = si[f"recurrent_states.{j}"][0].astype(np.float64), st[f"recurrent_states.{j}"][0].astype(np.float64)
        overview.append({
            "layer": layer, "type": "deltanet",
            "recurrent_drift": _rel(ri, rt), "recurrent_cos": _cos(ri, rt),
            "conv_drift": _rel(ci, ct), "conv_cos": _cos(ci, ct),
            **{f"recurrent_{k}": v for k, v in _decompose(ri, rt).items()},
            **{f"conv_{k}": v for k, v in _decompose(ci, ct).items()},
        })
        deltanet_heads.append(np.round(
            np.linalg.norm((rt - ri).reshape(len(ri), -1), axis=1) / np.linalg.norm(ri.reshape(len(ri), -1), axis=1), 4
        ).tolist())
    overview.sort(key=lambda r: r["layer"])
    return {
        "meta": {"base_model": cfg_t.get("base_model"), "p": P, "stride": STRIDE, "n_layers": len(types),
                 "attn_layers": attn, "deltanet_layers": lin, "frozen_rel_diff": frozen_diff,
                 "trained_run": cfg_t.get("distill_run_id"), "group": cfg_t.get("group")},
        "overview": overview, "heat": heat, "pca": pca, "deltanet_heads": deltanet_heads,
    }


def build(init_dir: str, trained_dir: str, out: str) -> None:
    data = analyze(Path(init_dir), Path(trained_dir))
    template = (Path(__file__).parent / "cartridge_drift.html").read_text()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.2f} MB); frozen-position diff {data['meta']['frozen_rel_diff']:.2e}")


if __name__ == "__main__":
    import fire

    fire.Fire({"build": build})
