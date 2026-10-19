"""Embedding-neighbour review columns and summary metrics for the k-sweep of
real-question examples (k real questions shown per synthesis prompt).

Everything runs locally on CPU: a small embedding model (BAAI/bge-small-en-v1.5
from the local Hugging Face cache, CLS pooling, unit vectors), and no text
leaves the machine. Nothing here picks a winning k: it adds columns for the
user's review and logs per-k numbers next to a real-vs-real baseline.

Per draft (columns):
    nn_sim_generated   max cosine to another draft of the same k (high = duplicate)
    nn_sim_reference   max cosine to the held-out real questions (high = realistic, or copied)
    nn_sim_shown       max cosine to the examples shown in its own prompt (high = copying); None for k=0

Per k (summary), generated vs reference:
    prdc@{3,5,10}      precision, recall, density, coverage (Naeem et al. 2020), k-NN balls
                       in embedding space
    diversity          1 - mean pairwise cosine among the drafts
    disc_auc           cross-validated AUC of a mean-embedding classifier (score = x .
                       (mean_gen - mean_ref)); 0.5 = indistinguishable
A real-vs-real baseline (the reference split in seeded halves) gives the same
numbers for two samples of real questions.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
PRDC_KS = (3, 5, 10)

Embed = Callable[[Sequence[str]], np.ndarray]  # texts -> [n, d] unit vectors


def local_embedder(model_id: str = EMBED_MODEL, batch_size: int = 64) -> Embed:
    """A CPU embedder over the local HF cache only (local_files_only: no download,
    nothing sent anywhere)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if len(tok) < 1000:  # a cache without vocab files loads an empty tokenizer: every text embeds the same
        raise RuntimeError(f"{model_id}: tokenizer has {len(tok)} tokens; download its tokenizer files first")
    model = AutoModel.from_pretrained(model_id, local_files_only=True).eval()

    @torch.no_grad()
    def embed(texts: Sequence[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            x = tok(list(texts[i : i + batch_size]), padding=True, truncation=True, max_length=512, return_tensors="pt")
            cls = model(**x).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls, dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, model.config.hidden_size), dtype=np.float32)

    return embed


# ---------------------------------------------------------------- pure


def _max_sim(a: np.ndarray, b: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """For each row of a, its max cosine to the rows of b (unit vectors)."""
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), np.nan)
    sims = a @ b.T
    if exclude_self:
        np.fill_diagonal(sims, -np.inf)
    return sims.max(axis=1)


def neighbor_columns(drafts: Sequence[str], shown: Sequence[Sequence[str]], reference: Sequence[str],
                     embed: Embed) -> list[dict]:
    """The three review columns for the drafts of one k."""
    gen, ref = embed(drafts), embed(reference)
    nn_gen = _max_sim(gen, gen, exclude_self=True) if len(drafts) > 1 else np.full(len(drafts), np.nan)
    nn_ref = _max_sim(gen, ref)
    rows = []
    for i, examples in enumerate(shown):
        nn_shown = float(_max_sim(gen[i : i + 1], embed(examples))[0]) if examples else None
        rows.append({"nn_sim_generated": float(nn_gen[i]), "nn_sim_reference": float(nn_ref[i]),
                     "nn_sim_shown": nn_shown})
    return rows


def _dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T))


def _knn_radius(x: np.ndarray, k: int) -> np.ndarray:
    d = _dists(x, x)
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, min(k, len(x) - 1) - 1]


def prdc(real: np.ndarray, gen: np.ndarray, k: int = 5) -> dict:
    """Precision, recall, density and coverage (Naeem et al. 2020, Reliable
    Fidelity and Diversity Metrics for Generative Models)."""
    if len(real) < 2 or len(gen) < 2:
        return {"precision": None, "recall": None, "density": None, "coverage": None}
    r_real, r_gen = _knn_radius(real, k), _knn_radius(gen, k)
    d = _dists(real, gen)  # [n_real, n_gen]
    inside = d <= r_real[:, None]  # gen j inside real i's ball
    return {
        "precision": float(inside.any(axis=0).mean()),
        "recall": float((d <= r_gen[None, :]).any(axis=1).mean()),
        "density": float(inside.sum() / (min(k, len(real) - 1) * len(gen))),
        "coverage": float((d.min(axis=1) <= r_real).mean()),
    }


def diversity(x: np.ndarray) -> float | None:
    """1 - mean off-diagonal cosine (unit vectors)."""
    n = len(x)
    if n < 2:
        return None
    sims = x @ x.T
    return float(1 - (sims.sum() - np.trace(sims)) / (n * (n - 1)))


def auc(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    """Mann-Whitney AUC: P(score of a positive > score of a negative), ties half."""
    if not len(pos) or not len(neg):
        return None
    p, q = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > q).sum() + 0.5 * (p == q).sum()) / (p.size * q.size))


def discriminator_auc(real: np.ndarray, gen: np.ndarray, folds: int = 5, seed: int = 0) -> float | None:
    """Cross-validated AUC of score(x) = x . (mean_gen - mean_real), the means
    taken on the other folds. 0.5 = the means don't tell them apart."""
    if len(real) < folds or len(gen) < folds:
        return None
    rng = random.Random(seed)
    fold_r = [i % folds for i in rng.sample(range(len(real)), len(real))]
    fold_g = [i % folds for i in rng.sample(range(len(gen)), len(gen))]
    fr, fg = np.array(fold_r), np.array(fold_g)
    pos, neg = [], []
    for f in range(folds):
        w = gen[fg != f].mean(0) - real[fr != f].mean(0)
        pos += list(gen[fg == f] @ w)
        neg += list(real[fr == f] @ w)
    return auc(pos, neg)


def compare(real: np.ndarray, gen: np.ndarray, seed: int = 0) -> dict:
    out = {"n_real": len(real), "n_gen": len(gen), "diversity": diversity(gen),
           "disc_auc": discriminator_auc(real, gen, seed=seed)}
    for k in PRDC_KS:
        out |= {f"{m}@{k}": v for m, v in prdc(real, gen, k).items()}
    return out


def real_baseline(real: np.ndarray, seed: int = 0) -> dict:
    """compare() between two seeded halves of the reference: what two samples of
    real questions score against each other."""
    idx = random.Random(seed).sample(range(len(real)), len(real))
    half = len(real) // 2
    return compare(real[idx[:half]], real[idx[half:]], seed=seed)


def sweep_summary(drafts_by_k: dict[int, Sequence[str]], reference: Sequence[str], embed: Embed,
                  seed: int = 0) -> dict:
    """{"baseline": real vs real, k: generated vs reference, ...}; no winner."""
    ref = embed(reference)
    out = {"baseline": real_baseline(ref, seed) | {"diversity_reference": diversity(ref)}}
    for k, drafts in sorted(drafts_by_k.items()):
        out[k] = compare(ref, embed(drafts), seed)
    return out


# ---------------------------------------------------------------- k-sweep table API (datagen calls these)


def add_neighbor_columns(rows: Sequence[dict], reference_rows: Sequence[dict], shown_key: str = "exemplars_shown",
                         text_key: str = "question", embed: Embed | None = None) -> list[dict]:
    """The rows of one k table with nn_sim_generated, nn_sim_reference and
    nn_sim_shown added. rows[i][shown_key] is the list of example question
    texts shown in that draft's prompt ([] for k=0); reference_rows hold the
    held-out real questions under text_key."""
    embed = embed or local_embedder()
    cols = neighbor_columns([r[text_key] for r in rows], [list(r.get(shown_key) or []) for r in rows],
                            [r[text_key] for r in reference_rows], embed)
    return [{**r, **c} for r, c in zip(rows, cols, strict=True)]


def summarize_k(rows: Sequence[dict], reference_rows: Sequence[dict], text_key: str = "question",
                embed: Embed | None = None, seed: int = 0) -> dict:
    """Run-summary numbers for one k table, flat: qgen/<metric> (drafts vs the
    reference) and qgen/baseline/<metric> (real vs real halves), plus the mean
    of each review column when present. No winner is picked."""
    embed = embed or local_embedder()
    ref, gen = embed([r[text_key] for r in reference_rows]), embed([r[text_key] for r in rows])
    out = {f"qgen/{k}": v for k, v in compare(ref, gen, seed).items()}
    out |= {f"qgen/baseline/{k}": v for k, v in real_baseline(ref, seed).items()}
    out["qgen/baseline/diversity_reference"] = diversity(ref)
    for col in ("nn_sim_generated", "nn_sim_reference", "nn_sim_shown"):
        values = [r[col] for r in rows if r.get(col) is not None]
        if values:
            out[f"qgen/{col}_mean"] = float(np.mean(values))
    return out
