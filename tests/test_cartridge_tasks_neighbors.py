"""k-sweep embedding columns and metrics (tasks/neighbors.py) with a fake embedder."""

from __future__ import annotations

import numpy as np
import pytest

from cartridges.cartridge.tasks import neighbors as nb


def fake_embed(dim: int = 16):
    """Same text -> same unit vector; texts sharing a 'topic:' prefix are close."""
    def vec(text: str) -> np.ndarray:
        topic, _, rest = text.partition(":")
        base = np.random.default_rng(abs(hash(topic)) % 2**32).normal(size=dim)
        noise = np.random.default_rng(abs(hash(text)) % 2**32).normal(size=dim) * (0.1 if rest else 0.0)
        v = base + noise
        return v / np.linalg.norm(v)

    def embed(texts):
        return np.array([vec(t) for t in texts]).reshape(len(texts), dim)

    return embed


def test_neighbor_columns_uniqueness_reference_and_copying():
    embed = fake_embed()
    drafts = ["car:a", "car:a", "dog:b"]  # a duplicate pair and a distinct one
    shown = [["car:a"], ["tree:z"], []]  # first copies its example; third had k=0
    rows = nb.neighbor_columns(drafts, shown, ["dog:x", "fish:y"], embed)
    assert rows[0]["nn_sim_generated"] == pytest.approx(1.0) and rows[2]["nn_sim_generated"] < 0.9
    assert rows[0]["nn_sim_shown"] == pytest.approx(1.0) and rows[1]["nn_sim_shown"] < 0.9
    assert rows[2]["nn_sim_shown"] is None
    assert rows[2]["nn_sim_reference"] > rows[0]["nn_sim_reference"]  # dog draft is near the dog reference


def test_prdc_identical_sets_are_perfect_and_disjoint_sets_are_zero():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(40, 8))
    same = nb.prdc(real, real.copy(), k=5)
    assert (same["precision"], same["recall"], same["coverage"]) == (1.0, 1.0, 1.0)
    assert 0.8 < same["density"] < 1.5  # density is ~1 in expectation for matching distributions
    far = nb.prdc(real, real + 100.0, k=5)
    assert far == {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}


def test_prdc_mode_collapse_keeps_precision_but_loses_recall_and_coverage():
    rng = np.random.default_rng(1)
    real = rng.normal(size=(60, 8))
    collapsed = real[:2].repeat(30, axis=0) + rng.normal(scale=1e-3, size=(60, 8))
    m, full = nb.prdc(real, collapsed, k=5), nb.prdc(real, rng.normal(size=(60, 8)), k=5)
    assert m["precision"] == 1.0 and m["coverage"] < 0.5 * full["coverage"] and m["recall"] < full["recall"]


def test_diversity_and_auc():
    same = np.tile(np.array([[1.0, 0.0]]), (5, 1))
    assert nb.diversity(same) == pytest.approx(0.0) and nb.diversity(same[:1]) is None
    assert nb.auc([2, 3], [0, 1]) == 1.0 and nb.auc([1], [1]) == 0.5 and nb.auc([], [1]) is None


def test_discriminator_auc_separates_shifted_sets_not_same_distribution():
    rng = np.random.default_rng(2)
    real = rng.normal(size=(80, 16))
    assert nb.discriminator_auc(real, rng.normal(size=(80, 16)) + 1.0) > 0.95
    assert 0.3 < nb.discriminator_auc(real, rng.normal(size=(80, 16))) < 0.7
    assert nb.discriminator_auc(real[:3], real[:3]) is None


def test_sweep_summary_has_baseline_and_every_k_without_a_winner():
    embed = fake_embed()
    reference = [f"t{i % 10}:{i}" for i in range(40)]
    drafts = {0: [f"t0:{i}" for i in range(20)], 4: [f"t{i % 10}:g{i}" for i in range(20)]}
    s = nb.sweep_summary(drafts, reference, embed, seed=0)
    assert set(s) == {"baseline", 0, 4}
    assert {"precision@5", "recall@3", "coverage@10", "density@5", "diversity", "disc_auc"} <= set(s[4])
    assert s[4]["coverage@5"] > s[0]["coverage@5"]  # covering 10 topics beats one
    assert s[4]["diversity"] > s[0]["diversity"]
    assert "winner" not in str(s)


def test_add_neighbor_columns_and_summarize_k_on_table_rows():
    embed = fake_embed()
    rows = [{"task_id": "a", "k": 2, "question": "car:a", "exemplars_shown": ["car:a", "dog:x"]},
            {"task_id": "b", "k": 2, "question": "fish:b", "exemplars_shown": ["tree:z"]}]
    reference = [{"question": f"t{i % 5}:{i}"} for i in range(20)]
    out = nb.add_neighbor_columns(rows, reference, embed=embed)
    assert [r["task_id"] for r in out] == ["a", "b"] and out[0]["question"] == "car:a"  # original columns kept
    assert out[0]["nn_sim_shown"] == pytest.approx(1.0) and out[1]["nn_sim_shown"] < 0.9
    s = nb.summarize_k(out, reference, embed=embed)
    assert {"qgen/precision@5", "qgen/baseline/coverage@5", "qgen/disc_auc", "qgen/nn_sim_shown_mean"} <= set(s)
    assert "nn_sim_shown" not in rows[0]  # input rows untouched
