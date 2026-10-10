"""Check the NumPy ViT piece by piece, and as a whole against PyTorch."""
from pathlib import Path

import numpy as np
import pytest

from vit.numpy_vit import (patchify, layer_norm, softmax, multi_head_attention, forward,
                           load_weights, attention_rollout)
from vit.phantom import make_slice

rng = np.random.default_rng(0)
WEIGHTS = Path(__file__).parent.parent / "weights" / "vit_weights.npz"


def test_patchify_keeps_every_pixel_in_place():
    img = np.arange(96 * 96, dtype=float).reshape(96, 96)
    p = patchify(img)
    assert p.shape == (64, 144)
    # patch 9 is row 1, column 1 of the 8 x 8 grid
    assert np.array_equal(p[9].reshape(12, 12), img[12:24, 12:24])


def test_attention_matches_loops():
    T, D, H = 5, 8, 2
    x = rng.normal(size=(T, D))
    Wqkv, bqkv = rng.normal(size=(3 * D, D)), rng.normal(size=3 * D)
    Wo, bo = rng.normal(size=(D, D)), rng.normal(size=D)
    out, attn, _ = multi_head_attention(x, Wqkv, bqkv, Wo, bo, heads=H)

    d = D // H
    q, k, v = np.split(x @ Wqkv.T + bqkv, 3, axis=-1)
    concat = np.zeros((T, D))
    for h in range(H):
        sl = slice(h * d, (h + 1) * d)
        for i in range(T):
            s = np.array([q[i, sl] @ k[j, sl] for j in range(T)]) / np.sqrt(d)
            w = np.exp(s - s.max()); w /= w.sum()
            assert np.allclose(attn[h, i], w)
            concat[i, sl] = sum(w[j] * v[j, sl] for j in range(T))
    assert np.allclose(out, concat @ Wo.T + bo)


def test_layer_norm_and_softmax():
    y = layer_norm(rng.normal(3, 5, size=(4, 16)), np.ones(16), np.zeros(16))
    assert np.allclose(y.mean(-1), 0, atol=1e-6) and np.allclose(y.std(-1), 1, atol=1e-3)
    assert np.allclose(softmax(rng.normal(size=(3, 7))).sum(-1), 1)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="no trained weights")
def test_rollout_is_a_distribution():
    s = forward(make_slice(rng=np.random.default_rng(1))[0], load_weights(WEIGHTS))
    r = attention_rollout(s["attn"])
    assert r.shape == (8, 8) and np.all(r >= 0)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="no trained weights")
def test_numpy_matches_pytorch():
    torch = pytest.importorskip("torch")
    from train import TinyViT

    P = load_weights(WEIGHTS)
    model = TinyViT()
    model.load_state_dict({k: torch.tensor(v) for k, v in P.items()})
    model.eval()
    for seed in range(3):
        img = make_slice(rng=np.random.default_rng(seed))[0]
        ours = forward(img, P)["logits"]
        with torch.no_grad():
            ref = model(torch.tensor(img[None], dtype=torch.float32))[0].numpy()
        assert np.allclose(ours, ref, atol=1e-4), (ours, ref)
