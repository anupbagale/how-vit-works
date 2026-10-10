"""
Synthetic brain-MRI-like slices ("phantoms").

Why synthetic? It needs no patient data, no licence, and no download, so anyone
can clone the repo and reproduce every figure exactly. Each slice has a skull,
brain tissue, two ventricles, mild scanner noise and intensity bias, and
(optionally) one bright round lesion whose position and size are random.
"""
import numpy as np


def _ellipse(yy, xx, cy, cx, ry, rx, angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    y, x = yy - cy, xx - cx
    u = (x * c + y * s) / rx
    v = (-x * s + y * c) / ry
    return u ** 2 + v ** 2 <= 1.0


def _smooth(img, sigma):
    """Small separable Gaussian blur so edges look like a scan, not a vector drawing."""
    r = int(3 * sigma) + 1
    t = np.arange(-r, r + 1)
    g = np.exp(-(t ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    img = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 0, img)
    img = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 1, img)
    return img


def make_slice(size=96, lesion=True, rng=None):
    """Return (image in [0, 1], lesion info dict or None)."""
    rng = np.random.default_rng() if rng is None else rng
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    c = size / 2 + rng.normal(0, 1.0, 2)
    tilt = rng.normal(0, 0.06)
    ry, rx = size * 0.43 * rng.uniform(0.96, 1.04), size * 0.35 * rng.uniform(0.96, 1.04)

    img = np.zeros((size, size))
    head = _ellipse(yy, xx, c[0], c[1], ry, rx, tilt)
    brain = _ellipse(yy, xx, c[0], c[1], ry - 4, rx - 4, tilt)
    img[head] = 0.85                      # skull / scalp
    img[brain] = 0.45                     # brain tissue

    # Two ventricles: dark, slightly asymmetric, like CSF on a T1 scan
    for side in (-1, 1):
        vy = c[0] + rng.normal(-2, 0.8)
        vx = c[1] + side * size * 0.07 + rng.normal(0, 0.6)
        img[_ellipse(yy, xx, vy, vx, size * 0.13, size * 0.035, side * 0.25 + tilt)] = 0.15

    info = None
    if lesion:
        # Place the lesion somewhere inside brain tissue, away from the skull
        for _ in range(100):
            ly = c[0] + rng.uniform(-0.65, 0.65) * (ry - 8)
            lx = c[1] + rng.uniform(-0.65, 0.65) * (rx - 8)
            if _ellipse(np.array(ly), np.array(lx), c[0], c[1], ry - 10, rx - 10, tilt):
                break
        lr = rng.uniform(3.0, 5.5)
        img[_ellipse(yy, xx, ly, lx, lr, lr * rng.uniform(0.8, 1.2))] = rng.uniform(0.75, 0.9)
        info = {"row": ly, "col": lx, "radius": lr}

    img = _smooth(img, 0.9)
    bias = 1.0 + 0.08 * ((xx - c[1]) / size) * rng.normal(1, 0.3)   # coil bias field
    img = img * bias + rng.normal(0, 0.025, img.shape)                # scanner noise
    return np.clip(img, 0, 1), info


def make_dataset(n, size=96, seed=0):
    """Balanced set of slices with (1) and without (0) a lesion."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 2
    rng.shuffle(labels)
    images = np.stack([make_slice(size, bool(l), rng)[0] for l in labels])
    return images, labels
