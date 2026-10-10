"""
Follow ONE image through a Vision Transformer and save a figure for every step.

    python walkthrough.py

Needs only NumPy, Matplotlib and Pillow: it loads the trained weights from
weights/vit_weights.npz and runs the forward pass in vit/numpy_vit.py.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

from vit.numpy_vit import (forward, load_weights, attention_rollout, softmax,
                           PATCH, GRID, DIM, HEADS, DEPTH, CLASSES)
from vit.phantom import make_slice, make_dataset

ROOT = Path(__file__).parent
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
EXAMPLE_SEED = 6          # the same slice used in the CNN walkthrough

# ---- one visual language (matches the companion CNN repo) ---------------------
SURFACE, INK, INK2, MUTED, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"                       # normal, lesion
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f4f3f0", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div", ["#8f2323", "#e34948", "#f2a8a3", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
OVL = LinearSegmentedColormap.from_list(                  # attention overlay: visible on black and gray
    "ovl", ["#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRIDC, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
})


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRIDC)


def title(ax, t, sub=None):
    ax.set_title(t, fontsize=10.5, fontweight="bold", pad=4)
    if sub:
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK2)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  saved assets/{name}")


def overlay(ax, img, amap, vmax=None, alpha=0.8):
    """Show the image in gray with an 8x8 attention map laid over it in blue."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    a = np.kron(amap, np.ones((PATCH, PATCH)))
    a = a / (vmax or a.max())                    # strongest patch = fully lit
    a = np.clip(a, 0, 1)
    rgba = OVL(a)
    rgba[..., 3] = np.where(a < 0.12, 0, a ** 1.3 * alpha)
    ax.imshow(rgba)
    clean(ax)


def grid_lines(ax, color="white", lw=0.8, alpha=0.6):
    for t in range(1, GRID):
        ax.axhline(t * PATCH - .5, color=color, lw=lw, alpha=alpha)
        ax.axvline(t * PATCH - .5, color=color, lw=lw, alpha=alpha)


def patch_box(ax, idx, color=ORANGE, lw=2.2, **kw):
    r, c = divmod(idx, GRID)
    ax.add_patch(Rectangle((c * PATCH - .5, r * PATCH - .5), PATCH, PATCH, fill=False, ec=color, lw=lw, **kw))


def number_grid(ax, values, fmt="{:.2f}", cmap=None, vmin=None, vmax=None, fontsize=9):
    n, m = values.shape
    ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(n):
        for j in range(m):
            r, g, b, _ = cmap(np.clip((values[i, j] - vmin) / (vmax - vmin + 1e-9), 0, 1))
            col = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=fontsize, color=col)
    ax.set_xticks(np.arange(-.5, m), minor=True); ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def prob_bars(ax, probs, ttl="prediction"):
    ys = [1, 0]
    ax.barh(ys, probs, color=[BLUE, ORANGE], height=0.55)
    ax.barh(ys, [1, 1], color="none", edgecolor=GRIDC, height=0.55)
    for y, p, name in zip(ys, probs, CLASSES):
        ax.text(-0.04, y, name, ha="right", va="center", fontsize=10)
        ax.text(1.04, y, f"{p:.1%}", ha="left", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.42, 1.36); ax.set_ylim(-0.6, 1.6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if ttl:
        ax.set_title(ttl, fontsize=10.5, fontweight="bold", pad=4)


# ==============================================================================
print("1/3  loading weights and checking accuracy on 400 held-out slices ...")
P = load_weights(ROOT / "weights" / "vit_weights.npz")
X_test, y_test = make_dataset(400, seed=2)
pred = np.array([forward(x, P)["logits"].argmax() for x in X_test])
test_acc = (pred == y_test).mean()
n_params = sum(v.size for v in P.values())
print(f"     {n_params:,} parameters · test accuracy {test_acc:.1%}")

print("2/3  passing the one image through the ViT ...")
img, lesion = make_slice(96, lesion=True, rng=np.random.default_rng(EXAMPLE_SEED))
twin, _ = make_slice(96, lesion=False, rng=np.random.default_rng(EXAMPLE_SEED))
s, s_twin = forward(img, P), forward(twin, P)
ly, lx, lr = lesion["row"], lesion["col"], lesion["radius"]
L = int(ly // PATCH) * GRID + int(lx // PATCH)          # index of the patch holding the lesion
for k in ["patches", "patch_tokens", "tokens"]:
    print(f"     {k:13s} {s[k].shape}")
print(f"     attention    {DEPTH} layers x {s['attn'][0].shape}")
print(f"     P(lesion) = {s['probs'][1]:.3f}  (same slice without lesion: {s_twin['probs'][1]:.3f})")
roll, roll_twin = attention_rollout(s["attn"]), attention_rollout(s_twin["attn"])

print("3/3  drawing figures ...")

# ---- HERO ----------------------------------------------------------------------
fig = plt.figure(figsize=(17, 5.0))
H, y0 = 0.56, 0.2
sq = H * 5.0 / 17
stages = [("Input", "1 × 96 × 96", "an MRI-like slice", sq),
          ("Cut into patches", "64 × (12 × 12)", "an image becomes a 'sentence'", sq),
          ("Embed + position", f"65 × {DIM}", f"each patch → a vector of {DIM} numbers", sq * 0.75),
          ("Self-attention", f"{HEADS} heads", "every patch looks at every patch", sq),
          ("Read [CLS]", f"{DIM} → 2", "the summary token decides", sq * 0.95)]
gap = (1 - 0.03 - sum(st[3] for st in stages)) / (len(stages) - 1)
x, boxes = 0.015, []
for i, (t, shape, expl, w) in enumerate(stages):
    boxes.append((x, w))
    fig.text(x + w / 2, y0 + H + 0.13, f"{i + 1}  {t}", ha="center", fontsize=12.5, fontweight="bold")
    fig.text(x + w / 2, y0 + H + 0.065, shape, ha="center", fontsize=10, color=BLUE, family="DejaVu Sans Mono")
    fig.text(x + w / 2, y0 - 0.075, expl, ha="center", fontsize=9.5, color=INK2)
    x += w + gap
ax = fig.add_axes([boxes[0][0], y0, boxes[0][1], H]); ax.imshow(img, cmap="gray", vmin=0, vmax=1); clean(ax)
# exploded patches
xb, wb = boxes[1]
gp = 0.0025
cw = (wb - gp * (GRID - 1)) / GRID
chh = (H - gp * 17 / 5 * (GRID - 1)) / GRID
for i in range(GRID * GRID):
    r, c = divmod(i, GRID)
    a = fig.add_axes([xb + c * (cw + gp), y0 + H - (r + 1) * chh - r * gp * 17 / 5, cw, chh])
    a.imshow(s["patches_raw"][i].reshape(PATCH, PATCH), cmap="gray", vmin=0, vmax=1)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(ORANGE if i == L else SURFACE); sp.set_linewidth(2 if i == L else 0.5)
ax = fig.add_axes([boxes[2][0], y0, boxes[2][1], H])
tok = s["tokens"]
ax.imshow(tok, cmap=DIV, vmin=-np.abs(tok).max(), vmax=np.abs(tok).max(), aspect="auto", interpolation="nearest")
ax.axhline(0.5, color=INK, lw=1)
ax.text(-1.2, 0, "[CLS]", ha="right", va="center", fontsize=8, color=INK2)
ax.text(-1.2, L + 1, "lesion →", ha="right", va="center", fontsize=8, color=ORANGE)
clean(ax)
ax = fig.add_axes([boxes[3][0], y0, boxes[3][1], H]); overlay(ax, img, roll)
ax = fig.add_axes([boxes[4][0], y0 + H * 0.2, boxes[4][1], H * 0.6]); prob_bars(ax, s["probs"], ttl="")
for (xa, wa), (xc, _) in zip(boxes[:-1], boxes[1:]):
    fig.patches.append(FancyArrowPatch((xa + wa + 0.006, y0 + H / 2), (xc - 0.006, y0 + H / 2),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
                                       color=MUTED, lw=1.4))
save(fig, "hero.png")

# ---- STEP 1: patches ----------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.9]})
axs[0].imshow(img, cmap="gray", vmin=0, vmax=1); clean(axs[0]); grid_lines(axs[0])
for i in range(GRID * GRID):
    r, c = divmod(i, GRID)
    axs[0].text(c * PATCH + 1, r * PATCH + 1, str(i), color="white", fontsize=6.5, va="top", alpha=0.85)
patch_box(axs[0], L)
title(axs[0], "Cut the image into an 8 × 8 grid", "each square is one 12 × 12 patch")
axs[1].axis("off")
title(axs[1], "…and read the patches left-to-right, top-to-bottom, like words in a sentence",
      "64 'words', each one made of 144 pixel values")
for i in range(GRID * GRID):
    r, c = divmod(i, 16)
    a = axs[1].inset_axes([c / 16 + 0.004, 1 - (r + 1) / 4 + 0.02, 1 / 16 - 0.008, 1 / 4 - 0.07])
    a.imshow(s["patches_raw"][i].reshape(PATCH, PATCH), cmap="gray", vmin=0, vmax=1)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(ORANGE if i == L else GRIDC); sp.set_linewidth(2.2 if i == L else 0.6)
    a.text(0.5, -0.06, str(i), transform=a.transAxes, ha="center", va="top", fontsize=7,
           color=ORANGE if i == L else MUTED, fontweight="bold" if i == L else "normal")
save(fig, "01_patches.png")

# ---- STEP 2: patch embedding --------------------------------------------------------
W = P["embed.weight"]                                    # (32, 144)
fig = plt.figure(figsize=(14, 7.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], width_ratios=[0.8, 1.6, 0.35, 0.8], hspace=0.35, wspace=0.25)
a = fig.add_subplot(gs[0, 0]); a.imshow(s["patches_raw"][L].reshape(PATCH, PATCH), cmap="gray", vmin=0, vmax=1)
clean(a); [sp.set_color(ORANGE) for sp in a.spines.values()]
title(a, f"patch {L} (the lesion)", "12 × 12 = 144 numbers")
a = fig.add_subplot(gs[0, 1]); wv = np.abs(W).max()
a.imshow(W, cmap=DIV, vmin=-wv, vmax=wv, aspect="auto", interpolation="nearest"); clean(a)
title(a, f"× learned matrix W  ({DIM} × 144)", "each row is one 'question' asked of the patch")
a = fig.add_subplot(gs[0, 2]); t = s["patch_tokens"][L][:, None]; tv = np.abs(t).max()
a.imshow(t, cmap=DIV, vmin=-tv, vmax=tv, aspect="auto", interpolation="nearest"); clean(a)
title(a, "= token", f"{DIM} numbers")
a = fig.add_subplot(gs[0, 3]); a.axis("off")
a.text(0.0, 0.62, f"One matrix multiply\nturns 144 pixels\ninto {DIM} features.", fontsize=11, va="center")
a.text(0.0, 0.25, "The same W is used\nfor all 64 patches.", fontsize=10, color=INK2, va="center")
sub = fig.add_gridspec(2, 8, left=0.12, right=0.88, bottom=0.04, top=0.43, wspace=0.08, hspace=0.12)
order = np.argsort(-np.abs(W).sum(1))[:16]
for n, r in enumerate(order):
    a = fig.add_subplot(sub[n // 8, n % 8])
    f = W[r].reshape(PATCH, PATCH); fv = np.abs(f).max()
    a.imshow(f, cmap=DIV, vmin=-fv, vmax=fv, interpolation="nearest"); clean(a)
fig.text(0.5, 0.465, "16 rows of W, reshaped back to 12 × 12: what each feature is looking for",
         ha="center", fontsize=10.5, fontweight="bold")
fig.text(0.5, -0.02, "Blue = bright pixels raise the feature, red = bright pixels lower it. "
         "This is exactly a 12 × 12 convolution with stride 12: a ViT starts with one CNN-like layer.\n"
         "Learned from only 6,000 small images, so they look noisy. ViTs trained on millions of images learn smooth, edge-like filters.",
         ha="center", va="top", fontsize=9.5, color=INK2, linespacing=1.6)
save(fig, "02_patch_embedding.png")

# ---- STEP 3: position embeddings -----------------------------------------------------
pos = P["pos"][1:]
pn = pos / np.linalg.norm(pos, axis=1, keepdims=True)
sim = pn @ pn.T
fig, axs = plt.subplots(GRID, GRID, figsize=(8.4, 8.8))
for i in range(GRID * GRID):
    a = axs[divmod(i, GRID)]
    a.imshow(sim[i].reshape(GRID, GRID), cmap=DIV, vmin=-1, vmax=1, interpolation="nearest")
    clean(a)
    r, c = divmod(i, GRID)
    a.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False, ec=ORANGE, lw=1.2))
fig.suptitle("Learned position embeddings: how similar is each position to every other?",
             fontsize=11.5, fontweight="bold", y=0.95)
fig.text(0.5, 0.06, "Each small square is one patch position (orange). Blue = similar position code, red = opposite.\n"
         "Without these, the transformer would see a bag of patches with no idea where each one came from.\n"
         "With little training data the pattern is only partly smooth: neighbours tend to be similar, but not always.",
         ha="center", fontsize=9.5, color=INK2, linespacing=1.5)
fig.subplots_adjust(wspace=0.08, hspace=0.08, top=0.92, bottom=0.14)
save(fig, "03_position_embeddings.png")

# ---- STEP 4: attention, worked by hand --------------------------------------------------
layer = DEPTH - 1
A_last = s["attn"][layer][:, 0, 1:]                         # (heads, 64): [CLS] -> patches
head = int(A_last[:, L].argmax())                           # the head that cares most about the lesion
q = s["qkv"][layer]["q"][head, 0]
K = s["qkv"][layer]["k"][head, 1:]
scores = K @ q / np.sqrt(DIM // HEADS)
weights = s["attn"][layer][head, 0, 1:]
low = int(np.argmin(np.where(np.arange(64) == L, np.inf, weights)))
mid = int(np.argsort(weights)[len(weights) // 2])
ex = [L, int(np.argsort(-weights)[1]) if np.argsort(-weights)[0] == L else int(np.argsort(-weights)[0]), mid, low]
fig = plt.figure(figsize=(14, 6.2))
gs = fig.add_gridspec(len(ex) + 1, 6, width_ratios=[0.55, 1.7, 0.55, 0.55, 0.3, 2.1], hspace=0.35, wspace=0.25)
for j, ttxt in enumerate(["patch", f"key k  ({DIM // HEADS} numbers, blue + / red −)", f"q · k / √{DIM // HEADS}", "softmax"]):
    a = fig.add_subplot(gs[0, j]); a.axis("off")
    a.text(0.5, 0.2, ttxt, ha="center", fontsize=10.5, fontweight="bold")
kv = np.abs(K).max()
for row, idx in enumerate(ex, start=1):
    a = fig.add_subplot(gs[row, 0]); a.imshow(s["patches_raw"][idx].reshape(PATCH, PATCH), cmap="gray", vmin=0, vmax=1)
    clean(a); [sp.set_color(ORANGE if idx == L else GRIDC) for sp in a.spines.values()]
    a.set_ylabel(f"#{idx}", rotation=0, labelpad=14, fontsize=9, color=INK2, va="center")
    a = fig.add_subplot(gs[row, 1]); a.imshow(K[idx][None], cmap=DIV, vmin=-kv, vmax=kv, aspect="auto")
    a.set_ylim(0.9, -0.9); clean(a)
    for j, (val, fmt) in enumerate([(scores[idx], "{:+.2f}"), (weights[idx], "{:.1%}")]):
        a = fig.add_subplot(gs[row, 2 + j]); a.axis("off")
        a.text(0.5, 0.5, fmt.format(val), ha="center", va="center", fontsize=13,
               fontweight="bold" if j == 1 else "normal", color=BLUE if j == 1 else INK)
a = fig.add_subplot(gs[1:, 5]); overlay(a, img, weights.reshape(GRID, GRID))
for idx in ex:
    patch_box(a, idx, color=ORANGE if idx == L else INK2, lw=1.6 if idx != L else 2.2)
title(a, f"all 64 weights (layer {layer + 1}, head {head + 1})", "they add up to 100%")
a = fig.add_subplot(gs[0, 5]); a.axis("off")
qv = np.abs(q).max()
ia = a.inset_axes([0.05, 0.12, 0.9, 0.3]); ia.imshow(q[None], cmap=DIV, vmin=-qv, vmax=qv, aspect="auto"); clean(ia)
a.text(0.5, 0.75, "query q of [CLS]: \"what am I looking for?\"", ha="center", fontsize=10, fontweight="bold")
save(fig, "04_attention_by_hand.png")

# ---- STEP 5: every head, then combined ----------------------------------------------------
fig, axs = plt.subplots(DEPTH, HEADS + 1, figsize=(15, 3.9 * DEPTH), squeeze=False)
for l in range(DEPTH):
    vm = s["attn"][l][:, 0, 1:].max()                       # same scale for every head
    for h in range(HEADS):
        a = s["attn"][l][h, 0, 1:]
        overlay(axs[l, h], img, a.reshape(GRID, GRID), vmax=vm)
        share = a[L] * 100
        title(axs[l, h], f"head {h + 1}", f"{share:.0f}% of its attention on patch {L}")
overlay(axs[0, HEADS], img, roll)
title(axs[0, HEADS], "all heads combined", "attention rollout")
fig.text(0.5, -0.04, "Each map shows where [CLS] looks in one attention head. "
         "Same brightness scale for all heads. Each head learned its own job: one locks onto the lesion, one spreads thin.",
         ha="center", fontsize=9.5, color=INK2)
save(fig, "05_attention_heads.png")

# ---- STEP 6: animation, any patch can look anywhere --------------------------------------
print("     rendering the attention animation ...")
A1 = s["attn"][0].mean(0)[1:, 1:]                           # layer 1, patch -> patch
frames = []
fig, axs = plt.subplots(1, 2, figsize=(9, 4.4))
order = list(range(GRID * GRID))
for i in order:
    for a in axs:
        a.clear()
    axs[0].imshow(img, cmap="gray", vmin=0, vmax=1); clean(axs[0]); grid_lines(axs[0], alpha=0.35)
    patch_box(axs[0], i, color=ORANGE, lw=2.6)
    axs[0].set_title(f"query: patch {i}", fontsize=11, fontweight="bold")
    overlay(axs[1], img, A1[i].reshape(GRID, GRID), vmax=A1[i].max())
    patch_box(axs[1], i, color=ORANGE, lw=1.6)
    axs[1].set_title("where it looks (layer 1)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
plt.close(fig)
small = [f.resize((f.width * 2 // 3, f.height * 2 // 3), Image.LANCZOS).quantize(96) for f in frames]
small[0].save(OUT / "06_attention_any_patch.gif", save_all=True, append_images=small[1:],
              duration=[1200 if i == L else 260 for i in order], loop=0, optimize=True)
print("  saved assets/06_attention_any_patch.gif")

# ---- STEP 7: decision, with vs without lesion ---------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(12, 7.4), gridspec_kw={"width_ratios": [1, 1, 0.95]})
vmax = max(roll.max(), roll_twin.max())
for r, (im, st, rl, lab) in enumerate([(img, s, roll, "with lesion"), (twin, s_twin, roll_twin, "without lesion")]):
    axs[r, 0].imshow(im, cmap="gray", vmin=0, vmax=1); clean(axs[r, 0]); title(axs[r, 0], f"input · {lab}")
    overlay(axs[r, 1], im, rl, vmax=vmax); title(axs[r, 1], "where [CLS] looked (rollout)")
    prob_bars(axs[r, 2], st["probs"])
save(fig, "07_with_vs_without.png")

# ---- STEP 8: CNN vs ViT receptive field ----------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(14, 5))
cy, cx = int(ly), int(lx)
for a, size, t, sub in [(axs[0], 3, "CNN, layer 1", "each output sees 3 × 3 pixels"),
                        (axs[1], 22, "CNN, layer 2", "each output sees ≈ 22 × 22 pixels")]:
    a.imshow(img, cmap="gray", vmin=0, vmax=1); clean(a)
    a.add_patch(Rectangle((cx - size / 2 - .5, cy - size / 2 - .5), size, size, fill=False, ec=ORANGE, lw=2.2))
    title(a, t, sub)
overlay(axs[2], img, A1[L].reshape(GRID, GRID)); patch_box(axs[2], L, lw=1.6)
title(axs[2], "ViT, layer 1", "the lesion patch can already look at the whole image")
fig.text(0.5, -0.02, "A CNN grows its view slowly, layer by layer. A ViT starts global, "
         "but has to learn that nearby pixels belong together.", ha="center", fontsize=9.5, color=INK2)
save(fig, "08_cnn_vs_vit.png")

print(f"done · {n_params:,} parameters · test accuracy {test_acc:.1%}")
