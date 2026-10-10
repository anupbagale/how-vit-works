"""
A Vision Transformer forward pass in plain NumPy.

Every operation a ViT performs is written out here, in order, with nothing hidden
inside a framework. The weights come from `weights/vit_weights.npz` (trained by
`train.py`), and `tests/` checks this file gives the same answer as PyTorch.

Shapes:  N = 64 patches, T = 65 tokens (64 patches + 1 [CLS]), D = 64 dims,
         H = 4 attention heads, d = D / H = 16 dims per head, 1 transformer block.
"""
import numpy as np

PATCH = 12          # a 96 x 96 image becomes an 8 x 8 grid of 12 x 12 patches
GRID = 8
DIM = 64
HEADS = 4
DEPTH = 1
CLASSES = ["normal", "lesion"]
PIXEL_MEAN, PIXEL_STD = 0.25, 0.26   # mean and spread of pixel values in the training set


# ---- the individual operations ----------------------------------------------
def patchify(img, p=PATCH):
    """(96, 96) image -> (64, 144): cut into p x p tiles, flatten each, raster order."""
    h, w = img.shape
    g = h // p
    return img.reshape(g, p, g, p).transpose(0, 2, 1, 3).reshape(g * g, p * p)


def linear(x, W, b):
    """Same convention as torch.nn.Linear: W has shape (out, in)."""
    return x @ W.T + b


def layer_norm(x, gamma, beta, eps=1e-5):
    """Rescale each token to mean 0, variance 1, then apply a learned scale and shift."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def gelu(x):
    """Smooth version of ReLU (tanh approximation, as used in GPT-2 and many ViTs)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def multi_head_attention(x, Wqkv, bqkv, Wo, bo, heads=HEADS):
    """
    x: (T, D). Every token asks a question (query), advertises what it contains
    (key) and offers some information (value). Attention = softmax(Q K^T / sqrt(d)).
    Returns the output (T, D) and the attention weights (heads, T, T).
    """
    T, D = x.shape
    d = D // heads
    qkv = linear(x, Wqkv, bqkv)                                # (T, 3D)
    q, k, v = np.split(qkv, 3, axis=-1)                        # each (T, D)
    q, k, v = (m.reshape(T, heads, d).transpose(1, 0, 2) for m in (q, k, v))   # (H, T, d)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d)             # (H, T, T)
    attn = softmax(scores, axis=-1)                            # each row sums to 1
    out = (attn @ v).transpose(1, 0, 2).reshape(T, D)          # concatenate heads
    return linear(out, Wo, bo), attn, {"q": q, "k": k, "v": v, "scores": scores}


# ---- the whole model ------------------------------------------------------------
def forward(img, P):
    """Run one image through the ViT and keep every intermediate for the figures."""
    x = (img - PIXEL_MEAN) / PIXEL_STD                         # centre pixels around 0
    s = {"patches": patchify(x), "patches_raw": patchify(img)}  # (64, 144) each
    s["patch_tokens"] = linear(s["patches"], P["embed.weight"], P["embed.bias"])   # (64, 64)
    x = np.concatenate([P["cls"], s["patch_tokens"]], axis=0)  # (65, 64): [CLS] goes first
    s["tokens_no_pos"] = x
    x = x + P["pos"]                                           # add "where am I" information
    s["tokens"] = x
    s["attn"], s["qkv"], s["blocks"] = [], [], [x]
    for i in range(DEPTH):
        b = f"blocks.{i}."
        h = layer_norm(x, P[b + "ln1.weight"], P[b + "ln1.bias"])
        a, attn, qkv = multi_head_attention(h, P[b + "qkv.weight"], P[b + "qkv.bias"],
                                            P[b + "proj.weight"], P[b + "proj.bias"])
        x = x + a                                              # residual connection
        h = layer_norm(x, P[b + "ln2.weight"], P[b + "ln2.bias"])
        h = linear(gelu(linear(h, P[b + "fc1.weight"], P[b + "fc1.bias"])),
                   P[b + "fc2.weight"], P[b + "fc2.bias"])
        x = x + h                                              # residual connection
        s["attn"].append(attn)
        s["qkv"].append(qkv)
        s["blocks"].append(x)
    cls = layer_norm(x[0], P["norm.weight"], P["norm.bias"])   # only [CLS] is read out
    s["cls"] = cls
    s["logits"] = linear(cls, P["head.weight"], P["head.bias"])
    s["probs"] = softmax(s["logits"])
    return s


def attention_rollout(attn_layers):
    """
    Combine attention across layers (Abnar & Zuidema, 2020): average the heads,
    add the identity for the residual path, renormalise, multiply layer by layer.
    Row 0 then says how much [CLS] ultimately draws from each patch. Returns (8, 8).
    """
    T = attn_layers[0].shape[-1]
    R = np.eye(T)
    for A in attn_layers:
        A = A.mean(0) + np.eye(T)
        A = A / A.sum(-1, keepdims=True)
        R = A @ R
    return R[0, 1:].reshape(GRID, GRID)


def load_weights(path):
    return dict(np.load(path))
