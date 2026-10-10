"""
Train the tiny ViT in PyTorch and save its weights for the NumPy walkthrough.

    pip install torch
    python train.py

You only need this if you want to retrain: the trained weights are already in
weights/vit_weights.npz, and walkthrough.py needs only NumPy.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit.phantom import make_dataset
from vit.numpy_vit import PATCH, GRID, DIM, HEADS, DEPTH, PIXEL_MEAN, PIXEL_STD

torch.manual_seed(0)
MLP = 2 * DIM


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.qkv, self.proj = nn.Linear(DIM, 3 * DIM), nn.Linear(DIM, DIM)
        self.fc1, self.fc2 = nn.Linear(DIM, MLP), nn.Linear(MLP, DIM)

    def forward(self, x):
        B, T, D = x.shape
        d = D // HEADS
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (m.view(B, T, HEADS, d).transpose(1, 2) for m in (q, k, v))
        attn = (q @ k.transpose(-2, -1) / math.sqrt(d)).softmax(-1)
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(PATCH * PATCH, DIM)
        self.cls = nn.Parameter(torch.zeros(1, DIM))
        self.pos = nn.Parameter(torch.randn(GRID * GRID + 1, DIM) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, 2)

    def forward(self, img):                                   # img: (B, 96, 96)
        B = img.shape[0]
        img = (img - PIXEL_MEAN) / PIXEL_STD
        p = img.reshape(B, GRID, PATCH, GRID, PATCH).permute(0, 1, 3, 2, 4).reshape(B, GRID * GRID, -1)
        x = torch.cat([self.cls.expand(B, 1, DIM), self.embed(p)], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))


def main(n_train=6000, epochs=30, batch=64):
    print("generating data ...")
    Xtr, ytr = make_dataset(n_train, seed=11)
    Xte, yte = make_dataset(400, seed=2)                       # same test seed as the CNN repo
    Xtr, ytr = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    Xte, yte = torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte)

    model = TinyViT()
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = epochs * math.ceil(n_train / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)   # slow down smoothly
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total = 0.0
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        history.append((total / n_train, acc))
        print(f"epoch {ep + 1:2d}  loss {total / n_train:.4f}  test acc {acc:.1%}")

    out = Path(__file__).parent / "weights"
    out.mkdir(exist_ok=True)
    np.savez(out / "vit_weights.npz", **{k: v.detach().numpy() for k, v in model.state_dict().items()})
    np.savetxt(out / "training_log.csv", np.array(history), delimiter=",",
               header="train_loss,test_accuracy", comments="")
    print("saved weights/vit_weights.npz")


if __name__ == "__main__":
    main()
