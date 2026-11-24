"""HIPPO's step 1: contrastive clustering of H&E tiles, trained from scratch.

The original trains a ResNet50 for 100 epochs on 227k tiles with a loss that
combines a tile-level and a cluster-level contrastive term, and reads out 64
clusters by k-means on the learned features.  No pretrained weights are involved
in HIPPO either, which is convenient here: download.pytorch.org and HuggingFace
are both unreachable from this server, so a from-scratch encoder is not a
compromise, it is the method.

Differences from the original, stated plainly:
  * a small CNN instead of ResNet50 -- ResNet50 on ~4k tiles would be badly
    overparameterised, and HIPPO had 227k
  * NT-Xent at the tile level (theirs), plus a prototype contrastive term at the
    cluster level standing in for their cluster loss
  * these tiles come from 3 slides, not 209
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def build_encoder(width=32, out_dim=128):
    import torch.nn as nn

    def block(i, o, stride=2):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, stride, 1, bias=False),
            nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, 1, 1, bias=False),
            nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    return nn.Sequential(
        block(3, width),            # 224 -> 112
        block(width, width * 2),    # 112 -> 56
        block(width * 2, width * 4),   # 56 -> 28
        block(width * 4, width * 8),   # 28 -> 14
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(width * 8, out_dim))


def augment(x, rng, jitter=0.25):
    """Random crop, flip and colour jitter -- the two-view machinery.

    Tiles arrive as NHWC uint8 from the npz and are converted to NCHW float here;
    the earlier version applied NCHW-shaped jitter to NHWC tensors, which is the
    shape error this fixes.

    The crop is a single batched `affine_grid` + `grid_sample`, not a Python loop
    of `F.interpolate` calls.  The loop version launched one kernel per sample,
    twice per step, and measured 6 s per step on a batch of 32 on an RTX 4090 --
    slower per tile than the CPU path it replaced.  One kernel per batch keeps
    the GPU busy.
    """
    import torch
    import torch.nn.functional as F

    x = x.permute(0, 3, 1, 2).float()
    n, c, h, w = x.shape
    dev = x.device

    crop = rng.uniform(0.60, 1.0, size=n).astype(np.float32)
    flip = rng.random(n) < 0.5
    # a crop of side `crop` must stay inside the tile, so its centre can only
    # move within +-(1 - crop) in normalised coordinates
    room = 1.0 - crop
    ox = rng.uniform(-1.0, 1.0, size=n).astype(np.float32) * room
    oy = rng.uniform(-1.0, 1.0, size=n).astype(np.float32) * room

    theta = np.zeros((n, 2, 3), dtype=np.float32)
    theta[:, 0, 0] = np.where(flip, -crop, crop)   # negative scale = mirror
    theta[:, 1, 1] = crop
    theta[:, 0, 2] = ox
    theta[:, 1, 2] = oy

    grid = F.affine_grid(torch.from_numpy(theta).to(dev), x.shape,
                         align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)

    # colour jitter, one broadcast each
    gain = torch.as_tensor(
        rng.uniform(1.0 - jitter, 1.0 + jitter, size=(n, c, 1, 1)),
        dtype=out.dtype, device=dev)
    bias = torch.as_tensor(
        rng.uniform(-20, 20, size=(n, 1, 1, 1)), dtype=out.dtype, device=dev)
    return (out * gain + bias).clamp(0, 255)


def nt_xent(z1, z2, temperature=0.2):
    """HIPPO's tile-level contrastive term."""
    import torch
    import torch.nn.functional as F

    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature
    n = z1.shape[0]
    # positive pairs are (i, i+n) and (i+n, i); mask the diagonal
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def prototype_loss(feats, protos, temperature=0.1):
    """Cluster-level term: a tile should be closer to its own prototype than to
    any other, under a sharpened soft assignment."""
    import torch
    import torch.nn.functional as F

    z = F.normalize(feats, dim=1)
    p = F.normalize(protos, dim=1)
    logits = z @ p.t() / temperature
    with torch.no_grad():
        tgt = logits.argmax(dim=1)
        # sharpen so the target is a distribution, not a one-hot
        t = torch.softmax(logits.detach() / 0.05, dim=1)
        t = t / t.sum(dim=1, keepdim=True)
    return -(t * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clusters", type=int, default=64)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto",
                    help="'auto' picks the GPU with the most free memory; "
                         "'cpu' is an explicit opt-in to the slow path")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    # Device selection must be loud.  An earlier run of this script on 10,052
    # tiles fell back to the CPU because torch.cuda.is_available() came back
    # False, and the `if ... else "cpu"` turned that into a silent 4x slowdown
    # that was only noticed 45 minutes in.  Refuse to run on the CPU unless the
    # caller asked for it explicitly.
    req = a.device
    if req in ("auto", "") and not torch.cuda.is_available():
        raise SystemExit(
            "refusing to train on the CPU: 'auto' found no usable CUDA device "
            f"(torch {torch.__version__}, cuda build {torch.version.cuda}). "
            "Pass --device cpu to override.")
    if req.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "refusing to train on the CPU: cuda is not available "
            f"(torch {torch.__version__}, cuda build {torch.version.cuda}). "
            "Pass --device cpu to override.")
    if req == "auto":
        # pick the GPU with the most free memory: this box is shared and two
        # other jobs were holding ~6.5 GB of GPU 0 when this was first run
        best, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"  gpu{i}: {free/1e9:.1f} GB free of {total/1e9:.1f} GB",
                  flush=True)
            if free > best_free:
                best, best_free = i, free
        req = f"cuda:{best}"
    dev = torch.device(req)
    print(f"  torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"available {torch.cuda.is_available()}", flush=True)
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        print(f"  device {dev} = {p.name}, {p.total_memory/1e9:.1f} GB",
              flush=True)
        # a real allocation, not just a capability query: if this succeeds the
        # training loop will be on the GPU
        _probe = torch.zeros(1024, device=dev)
        _probe += 1.0
        print(f"  cuda allocation probe ok ({float(_probe.sum()):.0f})",
              flush=True)
    else:
        print("  WARNING training on the CPU; this is much slower", flush=True)

    imgs = []
    for p in a.tiles:
        z = np.load(p)
        imgs.append(z["img"])
        print(f"  {os.path.basename(p)}: {z['img'].shape}")
    X = np.concatenate(imgs, axis=0)
    print(f"tiles {X.shape}  ({X.nbytes/1e9:.2f} GB raw)")

    enc = build_encoder(a.width).to(dev)
    n_par = sum(p.numel() for p in enc.parameters())
    print(f"encoder params {n_par/1e6:.2f}M  device {dev}")

    protos = torch.nn.Parameter(
        F.normalize(torch.randn(a.clusters, 128, device=dev), dim=1))
    opt = torch.optim.AdamW(list(enc.parameters()) + [protos], lr=a.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * max(1, len(X) // a.batch),
        pct_start=0.15)

    Xt = torch.from_numpy(X).to(torch.float32)
    n = len(Xt)
    steps = max(1, n // a.batch)
    t0 = time.time()
    for ep in range(a.epochs):
        enc.train()
        perm = torch.randperm(n)
        tot_t = tot_c = 0.0
        for s in range(steps):
            idx = perm[s * a.batch:(s + 1) * a.batch]
            if len(idx) < 4:
                continue
            x = Xt[idx].to(dev, non_blocking=True)
            v1 = augment(x, rng)
            v2 = augment(x, rng)
            b = torch.cat([v1, v2], dim=0)
            b = (b - 128.0) / 64.0                 # simple, fixed normalisation
            f = enc(b)
            lt = nt_xent(f[:len(idx)], f[len(idx):])
            lc = prototype_loss(f[:len(idx)], protos)
            loss = lt + 0.5 * lc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step()
            sched.step()
            tot_t += float(lt)
            tot_c += float(lc)
        if ep % 5 == 0 or ep == a.epochs - 1:
            el = time.time() - t0
            print(f"  epoch {ep:3d}/{a.epochs}  tile {tot_t/steps:.3f}  "
                  f"proto {tot_c/steps:.3f}  {el:.0f}s", flush=True)

    # read out embeddings
    enc.eval()
    embs = []
    with torch.no_grad():
        for s in range(0, n, 256):
            b = Xt[s:s + 256].to(dev).permute(0, 3, 1, 2).contiguous()
            b = (b - 128.0) / 64.0
            embs.append(enc(b).cpu().numpy())
    E = np.concatenate(embs).astype(np.float32)
    print(f"embeddings {E.shape}")

    # k-means -> 64 clusters, HIPPO's read-out
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=a.clusters, random_state=a.seed, n_init=10).fit(E)
    print(f"k-means: {a.clusters} clusters, inertia {km.inertia_:.1f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, emb=E, cluster=km.labels_,
                        centroid=km.cluster_centers_.astype(np.float32))
    torch.save(enc.state_dict(), a.out.replace(".npz", "_encoder.pt"))

    # also write one file per source, in the order the tiles were concatenated,
    # because the downstream fusion joins on tile index within each slide
    off = 0
    for p in a.tiles:
        z = np.load(p)
        m = len(z["img"])
        tag = os.path.basename(p).replace("_tiles.npz", "")
        per = os.path.join(os.path.dirname(a.out), f"{tag}_cnn.npz")
        np.savez_compressed(per, emb=E[off:off + m],
                            cluster=km.labels_[off:off + m])
        print(f"  wrote {per}  ({m} tiles)")
        off += m
    print(f"wrote {a.out}")

    json.dump({"n_tiles": int(n), "params": int(n_par), "epochs": a.epochs,
               "clusters": a.clusters, "source": a.tiles,
               "device": str(dev), "cuda": bool(dev.type == "cuda"),
               "torch": torch.__version__, "cuda_build": torch.version.cuda},
              open(a.out.replace(".npz", "_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
