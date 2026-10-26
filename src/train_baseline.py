"""Baselines: majority-class classifier and a compact CNN trained from scratch.

Usage
  python -m src.train_baseline --dataset patch --task perf_class --model majority
  python -m src.train_baseline --dataset patch --task perf_class --model cnn --seed 42
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .train_common import (Timer, class_names, class_weights, eval_loss_f1, finalize_run, make_loaders, predict,
                           train_epoch)
from .utils import load_config, p, set_seed

MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


class CompactCNN(nn.Module):
    """Stride-2 stem + four conv blocks (32-64-128-128) + global average pooling; ~0.27 M parameters."""

    def __init__(self, k: int, dropout: float = 0.2):
        super().__init__()
        layers = [nn.Conv2d(3, 16, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True)]
        c_in = 16
        for c in [32, 64, 128, 128]:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, k))

    def forward(self, x):
        return self.head(self.features(x))


def run_majority(args, cfg):
    names = class_names(cfg, args.task)
    _, tables, _ = make_loaders(args.dataset, args.task, MEAN, STD, 64)
    prior = np.bincount(tables["train"].labels, minlength=len(names)) / len(tables["train"])
    logits = {s: np.tile(np.log(prior + 1e-9), (len(t), 1)) for s, t in tables.items()}
    ys = {s: t.labels for s, t in tables.items()}
    finalize_run(f"majority_{args.dataset}_{args.task}", cfg, args.task, tables, logits, ys,
                 {"mode": "majority", "dataset": args.dataset, "train_prior": prior.tolist()})


def run_cnn(args, cfg):
    names = class_names(cfg, args.task)
    k = len(names)
    c = cfg["cnn"]
    _, tables, loaders = make_loaders(args.dataset, args.task, MEAN, STD, c["batch_size"], augment=not args.no_aug)
    model = CompactCNN(k)
    w = class_weights(tables["train"].labels, k)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])
    run = f"cnn{'_noaug' if args.no_aug else ''}_{args.dataset}_{args.task}_s{args.seed}"
    best, bad, hist = {"f1": -1}, 0, []
    for ep in range(1, c["epochs"] + 1):
        with Timer() as t:
            tl, ta = train_epoch(model, loaders["train"], opt, w)
            vl, vf1, va = eval_loss_f1(model, loaders["val"], w)
        sched.step()
        hist.append({"epoch": ep, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_macro_f1": vf1,
                     "val_acc": va, "seconds": t.elapsed})
        print(hist[-1], flush=True)
        if vf1 > best["f1"] + 1e-4:
            best, bad = {"f1": vf1, "state": copy.deepcopy(model.state_dict()), "epoch": ep}, 0
        else:
            bad += 1
            if bad >= c["patience"]:
                print("early stopping", flush=True)
                break
    pd.DataFrame(hist).to_csv(p("results", "metrics", f"{run}_history.csv"), index=False)
    model.load_state_dict(best["state"])
    torch.save(best["state"], p(cfg["paths"]["checkpoints"], f"{run}.pt"))
    logits, ys = {}, {}
    for s in tables:
        tables[s].transform = tables["val"].transform
        logits[s], ys[s], _ = predict(model, torch.utils.data.DataLoader(tables[s], batch_size=64))
    finalize_run(run, cfg, args.task, tables, logits, ys,
                 {"mode": "cnn_noaug" if args.no_aug else "cnn", "augmentation": not args.no_aug, "dataset": args.dataset, "seed": args.seed, "best_epoch": best["epoch"],
                  "best_val_macro_f1": best["f1"], "trainable_params": sum(q.numel() for q in model.parameters()),
                  "train_seconds": float(sum(h["seconds"] for h in hist))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="patch", choices=["patch", "radpat"])
    ap.add_argument("--task", default="perf_class", choices=["perf_class", "band_class", "hpbw_class"])
    ap.add_argument("--model", default="cnn", choices=["cnn", "majority"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-aug", action="store_true", help="ablation: disable training augmentation")
    args = ap.parse_args()
    cfg = load_config()
    args.seed = cfg["seed"] if args.seed is None else args.seed
    set_seed(args.seed, cfg["num_threads"])
    (run_cnn if args.model == "cnn" else run_majority)(args, cfg)


if __name__ == "__main__":
    main()
