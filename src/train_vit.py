"""Module 1 - Vision Transformer (ViT-Small/16, ImageNet-21k pretrained, timm).

Modes
  finetune     : stage A head warm-up, stage B head + last N transformer blocks, early stopping on
                 validation macro-F1, temperature scaling on validation, single locked test evaluation.
  linear_probe : frozen pretrained backbone, CLS features + multinomial logistic regression
                 (C selected on validation macro-F1).

Usage
  python -m src.train_vit --dataset patch  --task perf_class --mode finetune --seed 42
  python -m src.train_vit --dataset patch  --task perf_class --mode linear_probe
  python -m src.train_vit --dataset radpat --task hpbw_class --mode finetune
"""
from __future__ import annotations

import argparse
import copy
import math
import platform

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from .train_common import (Timer, class_names, class_weights, eval_loss_f1, finalize_run, make_loaders, predict,
                           train_epoch)
from .utils import load_config, p, set_seed


def build_vit(cfg: dict, k: int):
    m = timm.create_model(cfg["vit"]["model_name"], pretrained=True, num_classes=k, drop_rate=cfg["vit"]["dropout"])
    dc = timm.data.resolve_data_config({}, model=m)
    return m, dc["mean"], dc["std"]


def features(model):
    return lambda x: model.forward_head(model.forward_features(x), pre_logits=True)


def set_trainable(model, n_blocks: int | None):
    """n_blocks=None -> head only; otherwise head + final norm + last n_blocks blocks."""
    for prm in model.parameters():
        prm.requires_grad = False
    for prm in model.head.parameters():
        prm.requires_grad = True
    if n_blocks:
        for blk in model.blocks[-n_blocks:]:
            for prm in blk.parameters():
                prm.requires_grad = True
        for prm in model.norm.parameters():
            prm.requires_grad = True


def run_finetune(args, cfg):
    names = class_names(cfg, args.task)
    k = len(names)
    v = cfg["vit"] if args.profile == "preregistered" else cfg["vit_revised"]
    mode = "finetune" if args.profile == "preregistered" else "finetune_revised"
    if args.no_aug:
        mode += "_noaug"  # ablation listed in the Module 1 report (Sec. 8.2): with vs without augmentation
    model, mean, std = build_vit(cfg, k)
    df, tables, loaders = make_loaders(args.dataset, args.task, mean, std, v["batch_size"], augment=not args.no_aug)
    w = class_weights(tables["train"].labels, k)
    history, best = [], {"f1": -1, "state": None, "epoch": 0}
    run = f"vit_{mode}_{args.dataset}_{args.task}_s{args.seed}"
    ep = 0

    # Stage A - classification head warm-up
    set_trainable(model, None)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=v["lr_head"], weight_decay=v["weight_decay"])
    for _ in range(v["warmup_epochs"]):
        ep += 1
        with Timer() as t:
            tl, ta = train_epoch(model, loaders["train"], opt, w)
            vl, vf1, va = eval_loss_f1(model, loaders["val"], w)
        history.append({"epoch": ep, "stage": "A_head", "train_loss": tl, "train_acc": ta, "val_loss": vl,
                        "val_macro_f1": vf1, "val_acc": va, "seconds": t.elapsed})
        print(history[-1], flush=True)
        if vf1 > best["f1"]:
            best = {"f1": vf1, "state": copy.deepcopy(model.state_dict()), "epoch": ep}

    # Stage B - partial fine-tuning of the upper transformer blocks
    set_trainable(model, v["unfreeze_blocks"])
    head = list(model.head.parameters())
    body = [q for n, q in model.named_parameters() if q.requires_grad and not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": body, "lr": v["lr_backbone"]}, {"params": head, "lr": v["lr_backbone"] * 10}],
                            weight_decay=v["weight_decay"])
    steps = v["finetune_epochs"] * math.ceil(len(tables["train"]) / v["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[v["lr_backbone"], v["lr_backbone"] * 10], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    bad = 0
    for _ in range(v["finetune_epochs"]):
        ep += 1
        with Timer() as t:
            tl, ta = train_epoch(model, loaders["train"], opt, w, sched)
            vl, vf1, va = eval_loss_f1(model, loaders["val"], w)
        history.append({"epoch": ep, "stage": "B_partial", "train_loss": tl, "train_acc": ta, "val_loss": vl,
                        "val_macro_f1": vf1, "val_acc": va, "seconds": t.elapsed})
        print(history[-1], flush=True)
        if vf1 > best["f1"] + 1e-4:
            best, bad = {"f1": vf1, "state": copy.deepcopy(model.state_dict()), "epoch": ep}, 0
        else:
            bad += 1
            if bad >= v["patience"]:
                print("early stopping", flush=True)
                break
    pd.DataFrame(history).to_csv(p("results", "metrics", f"{run}_history.csv"), index=False)

    model.load_state_dict(best["state"])
    torch.save(best["state"], p(cfg["paths"]["checkpoints"], f"{run}.pt"))
    logits, ys, feats = {}, {}, {}
    for s, ld in loaders.items():
        ld_eval = torch.utils.data.DataLoader(tables[s], batch_size=32, shuffle=False)
        # evaluation uses the non-augmented transform
        tables[s].transform = loaders["val"].dataset.transform
        logits[s], ys[s], feats[s] = predict(model, ld_eval, features(model))
        np.save(p("results", "features", f"{run}_{s}.npy"), feats[s])
    n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    extra = {"mode": mode, "profile": args.profile, "augmentation": not args.no_aug, "config": v, "dataset": args.dataset, "seed": args.seed, "best_epoch": best["epoch"],
             "best_val_macro_f1": best["f1"], "epochs_run": ep, "trainable_params": n_train,
             "total_params": sum(q.numel() for q in model.parameters()), "model": cfg["vit"]["model_name"],
             "train_seconds": float(sum(h["seconds"] for h in history)), "torch": torch.__version__,
             "timm": timm.__version__, "cpu": platform.processor(), "hardware": "CPU only (no CUDA)"}
    finalize_run(run, cfg, args.task, tables, logits, ys, extra)


def run_linear_probe(args, cfg):
    names = class_names(cfg, args.task)
    k = len(names)
    model, mean, std = build_vit(cfg, k)
    df, tables, loaders = make_loaders(args.dataset, args.task, mean, std, 32)
    cache = p("results", "features", f"vit_pretrained_{args.dataset}.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        feats = {s: z[s] for s in tables}
    else:
        feats = {}
        for s in tables:
            tables[s].transform = loaders["val"].dataset.transform
            ld = torch.utils.data.DataLoader(tables[s], batch_size=32, shuffle=False)
            _, _, feats[s] = predict(model, ld, features(model))
        np.savez(cache, **feats)
    ys = {s: tables[s].labels for s in tables}
    sc = StandardScaler().fit(feats["train"])
    best = (-1, None)
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(feats["train"]), ys["train"])
        f1 = f1_score(ys["val"], clf.predict(sc.transform(feats["val"])), average="macro")
        print(f"C={C} val macroF1={f1:.3f}")
        if f1 > best[0]:
            best = (f1, C, clf)
    clf = best[2]
    logits = {s: clf.decision_function(sc.transform(feats[s])) for s in tables}
    run = f"vit_linearprobe_{args.dataset}_{args.task}"
    finalize_run(run, cfg, args.task, tables, logits, ys,
                 {"mode": "linear_probe", "dataset": args.dataset, "C": best[1], "best_val_macro_f1": best[0],
                  "model": cfg["vit"]["model_name"], "trainable_params": int(clf.coef_.size + clf.intercept_.size)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="patch", choices=["patch", "radpat"])
    ap.add_argument("--task", default="perf_class", choices=["perf_class", "band_class", "hpbw_class"])
    ap.add_argument("--mode", default="finetune", choices=["finetune", "linear_probe"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--profile", default="preregistered", choices=["preregistered", "revised"],
                    help="hyper-parameter block: 'vit' (pre-registered) or 'vit_revised' (post-hoc, validation-selected)")
    ap.add_argument("--no-aug", action="store_true", help="ablation: disable training augmentation")
    args = ap.parse_args()
    cfg = load_config()
    args.seed = cfg["seed"] if args.seed is None else args.seed
    set_seed(args.seed, cfg["num_threads"])
    (run_finetune if args.mode == "finetune" else run_linear_probe)(args, cfg)


if __name__ == "__main__":
    main()
