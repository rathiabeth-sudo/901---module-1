"""Shared training / evaluation utilities for image classifiers."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)

from .dataset import ImageTable, build_transforms, load_table
from .utils import ROOT, p, save_json


def class_names(cfg: dict, task: str) -> list[str]:
    return {"perf_class": cfg["labels"]["perf_names"], "band_class": cfg["labels"]["band_names"],
            "hpbw_class": cfg["labels"]["hpbw_names"]}[task]


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def classification_metrics(y: np.ndarray, probs: np.ndarray, names: list[str]) -> dict:
    pred = probs.argmax(1)
    k = len(names)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(k)), zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": f1_score(y, pred, average="macro", labels=list(range(k)), zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", labels=list(range(k)), zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "ece": expected_calibration_error(probs, y),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(k))).tolist(),
        "per_class": {names[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                                 "support": int(sup[i])} for i in range(k)},
    }
    try:
        if len(np.unique(y)) == k:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError:
        pass
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling on validation logits (Guo et al., 2017)."""
    t = torch.zeros(1, requires_grad=True)
    lg = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.exp().item())


def softmax_np(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = z / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def make_loaders(dataset: str, task: str, mean, std, batch_size: int, hflip: bool = True, augment: bool = True):
    df = load_table(dataset, ROOT)
    splits = ["train", "val", "test"] + (["ood"] if (df.split == "ood").any() else [])
    tables, loaders = {}, {}
    for s in splits:
        part = df[df.split == s].reset_index(drop=True)
        tf = build_transforms(s == "train" and augment, mean, std, hflip=hflip)
        tables[s] = ImageTable(part, task, ROOT, tf)
        loaders[s] = torch.utils.data.DataLoader(tables[s], batch_size=batch_size, shuffle=(s == "train"), num_workers=0)
    return df, tables, loaders


@torch.no_grad()
def predict(model, loader, feature_fn=None):
    model.eval()
    logits, ys, feats = [], [], []
    for x, y in loader:
        if feature_fn is not None:
            f = feature_fn(x)
            out = model.get_classifier()(f) if hasattr(model, "get_classifier") else model.head(f)
            feats.append(f.numpy())
        else:
            out = model(x)
        logits.append(out.numpy())
        ys.append(y.numpy())
    return np.concatenate(logits), np.concatenate(ys), (np.concatenate(feats) if feats else None)


def train_epoch(model, loader, opt, weights, sched=None):
    model.train()
    tot, correct, n = 0.0, 0, 0
    for x, y in loader:
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y, weight=weights)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        tot += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return tot / n, correct / n


def eval_loss_f1(model, loader, weights):
    logits, y, _ = predict(model, loader)
    loss = F.cross_entropy(torch.tensor(logits), torch.tensor(y), weight=weights).item()
    pred = logits.argmax(1)
    return loss, f1_score(y, pred, average="macro"), accuracy_score(y, pred)


def class_weights(labels: np.ndarray, k: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=k).astype(float)
    w = counts.sum() / (k * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def finalize_run(run_name: str, cfg: dict, task: str, tables: dict, split_logits: dict, split_y: dict,
                 extra: dict) -> dict:
    """Temperature-scale on val, compute metrics for every split, save predictions & metrics."""
    names = class_names(cfg, task)
    T = fit_temperature(split_logits["val"], split_y["val"])
    metrics = {"run": run_name, "task": task, "temperature": T, **extra, "splits": {}}
    rows = []
    for s, lg in split_logits.items():
        probs = softmax_np(lg, T)
        metrics["splits"][s] = classification_metrics(split_y[s], probs, names)
        metrics["splits"][s]["ece_uncalibrated"] = expected_calibration_error(softmax_np(lg), split_y[s])
        for i, did in enumerate(tables[s].ids):
            rows.append({"design_id": did, "split": s, "true": int(split_y[s][i]), "pred": int(probs[i].argmax()),
                         "confidence": float(probs[i].max()), **{f"p_{n}": float(probs[i, j]) for j, n in enumerate(names)}})
    pd.DataFrame(rows).to_csv(p("results", "predictions", f"{run_name}.csv"), index=False)
    save_json(metrics, p("results", "metrics", f"{run_name}.json"))
    t = metrics["splits"]["test"]
    print(f"[{run_name}] TEST acc={t['accuracy']:.3f} macroF1={t['macro_f1']:.3f} balAcc={t['balanced_accuracy']:.3f} ECE={t['ece']:.3f}")
    return metrics


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t
