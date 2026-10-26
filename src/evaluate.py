"""Aggregate all executed runs into tables, statistical tests and figures.

Reads results/metrics/*.json and results/predictions/*.csv written by the training scripts.
Only runs that actually exist are reported; nothing is filled in.

Usage:  python -m src.evaluate
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from .utils import ROOT, load_config, p, save_json

NAVY, BLUE, LIGHT, ACCENT, GREY = "#0b2a4a", "#2f6fae", "#9cc3e6", "#e07b39", "#9aa3ad"
MODEL_LABEL = {"majority": "Majority baseline", "cnn": "Compact CNN", "linear_probe": "ViT-S/16 linear probe",
               "finetune": "ViT-S/16 fine-tuned (pre-registered)",
               "finetune_revised": "ViT-S/16 fine-tuned (revised, post-hoc)",
               "finetune_noaug": "ViT-S/16 fine-tuned, no augmentation (ablation)",
               "cnn_noaug": "Compact CNN, no augmentation (ablation)"}
INPUT_LABEL = {"patch": "Geometry image", "radpat": "Radiation-pattern image"}


def load_runs() -> list[dict]:
    runs = []
    for f in sorted(glob.glob(str(ROOT / "results" / "metrics" / "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        if isinstance(d, dict) and "splits" in d and "mode" in d:
            runs.append(d)
    return runs


def main_table(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        for split in ["test", "ood"]:
            if split not in r["splits"]:
                continue
            m = r["splits"][split]
            rows.append({"dataset": r["dataset"], "task": r["task"], "mode": r["mode"], "seed": r.get("seed"),
                         "split": split, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "balanced_accuracy": m["balanced_accuracy"], "weighted_f1": m["weighted_f1"],
                         "ece": m["ece"], "ece_uncal": m.get("ece_uncalibrated"), "roc_auc": m.get("roc_auc_ovr_macro"),
                         "trainable_params": r.get("trainable_params"), "train_seconds": r.get("train_seconds")})
    df = pd.DataFrame(rows)
    df.to_csv(p("results", "metrics", "all_runs.csv"), index=False)
    agg = (df.groupby(["dataset", "task", "mode", "split"])
             .agg(n_seeds=("macro_f1", "size"), accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                  macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                  balanced_accuracy=("balanced_accuracy", "mean"), ece=("ece", "mean"), roc_auc=("roc_auc", "mean"),
                  trainable_params=("trainable_params", "first"))
             .reset_index())
    order = {"majority": 0, "cnn": 1, "linear_probe": 2, "finetune": 3, "finetune_revised": 4, "cnn_noaug": 5, "finetune_noaug": 6}
    agg["o"] = agg["mode"].map(order)
    agg = agg.sort_values(["dataset", "task", "split", "o"]).drop(columns="o")
    agg.to_csv(p("results", "metrics", "summary_table.csv"), index=False)
    return agg


def preds(run):
    f = ROOT / "results" / "predictions" / f"{run}.csv"
    return pd.read_csv(f) if f.exists() else None


def mcnemar(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    m = a[a.split == "test"].merge(b[b.split == "test"], on="design_id", suffixes=("_a", "_b"))
    ca, cb = m.pred_a == m.true_a, m.pred_b == m.true_b
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    pval = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_correct_b_wrong": n01, "a_wrong_b_correct": n10, "exact_p_value": float(pval)}


def bootstrap_ci(df: pd.DataFrame, n: int = 2000, seed: int = 42) -> list[float]:
    t = df[df.split == "test"]
    rng = np.random.default_rng(seed)
    y, yp = t.true.to_numpy(), t.pred.to_numpy()
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[i], yp[i], average="macro"))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def plot_confusions(cfg, runs_to_plot):
    items = [(r, t) for r, t in runs_to_plot if preds(r) is not None]
    if not items:
        return
    fig, axs = plt.subplots(1, len(items), figsize=(3.6 * len(items), 3.5))
    axs = np.atleast_1d(axs)
    for ax, (run, title) in zip(axs, items):
        d = json.load(open(ROOT / "results" / "metrics" / f"{run}.json"))
        names = list(d["splits"]["test"]["per_class"].keys())
        cm = np.array(d["splits"]["test"]["confusion_matrix"])
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.55 else "black")
        band_short = {"FR1_mid_below5GHz": "<5 GHz", "FR1_upper_5to7GHz": "5-7.1 GHz", "FR3_6Gcandidate_above7GHz": ">=7.1 GHz"}
        short = [band_short.get(n, n) for n in names]
        ax.set_xticks(range(len(names)), short, fontsize=8)
        ax.set_yticks(range(len(names)), short, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title}\nmacro-F1 {d['splits']['test']['macro_f1']:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(p("results", "figures", "confusion_matrices.png"), dpi=200)
    plt.close(fig)


def plot_curves(items):
    items = [(r, t) for r, t in items if (ROOT / "results" / "metrics" / f"{r}_history.csv").exists()]
    if not items:
        return
    fig, axs = plt.subplots(1, len(items), figsize=(4.6 * len(items), 3.4))
    axs = np.atleast_1d(axs)
    for ax, (run, title) in zip(axs, items):
        h = pd.read_csv(ROOT / "results" / "metrics" / f"{run}_history.csv")
        ax.plot(h.epoch, h.train_loss, "-o", ms=3, color=NAVY, label="train loss")
        ax.plot(h.epoch, h.val_loss, "-o", ms=3, color=ACCENT, label="val loss")
        ax2 = ax.twinx()
        ax2.plot(h.epoch, h.val_macro_f1, "--s", ms=3, color=BLUE, label="val macro-F1")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("val macro-F1")
        if "stage" in h:
            b = h[h.stage.str.startswith("B")].epoch.min()
            if not np.isnan(b):
                ax.axvline(b - 0.5, color=GREY, lw=0.8, ls=":")
                ax.text(b - 0.4, ax.get_ylim()[1] * 0.95, "partial fine-tuning", fontsize=7, color=GREY, va="top")
        ax.set_xlabel("epoch")
        ax.set_ylabel("weighted CE loss")
        ax.set_title(title, fontsize=9)
        l1, n1 = ax.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, n1 + n2, fontsize=7, loc="center right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(p("results", "figures", "training_curves.png"), dpi=200)
    plt.close(fig)


def plot_comparison(agg: pd.DataFrame, cfg):
    t = agg[agg.split == "test"]
    tasks = [("patch", "perf_class", "Geometry -> matching class"), ("patch", "band_class", "Geometry -> band class"),
             ("radpat", "hpbw_class", "Radiation pattern -> beamwidth class")]
    tasks = [x for x in tasks if ((t.dataset == x[0]) & (t.task == x[1])).any()]
    modes = ["majority", "cnn", "linear_probe", "finetune", "finetune_revised", "finetune_noaug"]
    colors = [GREY, LIGHT, BLUE, NAVY, ACCENT, "#f2b48a"]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    w = 0.14
    for i, (ds, task, title) in enumerate(tasks):
        for j, m in enumerate(modes):
            r = t[(t.dataset == ds) & (t.task == task) & (t["mode"] == m)]
            if r.empty:
                continue
            v, sd = r.macro_f1.iloc[0], r.macro_f1_sd.iloc[0]
            ax.bar(i + (j - 2.5) * w, v, w, color=colors[j], yerr=None if np.isnan(sd) else sd, capsize=3,
                   label=MODEL_LABEL[m] if i == 0 else None)
            ax.text(i + (j - 2.5) * w, v + (0 if np.isnan(sd) else sd) + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(tasks)), [x[2] for x in tasks], fontsize=8)
    ax.set_ylabel("test macro-F1")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(p("results", "figures", "model_comparison.png"), dpi=200)
    plt.close(fig)


def plot_dataset_overview(cfg):
    df = pd.read_csv(ROOT / "data" / "processed" / "patch_designs.csv")
    names = cfg["labels"]["perf_names"]
    fig = plt.figure(figsize=(11, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.3, 2.2])
    ax = fig.add_subplot(gs[0])
    ct = pd.crosstab(df.split, df.perf_class).reindex(["train", "val", "test", "ood"])
    ct.plot(kind="bar", stacked=True, ax=ax, color=[ACCENT, LIGHT, NAVY], width=0.7, legend=False)
    ax.set_title("Matching class by split", fontsize=9)
    ax.legend(names, fontsize=7)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax = fig.add_subplot(gs[1])
    ax.scatter(df.length_mm, df.f_res_ghz, s=4, c=df.perf_class.map({0: ACCENT, 1: LIGHT, 2: NAVY}), alpha=0.7)
    ax.set_xlabel("patch length L [mm]")
    ax.set_ylabel("first matched resonance [GHz]")
    ax.set_title("openEMS designs (n = %d)" % len(df), fontsize=9)
    for e in cfg["labels"]["band_edges_ghz"]:
        ax.axhline(e, color=GREY, lw=0.7, ls="--")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[2])
    sample = df[df.split == "train"].groupby("perf_class").sample(2, random_state=1)
    tiles = [Image.open(ROOT / q).convert("RGB") for q in sample.image_path]
    canvas = Image.new("RGB", (len(tiles) * 230, 224), (255, 255, 255))
    for i, im in enumerate(tiles):
        canvas.paste(im, (i * 230, 0))
    ax.imshow(canvas)
    ax.set_xticks([i * 230 + 112 for i in range(len(tiles))], [f"{d}\n{names[c]}" for d, c in zip(sample.design_id, sample.perf_class)], fontsize=7)
    ax.set_yticks([])
    ax.set_title("Example inputs (top: absolute scale | bottom: normalised)", fontsize=9)
    fig.tight_layout()
    fig.savefig(p("results", "figures", "dataset_overview.png"), dpi=200)
    plt.close(fig)

    rp = ROOT / "data" / "processed" / "radpat_designs.csv"
    if rp.exists():
        r = pd.read_csv(rp)
        s = r[r.split == "train"].groupby("hpbw_class").sample(2, random_state=1)
        fig, axs = plt.subplots(1, len(s), figsize=(2.0 * len(s), 2.3))
        for a, row in zip(axs, s.itertuples()):
            a.imshow(Image.open(ROOT / row.image_path))
            a.set_title(f"{cfg['labels']['hpbw_names'][row.hpbw_class]}\nHPBW {row.HPBW_deg:.1f} deg", fontsize=7)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(p("results", "figures", "radpat_samples.png"), dpi=200)
        plt.close(fig)


def plot_reliability(run="vit_finetune_patch_perf_class_s42"):
    d = preds(run)
    if d is None:
        return
    t = d[d.split == "test"]
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(t.confidence, bins) - 1
    acc = [((t.pred == t.true)[idx == b]).mean() if (idx == b).any() else np.nan for b in range(10)]
    conf = [t.confidence[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], "--", color=GREY)
    ax.plot(conf, acc, "-o", color=NAVY)
    ax.set_xlabel("confidence (temperature-scaled)")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability - ViT fine-tuned, test", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(p("results", "figures", "reliability_vit.png"), dpi=200)
    plt.close(fig)


def main():
    cfg = load_config()
    runs = load_runs()
    agg = main_table(runs)
    fm = ROOT / "results" / "metrics" / "final_model.json"
    final = json.load(open(fm))["final_mode"] if fm.exists() else "finetune"
    stats = {"final_mode": final}
    for mode in ["finetune", "finetune_revised", "finetune_noaug"]:
        for task in ["perf_class", "band_class"]:
            a, b = preds(f"vit_{mode}_patch_{task}_s42"), preds(f"cnn_patch_{task}_s42")
            if a is not None and b is not None:
                stats[f"mcnemar_{mode}_vs_cnn_{task}_s42"] = mcnemar(a, b)
            lp = preds(f"vit_linearprobe_patch_{task}")
            if a is not None and lp is not None:
                stats[f"mcnemar_{mode}_vs_linearprobe_{task}_s42"] = mcnemar(a, lp)
    for run in ["vit_finetune_patch_perf_class_s42", "vit_finetune_revised_patch_perf_class_s42", "vit_finetune_noaug_patch_perf_class_s42",
                "cnn_noaug_patch_perf_class_s42", "cnn_patch_perf_class_s42",
                "vit_linearprobe_patch_perf_class", f"vit_{final}_patch_band_class_s42", f"vit_{final}_radpat_hpbw_class_s42"]:
        d = preds(run)
        if d is not None:
            stats[f"bootstrap95_macro_f1_{run}"] = bootstrap_ci(d)
    save_json(stats, p("results", "metrics", "statistical_tests.json"))
    tag = "revised" if final == "finetune_revised" else "pre-reg."
    plot_confusions(cfg, [(f"vit_{final}_patch_perf_class_s42", f"ViT-S/16 ({tag}) - matching"),
                          ("cnn_patch_perf_class_s42", "Compact CNN - matching"),
                          (f"vit_{final}_patch_band_class_s42", f"ViT-S/16 ({tag}) - band"),
                          (f"vit_{final}_radpat_hpbw_class_s42", f"ViT-S/16 ({tag}) - beamwidth")])
    plot_curves([("vit_finetune_patch_perf_class_s42", "ViT-S/16 pre-registered (seed 42)"),
                 ("vit_finetune_revised_patch_perf_class_s42", "ViT-S/16 revised, post-hoc (seed 42)"),
                 ("cnn_patch_perf_class_s42", "Compact CNN (seed 42)")])
    plot_comparison(agg, cfg)
    plot_dataset_overview(cfg)
    plot_reliability(f"vit_{final}_patch_perf_class_s42")
    with open(p("results", "metrics", "summary_table.md"), "w", encoding="utf-8") as f:
        f.write(agg.round(4).to_markdown(index=False) if hasattr(agg, "to_markdown") else agg.round(4).to_string())
    print(agg.round(3).to_string())
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

