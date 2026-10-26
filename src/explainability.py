"""Explainability and robustness for the fine-tuned ViT (patch geometry, perf_class, seed 42).

1. Attention rollout (Abnar & Zuidema, 2020) on locked-test images; overlays for correct and
   incorrect predictions.
2. Quantitative attribution check (RQ4): share of rollout mass inside the patch and inside a
   6-pixel halo around the feed, compared with the area share of those regions (enrichment ratio).
   Masks are computed analytically from the design parameters, not hand-drawn.
3. Robustness (RQ5): test-set macro-F1 under small rendering perturbations.

Attention is MODEL ATTRIBUTION, not evidence of electromagnetic causation.

Usage:  python -m src.explainability [--run vit_finetune_patch_perf_class_s42]
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score

from .dataset import BG, build_transforms, geometry_masks
from .utils import ROOT, load_config, load_json, p, save_json


def load_model(cfg, run, k=3):
    m = timm.create_model(cfg["vit"]["model_name"], pretrained=False, num_classes=k)
    m.load_state_dict(torch.load(p(cfg["paths"]["checkpoints"], f"{run}.pt"), map_location="cpu"))
    m.eval()
    dc = timm.data.resolve_data_config({}, model=m)
    return m, dc["mean"], dc["std"]


def attention_maps(model, x):
    """Collect softmax attention of every block by recomputing q.k from the block input."""
    attns = []

    def hook(mod, inp, out):
        xin = inp[0]
        B, N, C = xin.shape
        qkv = mod.qkv(xin).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        q, k = mod.q_norm(q), mod.k_norm(k)
        a = (q * mod.scale) @ k.transpose(-2, -1)
        attns.append(a.softmax(-1).detach())

    hs = [b.attn.register_forward_hook(hook) for b in model.blocks]
    with torch.no_grad():
        logits = model(x)
    for h in hs:
        h.remove()
    return logits, attns


def rollout(attns, discard_ratio: float = 0.0):
    B, _, N, _ = attns[0].shape
    res = torch.eye(N).expand(B, N, N).clone()
    for a in attns:
        a = a.mean(1)
        a = a + torch.eye(N)
        a = a / a.sum(-1, keepdim=True)
        res = a @ res
    cls = res[:, 0, 1:]
    g = int(np.sqrt(cls.shape[-1]))
    m = cls.reshape(B, g, g)
    m = m / m.amax(dim=(1, 2), keepdim=True)
    return m.numpy()


def upsample(m, size=224):
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0


PERTURBATIONS = {
    "none": lambda im: im,
    "translate_4px": lambda im: im.transform(im.size, Image.AFFINE, (1, 0, -4, 0, 1, -4), fillcolor=BG),
    "rotate_3deg": lambda im: im.rotate(3, fillcolor=BG),
    "gaussian_blur_r1": lambda im: im.filter(ImageFilter.GaussianBlur(1)),
    "contrast_0.8": lambda im: ImageEnhance.Contrast(im).enhance(0.8),
    "jpeg_q30": None,  # handled below
    "copper_to_grey": None,  # colour-map shift: patch colour replaced by grey
}


def perturb(name, im):
    if name == "jpeg_q30":
        import io

        b = io.BytesIO()
        im.save(b, "JPEG", quality=30)
        return Image.open(b).convert("RGB")
    if name == "copper_to_grey":
        a = np.asarray(im).astype(int)
        patch = (np.abs(a - np.array([196, 134, 52])).sum(-1) < 90)
        a[patch] = [140, 140, 140]
        return Image.fromarray(a.astype(np.uint8))
    return PERTURBATIONS[name](im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="defaults to the final model recorded by src.select_final")
    args = ap.parse_args()
    cfg = load_config()
    if args.run is None:
        f = p("results", "metrics", "final_model.json")
        args.run = load_json(f)["final_run_main"] if f.exists() else "vit_finetune_patch_perf_class_s42"
    print("explaining", args.run)
    names = cfg["labels"]["perf_names"]
    model, mean, std = load_model(cfg, args.run)
    tf = build_transforms(False, mean, std)
    df = pd.read_csv(ROOT / "data" / "processed" / "patch_designs.csv")
    pred = pd.read_csv(p("results", "predictions", f"{args.run}.csv"))
    test = df[df.split == "test"].merge(pred[pred.split == "test"], on="design_id")

    # ---- rollout over the whole test split + region statistics
    maps, stats = [], []
    for i in range(0, len(test), 32):
        chunk = test.iloc[i:i + 32]
        x = torch.stack([tf(Image.open(ROOT / q).convert("RGB")) for q in chunk.image_path])
        _, at = attention_maps(model, x)
        for r, m in zip(chunk.itertuples(), rollout(at)):
            mu = upsample(m)
            pm, fm = geometry_masks(r.length_mm, r.width_mm, r.feed_y_mm)
            tot = mu.sum()
            stats.append({"design_id": r.design_id, "true": r.true, "pred": r.pred,
                          "patch_mass": mu[pm].sum() / tot, "patch_area": pm.mean(),
                          "feed_mass": mu[fm].sum() / tot, "feed_area": fm.mean(),
                          "top_panel_mass": mu[:112].sum() / tot})
            maps.append(mu)
    st = pd.DataFrame(stats)
    st["patch_enrichment"] = st.patch_mass / st.patch_area
    st["feed_enrichment"] = st.feed_mass / st.feed_area
    st.to_csv(p("results", "metrics", "attention_region_stats.csv"), index=False)
    summary = {
        "run": args.run,
        "n_test": len(st),
        "mean_patch_mass": st.patch_mass.mean(), "mean_patch_area_share": st.patch_area.mean(),
        "median_patch_enrichment": st.patch_enrichment.median(),
        "mean_feed_mass": st.feed_mass.mean(), "mean_feed_area_share": st.feed_area.mean(),
        "median_feed_enrichment": st.feed_enrichment.median(),
        "share_designs_feed_enrichment_gt1": float((st.feed_enrichment > 1).mean()),
        "mean_top_panel_mass": st.top_panel_mass.mean(),
        "feed_enrichment_correct_vs_wrong": [st[st.true == st.pred].feed_enrichment.median(),
                                             st[st.true != st.pred].feed_enrichment.median()],
        "note": "enrichment = attention share / area share; 1.0 = no preference",
    }

    # ---- overlay figure: 4 correct + 4 incorrect test examples
    test = test.reset_index(drop=True)
    corr = test.index[test.true == test.pred].tolist()
    wrong = test.index[test.true != test.pred].tolist()
    rng = np.random.default_rng(cfg["seed"])
    pick = list(rng.choice(corr, min(4, len(corr)), replace=False)) + list(rng.choice(wrong, min(4, len(wrong)), replace=False))
    fig, axs = plt.subplots(2, len(pick), figsize=(2.3 * len(pick), 5.9), gridspec_kw={"hspace": 0.42})
    for j, i in enumerate(pick):
        r = test.iloc[i]
        im = np.asarray(Image.open(ROOT / r.image_path).convert("RGB"))
        axs[0, j].imshow(im)
        axs[0, j].set_title(f"{r.design_id}\ntrue {names[r.true]}", fontsize=8)
        axs[1, j].imshow(im)
        axs[1, j].imshow(maps[i], cmap="jet", alpha=0.5)
        ok = "correct" if r.true == r.pred else "WRONG"
        axs[1, j].set_title(f"pred {names[r.pred]} ({r.confidence:.2f})\n{ok}", fontsize=8,
                            color="#1f6f2a" if ok == "correct" else "#b00020")
        for a in axs[:, j]:
            a.axis("off")
    fig.suptitle("Attention rollout - fine-tuned ViT-S/16, locked test split (model attribution, not EM causation)", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.06)
    fig.savefig(p("results", "figures", "attention_rollout_examples.png"), dpi=200)
    plt.close(fig)

    # mean attention map per true class
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    arr = np.stack(maps)
    for c in range(3):
        sel = (test.true == c).to_numpy()
        axs[c].imshow(arr[sel].mean(0), cmap="jet")
        axs[c].set_title(f"mean rollout, true {names[c]} (n={sel.sum()})", fontsize=9)
        axs[c].axhline(111.5, color="w", lw=0.8)
        axs[c].axis("off")
    fig.tight_layout()
    fig.savefig(p("results", "figures", "attention_mean_by_class.png"), dpi=200)
    plt.close(fig)

    # ---- robustness
    rob = {}
    full = df[df.split == "test"].reset_index(drop=True)
    for name in ["none", "translate_4px", "rotate_3deg", "gaussian_blur_r1", "contrast_0.8", "jpeg_q30", "copper_to_grey"]:
        preds = []
        for i in range(0, len(full), 32):
            chunk = full.iloc[i:i + 32]
            x = torch.stack([tf(perturb(name, Image.open(ROOT / q).convert("RGB"))) for q in chunk.image_path])
            with torch.no_grad():
                preds.append(model(x).argmax(1).numpy())
        yp = np.concatenate(preds)
        rob[name] = {"macro_f1": f1_score(full.perf_class, yp, average="macro"),
                     "accuracy": float((yp == full.perf_class).mean()),
                     "agreement_with_clean": None}
        rob[name]["_pred"] = yp
    clean = rob["none"]["_pred"]
    for name in rob:
        rob[name]["agreement_with_clean"] = float((rob[name]["_pred"] == clean).mean())
    for name in rob:
        del rob[name]["_pred"]
    summary["robustness"] = rob
    save_json(summary, p("results", "metrics", "explainability_summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
