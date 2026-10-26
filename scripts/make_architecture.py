"""Module 1 block diagram (implemented vs proposed), in the style of the framework figure.

Writes results/figures/module1_architecture.png. Thumbnails are real model inputs from data/processed.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from deliverables_common import ACCENT, BLUE, FIG, GREY, NAVY, ROOT

IMPL, PROP, VIT, OUT = "#e8f1fa", "#fdf1e7", "#eef0fb", "#eaf5ee"


def box(ax, x, y, w, h, text="", fc=IMPL, ec=BLUE, fs=8.4, bold=False, dashed=False, tc=NAVY, va="center", ty=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01", fc=fc, ec=ec,
                                lw=1.2, ls="--" if dashed else "-"))
    if text:
        ax.text(x + w / 2, (y + h / 2) if ty is None else ty, text, ha="center", va=va, fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", linespacing=1.3)


def group(ax, x, y, w, h, title, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015", fc="white", ec=ec, lw=1.6))
    ax.text(x + 0.008, y + h - 0.012, title, fontsize=9.5, fontweight="bold", color=NAVY, va="top")


def arrow(ax, x1, y1, x2, y2, color=NAVY, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color,
                                 ls="--" if dashed else "-"))


def thumb(fig, ax, path, x, y, w, h):
    """Place an image in data coordinates of `ax` (0..1 both axes)."""
    bb = ax.get_position()
    a = fig.add_axes([bb.x0 + x * bb.width, bb.y0 + y * bb.height, w * bb.width, h * bb.height])
    a.imshow(mpimg.imread(path))
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_color("#9cc3e6")
    return a


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "patch_designs.csv")
    rp = pd.read_csv(ROOT / "data" / "processed" / "radpat_designs.csv")
    geo = ROOT / df.loc[df.design_id == "P1307", "image_path"].iloc[0]
    rp_img = ROOT / rp.iloc[0].image_path
    if not rp_img.exists():
        rp_img = FIG / "radpat_samples.png"

    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_axes([0.005, 0.01, 0.99, 0.98])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0.935), 1, 0.065, fc=NAVY, ec="none"))
    ax.text(0.5, 0.967, "Module 1 - Image-Based Vision Transformer for Antenna Performance Screening", color="white",
            fontsize=13, fontweight="bold", ha="center", va="center")

    # 1 data
    group(ax, 0.008, 0.10, 0.17, 0.82, "1  Data sources", "#7fb37f")
    box(ax, 0.018, 0.66, 0.15, 0.20, "openEMS patch sweep\n2,051 coax-fed designs\nL, W, feed offset\n+ simulated S11 (1-10 GHz)\nHugging Face, MIT")
    box(ax, 0.018, 0.42, 0.15, 0.20, "RadPat-50K\n3,000 polar radiation-\npattern plots + HPBW\n(analytical array factors)\nZenodo, CC BY 4.0")
    box(ax, 0.018, 0.13, 0.15, 0.25, "Ansys HFSS hybrid-\nslotted DGS sweep\n(PyAEDT)\ngeometry, surface-current\n& E-field images\n(proposed - not run)",
        fc=PROP, ec=ACCENT, dashed=True)

    # 2 preprocessing
    group(ax, 0.19, 0.10, 0.19, 0.82, "2  Preprocessing", "#7fb37f")
    box(ax, 0.20, 0.56, 0.17, 0.30, "", ty=None)
    ax.text(0.285, 0.845, "Standardised 224×224 input", fontsize=8.4, ha="center", va="top", color=NAVY, fontweight="bold")
    thumb(fig, ax, geo, 0.212, 0.605, 0.072, 0.20)
    thumb(fig, ax, rp_img, 0.292, 0.62, 0.07, 0.17)
    ax.text(0.248, 0.597, "geometry", fontsize=6.8, ha="center", va="top", color=GREY)
    ax.text(0.327, 0.597, "radiation pattern", fontsize=6.8, ha="center", va="top", color=GREY)
    box(ax, 0.20, 0.39, 0.17, 0.14, "Labels from simulated S11\n(rules frozen in Git):\nmatching | band | beamwidth")
    box(ax, 0.20, 0.23, 0.17, 0.13, "Design-level split 70/15/15\n(seed 42) + OOD hold-out\nW/L > 1.95 (n = 307)")
    box(ax, 0.20, 0.12, 0.17, 0.08, "QC: MD5 de-duplication,\nno text / legends, ViT norm.", fs=7.8)

    # 3 ViT
    group(ax, 0.395, 0.10, 0.355, 0.82, "3  Vision Transformer (ViT-S/16, ImageNet-21k pretrained)", BLUE)
    ta = thumb(fig, ax, geo, 0.405, 0.60, 0.06, 0.18)
    for k in range(16, 224, 16):  # the 14 x 14 grid of 16 x 16 px ViT patches
        ta.axvline(k - 0.5, color="white", lw=0.4, alpha=0.9)
        ta.axhline(k - 0.5, color="white", lw=0.4, alpha=0.9)
    ax.text(0.435, 0.585, "196 patches\n(16×16 px)", fontsize=7, ha="center", va="top", color=GREY)
    arrow(ax, 0.468, 0.69, 0.482, 0.69)
    box(ax, 0.484, 0.55, 0.055, 0.30, "Patch\nembed-\nding\n384-d\n+ CLS\n+ pos.", fc=VIT, ec="#7b7fc4", fs=7.6)
    for k in range(7):  # token column
        ax.add_patch(Rectangle((0.546, 0.57 + k * 0.038), 0.012, 0.026, fc="#b9bbe8" if k else "#e07b39", ec="#7b7fc4", lw=0.6))
    ax.text(0.552, 0.555, "tokens", fontsize=6.5, ha="center", va="top", color=GREY)
    arrow(ax, 0.561, 0.69, 0.573, 0.69)
    box(ax, 0.575, 0.55, 0.08, 0.30, "Transformer\nencoder\n12 blocks\n6-head\nself-attention\n(last 4\nfine-tuned)", fc=VIT, ec="#7b7fc4", fs=7.6, bold=False)
    arrow(ax, 0.657, 0.69, 0.668, 0.69)
    box(ax, 0.67, 0.55, 0.07, 0.30, "Classifi-\ncation\nhead\n(CLS →\n3 classes)\n+ temp.\nscaling", fc="#e3f2e6", ec="#5e9e6e", fs=7.6)
    box(ax, 0.405, 0.30, 0.16, 0.20, "Comparators (same split)\n• majority class\n• compact CNN (0.25 M)\n• frozen-ViT linear probe\n• no-augmentation ablation", fs=7.8)
    box(ax, 0.575, 0.30, 0.165, 0.20, "Explainability & stress tests\n• attention rollout vs\n  analytic patch / feed masks\n• 7 rendering perturbations\n• OOD hold-out", fs=7.8)
    box(ax, 0.405, 0.13, 0.335, 0.13, "Training: stage A - head warm-up (2 epochs, lr 1e-3)\nstage B - last 4 blocks fine-tuned (≤ 10 epochs, lr 3e-5)\n"
        "AdamW, batch 16, class-weighted CE, early stopping on val macro-F1, CPU only", fs=7.8, fc="white")

    # 4 outputs
    group(ax, 0.765, 0.10, 0.228, 0.82, "4  Outputs", "#5e9e6e")
    outs = [("Matching class (Poor /\nAcceptable / High)\n+ calibrated probability", 0.735),
            ("Band class (<5 | 5-7.125 |\n≥7.125 GHz); beamwidth\nclass for RadPat", 0.585),
            ("Attention-rollout maps\n(model attribution only)", 0.435),
            ("384-d CLS embedding →\nModule 2 multimodal fusion", 0.285)]
    for t, y in outs:
        box(ax, 0.775, y, 0.208, 0.12, t, fc=OUT, ec="#5e9e6e", fs=8)
    box(ax, 0.775, 0.13, 0.208, 0.12, "HFSS full-wave verification\nof screened designs\n(proposed)", fc=PROP, ec=ACCENT, dashed=True, fs=8)

    # arrows between groups
    arrow(ax, 0.168, 0.76, 0.20, 0.72)
    arrow(ax, 0.168, 0.52, 0.20, 0.66)
    arrow(ax, 0.168, 0.25, 0.20, 0.30, color=ACCENT, dashed=True)
    arrow(ax, 0.37, 0.70, 0.403, 0.70)
    arrow(ax, 0.74, 0.74, 0.775, 0.79)
    arrow(ax, 0.74, 0.68, 0.775, 0.645)
    arrow(ax, 0.74, 0.40, 0.775, 0.49)
    arrow(ax, 0.74, 0.58, 0.775, 0.345)
    arrow(ax, 0.879, 0.285, 0.879, 0.25, color=ACCENT, dashed=True)

    ax.add_patch(FancyBboxPatch((0.30, 0.035), 0.02, 0.025, boxstyle="round,pad=0.002", fc=IMPL, ec=BLUE))
    ax.text(0.325, 0.047, "implemented & executed", fontsize=8, va="center", color=NAVY)
    ax.add_patch(FancyBboxPatch((0.50, 0.035), 0.02, 0.025, boxstyle="round,pad=0.002", fc=PROP, ec=ACCENT, ls="--"))
    ax.text(0.525, 0.047, "proposed future work (requires Ansys HFSS)", fontsize=8, va="center", color=NAVY)

    fig.savefig(FIG / "module1_architecture.png", dpi=220)
    plt.close(fig)
    print("module1_architecture.png written")


if __name__ == "__main__":
    main()
