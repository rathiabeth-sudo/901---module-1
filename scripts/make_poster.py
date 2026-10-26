"""Module 1 research poster (16:9, 20 x 11.25 in) -> poster/module1_poster.png (300 dpi) + .pdf.

All numbers and plots come from executed results (results/metrics, results/figures).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from deliverables_common import ACCENT, BLUE, FIG, GREY, LIGHT, NAVY, REPO_URL, ROOT, SUBTITLE, TITLE, hypotheses, load

W, H = 20, 11.25
EDGE = "#9cc3e6"


def panel(fig, x, y, w, h, title, num=None):
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.003, 0.003), 0.994, 0.994, boxstyle="round,pad=0,rounding_size=0.02",
                                fc="white", ec=EDGE, lw=1.2, transform=ax.transAxes))
    hh = 0.0486 / h  # constant physical header height
    ax.add_patch(Rectangle((0.003, 1 - hh), 0.994, hh - 0.003, fc=LIGHT, ec="none", transform=ax.transAxes))
    ax.text(0.02, 1 - hh / 2, (f"{num}  " if num else "") + title, fontsize=15, fontweight="bold", color=NAVY,
            va="center", transform=ax.transAxes)
    return ax, 1 - hh


def image(fig, path, x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    return ax


def bullets(ax, items, x, y, fs=11.2, dy=0.075, color="#1b2733"):
    for it in items:
        ax.text(x, y, "• " + it, fontsize=fs, color=color, va="top", transform=ax.transAxes, linespacing=1.3)
        y -= dy * (1 + it.count("\n"))
    return y


def main():
    R = load()
    row, fmt = R["row"], R["fmt"]
    vit, cnn = row("patch", "perf_class", "finetune"), row("patch", "perf_class", "cnn")
    lp, maj = row("patch", "perf_class", "linear_probe"), row("patch", "perf_class", "majority")
    vb, lb = row("patch", "band_class", "finetune"), row("patch", "band_class", "linear_probe")
    vr = row("radpat", "hpbw_class", "finetune")
    vood = row("patch", "perf_class", "finetune", "ood")
    noa = row("patch", "perf_class", "finetune_noaug")
    e, prep = R["expl"], R["prep"]
    rob = e["robustness"]
    sc = prep["patch_split_counts"]

    fig = plt.figure(figsize=(W, H), facecolor="white")
    hd = fig.add_axes([0, 0.885, 1, 0.115])
    hd.axis("off")
    hd.add_patch(Rectangle((0, 0), 1, 1, fc=NAVY, transform=hd.transAxes))
    hd.text(0.5, 0.70, TITLE, color="white", fontsize=25, fontweight="bold", ha="center", va="center")
    hd.text(0.5, 0.40, SUBTITLE, color="#cfe2f3", fontsize=15, ha="center", va="center")
    hd.text(0.5, 0.13, "Ms. Rathi Devadhas  |  Student ID 1737089  |  Master of Applied Artificial Intelligence  |  Applied AI Research Workshop - Module 1  |  26 October 2026",
            color="white", fontsize=11.5, ha="center", va="center", style="italic")

    # ---------------- column 1
    ax, top = panel(fig, 0.008, 0.585, 0.27, 0.29, "Research objective", "1")
    bullets(ax, ["Can standardised antenna geometry images predict\nsimulated performance classes - and show where\nthe model looks?",
                 "Module 1 = image branch of the two-module\nframework; Module 2 (surrogate, fusion, NSGA-II)\nis a separate study.",
                 "Proof of concept on public simulation data;\nAnsys HFSS was not available and was not run.",
                 "Hypotheses H1-H5 committed to Git before training."], 0.03, top - 0.04, fs=11, dy=0.085)

    ax, top = panel(fig, 0.008, 0.07, 0.27, 0.505, "Data & preprocessing", "2")
    bullets(ax, [f"openEMS patch sweep (Hugging Face, MIT): 2,051\ndesigns -> one 224x224 two-panel image each",
                 "RadPat-50K (Zenodo, CC BY 4.0): 3,000 polar\nradiation-pattern plots -> beamwidth class",
                 f"Locked design-level split 70/15/15 (seed 42):\n{sc['train']:,} / {sc['val']} / {sc['test']} + OOD hold-out W/L > 1.95 (n = {sc['ood']})",
                 "No text / legends, MD5 de-duplication,\ntraining-only mirror, ±3° and 2% shift"], 0.03, top - 0.03, fs=10.4, dy=0.04)
    ax.text(0.03, 0.47, "Matching class: Poor > -10 dB | Acceptable -20 to -10 dB | High <= -20 dB*", fontsize=8.9, color=NAVY,
            transform=ax.transAxes, fontweight="bold")
    ax.text(0.03, 0.435, "*-20 dB boundary is a project assumption awaiting engineering confirmation", fontsize=8.6, color=GREY,
            transform=ax.transAxes, style="italic")
    image(fig, FIG / "dataset_overview.png", 0.012, 0.08, 0.262, 0.2)

    # ---------------- column 2
    ax, top = panel(fig, 0.285, 0.505, 0.43, 0.37, "Module 1 block diagram (implemented vs proposed)", "3")
    image(fig, FIG / "module1_architecture.png", 0.29, 0.51, 0.42, 0.32)

    ax, top = panel(fig, 0.285, 0.07, 0.43, 0.425, "Results on the locked test split (executed)", "4")
    image(fig, FIG / "model_comparison.png", 0.29, 0.255, 0.205, 0.19)
    image(fig, FIG / "confusion_matrices.png", 0.497, 0.27, 0.215, 0.165)
    tbl = [("Model (geometry -> matching class)", "Macro-F1", "Bal. acc."),
           ("Majority baseline", f"{maj.macro_f1:.3f}", f"{maj.balanced_accuracy:.3f}"),
           ("Compact CNN (3 seeds)", fmt(cnn), f"{cnn.balanced_accuracy:.3f}"),
           ("ViT-S/16 linear probe", f"{lp.macro_f1:.3f}", f"{lp.balanced_accuracy:.3f}"),
           ("ViT-S/16 fine-tuned (3 seeds)", fmt(vit), f"{vit.balanced_accuracy:.3f}"),
           ("ViT fine-tuned - OOD hold-out", fmt(vood), f"{vood.balanced_accuracy:.3f}"),
           ("ViT, no augmentation (ablation*)", f"{noa.macro_f1:.3f}", f"{noa.balanced_accuracy:.3f}")]
    y0 = 0.40
    for i, (a, b, c) in enumerate(tbl):
        w = "bold" if i in (0, 4) else "normal"
        col = NAVY if i else BLUE
        ax.text(0.03, y0 - i * 0.047, a, fontsize=10.5, fontweight=w, color=col, transform=ax.transAxes)
        ax.text(0.53, y0 - i * 0.047, b, fontsize=10.5, fontweight=w, color=col, transform=ax.transAxes)
        ax.text(0.70, y0 - i * 0.047, c, fontsize=10.5, fontweight=w, color=col, transform=ax.transAxes)
    ax.text(0.80, y0, "Other tasks", fontsize=10.5, fontweight="bold", color=BLUE, transform=ax.transAxes)
    ax.text(0.80, y0 - 0.052, f"Band class\nViT {vb.macro_f1:.3f}\nprobe {lb.macro_f1:.3f}", fontsize=9.6, color=NAVY, transform=ax.transAxes, va="top")
    ax.text(0.80, y0 - 0.20, f"RadPat beamwidth\nViT {vr.macro_f1:.3f}", fontsize=9.6, color=NAVY, transform=ax.transAxes, va="top")
    ax.text(0.03, 0.035, "*exploratory, 1 seed, not pre-registered; underfitting traced to augmentation blurring the 1-2 px feed offset",
            fontsize=8.6, color=GREY, style="italic", transform=ax.transAxes)

    # ---------------- column 3
    ax, top = panel(fig, 0.722, 0.61, 0.27, 0.265, "Explainability (attention rollout)", "5")
    image(fig, FIG / "attention_rollout_examples.png", 0.726, 0.685, 0.262, 0.14)
    ax.text(0.03, 0.2, f"Median enrichment vs area share: feed halo ×{e['median_feed_enrichment']:.2f}, patch ×{e['median_patch_enrichment']:.2f}",
            fontsize=10, color=NAVY, transform=ax.transAxes)
    ax.text(0.03, 0.08, "Attribution only - not proof of electromagnetic causation", fontsize=9, color=GREY, style="italic",
            transform=ax.transAxes)

    ax, top = panel(fig, 0.722, 0.36, 0.27, 0.24, "Robustness & key insight", "6")
    yy = bullets(ax, [f"Blur (1 px) -> {rob['gaussian_blur_r1']['macro_f1']:.3f}, grey copper -> {rob['copper_to_grey']['macro_f1']:.3f}\n(clean {rob['none']['macro_f1']:.3f}); contrast and JPEG barely matter",
                      f"OOD {vood.macro_f1:.3f} > test: class balance differs, so the\nshift is confounded - not better generalisation",
                      f"Without augmentation the same ViT reaches\n{noa.macro_f1:.3f}: augmentations must preserve fine geometry"],
                 0.03, top - 0.05, fs=10.2, dy=0.1)

    ax, top = panel(fig, 0.722, 0.07, 0.27, 0.28, "Hypotheses & conclusion", "7")
    yy = top - 0.05
    for hid, stmt, verdict in hypotheses(R):
        ok = verdict.startswith("Supported")
        ax.text(0.03, yy, hid, fontsize=9.6, fontweight="bold", color=NAVY, transform=ax.transAxes, va="top")
        ax.text(0.11, yy, stmt, fontsize=9.2, color=NAVY, transform=ax.transAxes, va="top")
        ax.text(0.97, yy, verdict.split(":")[0], fontsize=9.2, fontweight="bold", ha="right",
                color="#1f6f2a" if ok else ACCENT, transform=ax.transAxes, va="top")
        yy -= 0.085
    bullets(ax, ["Working, reproducible image pipeline - ready for HFSS data",
                 "Next: >= 1,000 HFSS hybrid-slotted DGS designs (PyAEDT)"], 0.03, yy - 0.01, fs=9.4, dy=0.08, color=GREY)

    ft = fig.add_axes([0, 0, 1, 0.062])
    ft.axis("off")
    ft.add_patch(Rectangle((0, 0), 1, 1, fc="#eef4fa", transform=ft.transAxes))
    ft.text(0.5, 0.66, "Explainable image-based screening of compact patch antennas for 5G / 6G - simulation evidence only, not HFSS-validated",
            ha="center", fontsize=13, fontweight="bold", color=NAVY)
    ft.text(0.5, 0.33, "Data: LaBash et al. (2025) arXiv:2505.18188 [MIT]  |  Bhuma et al. (2025) RadPat-50K, doi:10.5281/zenodo.17067881 [CC BY 4.0]",
            ha="center", fontsize=8.6, color=GREY)
    ft.text(0.5, 0.08, "ViT: Dosovitskiy et al. (2021)  |  Attention rollout: Abnar & Zuidema (2020)  |  Calibration: Guo et al. (2017)  |  Code: " + REPO_URL,
            ha="center", fontsize=8.6, color=GREY)

    out = ROOT / "poster"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "module1_poster.png", dpi=300)
    fig.savefig(out / "module1_poster.pdf")
    plt.close(fig)
    print("poster written")


if __name__ == "__main__":
    main()
