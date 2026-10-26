"""Load executed Module 1 results for the diagram / poster / slides / report generators.

Every number shown in a deliverable comes from results/metrics/*.json|csv written by the pipeline.
Missing runs raise an error rather than being filled with placeholder values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"

NAVY = "#0b2a4a"
BLUE = "#2f6fae"
LIGHT = "#dbe9f6"
MID = "#9cc3e6"
ACCENT = "#e07b39"
GREEN = "#2e7d4f"
GREY = "#5f6b76"

MODULE = "Module 1 - Image-Based Vision Transformer"
TITLE = "Vision Transformer-Based Visual Analysis of Hybrid-Slotted DGS Compact Patch Antennas"
SUBTITLE = "Module 1 of the Applied Vision Transformer Framework for 5G and Emerging 6G Networks - Proof-of-Concept Prototype"
# Set this once after creating the GitHub repository; every deliverable reads it from here.
REPO_URL = "https://github.com/rathiabeth-sudo/901---module-1"
COMPANION_URL = "https://github.com/rathiabeth-sudo/902---Module-2"


def j(name):
    with open(M / name, encoding="utf-8") as f:
        return json.load(f)


def load() -> dict:
    s = pd.read_csv(M / "summary_table.csv")
    R = {"summary": s}

    def row(ds, task, mode, split="test"):
        r = s[(s.dataset == ds) & (s.task == task) & (s["mode"] == mode) & (s.split == split)]
        if r.empty:
            raise KeyError(f"missing run: {ds} {task} {mode} {split}")
        return r.iloc[0]

    def fmt(r, key="macro_f1"):
        sd = r[f"{key}_sd"] if f"{key}_sd" in r and pd.notna(r.get(f"{key}_sd")) else None
        return f"{r[key]:.3f}" + (f" ± {sd:.3f}" if sd is not None and r["n_seeds"] > 1 else "")

    R["row"], R["fmt"] = row, fmt
    R["stats"] = j("statistical_tests.json")
    R["expl"] = j("explainability_summary.json")
    R["prep"] = json.load(open(ROOT / "data" / "processed" / "preprocessing_log.json", encoding="utf-8"))
    R["vit42"] = j("vit_finetune_patch_perf_class_s42.json")
    R["final"] = j("final_model.json")
    R["noaug"] = j("vit_finetune_noaug_patch_perf_class_s42.json")
    R["cnn_noaug"] = j("cnn_noaug_patch_perf_class_s42.json")
    return R


def hypotheses(R) -> list[tuple[str, str, str]]:
    """Return (id, statement, verdict-with-evidence) for the Module 1 hypotheses H1-H5, computed from results."""
    row = R["row"]
    vit, cnn = row("patch", "perf_class", "finetune"), row("patch", "perf_class", "cnn")
    lp, maj = row("patch", "perf_class", "linear_probe"), row("patch", "perf_class", "majority")
    vb, mb = row("patch", "band_class", "finetune"), row("patch", "band_class", "majority")
    vood = row("patch", "perf_class", "finetune", "ood")
    e = R["expl"]
    out = []
    ok = vit.macro_f1 > maj.macro_f1 and vb.macro_f1 > mb.macro_f1
    out.append(("H1", "ViT > majority baseline (both patch tasks)",
                f"{'Supported' if ok else 'Not supported'}: matching {vit.macro_f1:.3f} vs {maj.macro_f1:.3f}; band {vb.macro_f1:.3f} vs {mb.macro_f1:.3f}"))
    ok = vit.macro_f1 >= cnn.macro_f1
    p = R["stats"].get("mcnemar_finetune_vs_cnn_perf_class_s42", {}).get("exact_p_value")
    out.append(("H2", "ViT ≥ compact CNN (3-seed mean, matching class)",
                f"{'Supported' if ok else 'Not supported'}: {R['fmt'](vit)} vs {R['fmt'](cnn)}"
                + (f"; McNemar p = {p:.3g} (seed 42)" if p is not None else "")))
    ok = vit.macro_f1 > lp.macro_f1
    pl = R["stats"].get("mcnemar_finetune_vs_linearprobe_perf_class_s42", {}).get("exact_p_value")
    out.append(("H3", "Partial fine-tune > frozen linear probe",
                f"{'Supported' if ok else 'Not supported'}: {vit.macro_f1:.3f} vs {lp.macro_f1:.3f}"
                + (f"; McNemar p = {pl:.3g}" if pl is not None else "")))
    pe, fe = e["median_patch_enrichment"], e["median_feed_enrichment"]
    ok = pe > 1 and fe > 1
    weak = " (patch effect negligible)" if ok and pe < 1.1 else ""
    out.append(("H4", "Attention enriched on patch and feed (median > 1)",
                f"{'Supported' if ok else 'Partly supported' if (pe > 1 or fe > 1) else 'Not supported'}: patch ×{pe:.2f}, feed halo ×{fe:.2f}{weak}"))
    ok = vood.macro_f1 < vit.macro_f1
    out.append(("H5", "Lower macro-F1 on OOD (W/L > 1.95)",
                f"{'Supported' if ok else 'Not supported'}: OOD {R['fmt'](vood)} vs test {R['fmt'](vit)} (confounded by class balance)"))
    return out


BRANCH = "main"


def gh(path: str, folder: bool = False) -> str:
    """Clickable GitHub link to a file (blob) or folder (tree) in this repository."""
    return f"{REPO_URL}/{'tree' if folder else 'blob'}/{BRANCH}/{path}"


def link(path: str, folder: bool = False, label: str | None = None) -> str:
    """Report markup [label](url) that report_render turns into a hyperlink."""
    return f"[{label or path}]({gh(path, folder)})"


def test_summary() -> str:
    """Last line of the stored pytest output, e.g. '18 passed in 4.18s'."""
    lines = [l.strip() for l in (ROOT / "results" / "test_report.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[-1] if lines else "not run"
