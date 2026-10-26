"""Apply the Amendment-1 adoption rule (validation data only) and record the final ViT configuration.

Rule: the revised config is adopted if its seed-42 validation macro-F1 exceeds both the pre-registered
fine-tune and the frozen linear probe on the main task; otherwise the pre-registered config remains final.
Test metrics are NOT read by this script.

Usage:  python -m src.select_final
"""
from __future__ import annotations

from .utils import load_json, p, save_json


def main():
    pre = load_json(p("results", "metrics", "vit_finetune_patch_perf_class_s42.json"))["best_val_macro_f1"]
    lp = load_json(p("results", "metrics", "vit_linearprobe_patch_perf_class.json"))["best_val_macro_f1"]
    rev = load_json(p("results", "metrics", "vit_finetune_revised_patch_perf_class_s42.json"))["best_val_macro_f1"]
    adopt = rev > pre and rev > lp
    out = {
        "rule": "adopt revised iff val macro-F1 (seed 42) > pre-registered AND > linear probe",
        "val_macro_f1": {"preregistered": pre, "linear_probe": lp, "revised": rev},
        "adopted_profile": "revised" if adopt else "preregistered",
        "final_mode": "finetune_revised" if adopt else "finetune",
        "final_run_main": ("vit_finetune_revised_patch_perf_class_s42" if adopt else "vit_finetune_patch_perf_class_s42"),
    }
    save_json(out, p("results", "metrics", "final_model.json"))
    print(out)


if __name__ == "__main__":
    main()
