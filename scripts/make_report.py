"""Module 1 methodology & implementation report -> reports/module1_report.docx + .tex.

All quantitative statements are computed from executed results. Run after `python -m src.evaluate`.
"""
from __future__ import annotations

import pandas as pd

from deliverables_common import COMPANION_URL, FIG, REPO_URL, ROOT, SUBTITLE, TITLE, hypotheses, load
from report_render import to_docx, to_tex


class Counter:
    def __init__(self):
        self.f = 0
        self.t = 0

    def fig(self, text):
        self.f += 1
        return f"Figure {self.f}. {text}"

    def tab(self, text):
        self.t += 1
        return f"Table {self.t}. {text}"


def build(R) -> list:
    C = Counter()
    row, fmt = R["row"], R["fmt"]
    vit, cnn = row("patch", "perf_class", "finetune"), row("patch", "perf_class", "cnn")
    lp, maj = row("patch", "perf_class", "linear_probe"), row("patch", "perf_class", "majority")
    vb, cb, lb, mb = (row("patch", "band_class", m) for m in ["finetune", "cnn", "linear_probe", "majority"])
    vr, cr, lr_, mr = (row("radpat", "hpbw_class", m) for m in ["finetune", "cnn", "linear_probe", "majority"])
    vood, cood, lood = (row("patch", "perf_class", m, "ood") for m in ["finetune", "cnn", "linear_probe"])
    vbood = row("patch", "band_class", "finetune", "ood")
    rev = row("patch", "perf_class", "finetune_revised")
    noa = row("patch", "perf_class", "finetune_noaug")
    cna = row("patch", "perf_class", "cnn_noaug")
    e, st, prep, v42 = R["expl"], R["stats"], R["prep"], R["vit42"]
    H = hypotheses(R)
    rob = e["robustness"]
    mc = st["mcnemar_finetune_vs_cnn_perf_class_s42"]
    mcl = st["mcnemar_finetune_vs_linearprobe_perf_class_s42"]
    ci = lambda k: st[f"bootstrap95_macro_f1_{k}"]
    ci_v, ci_c = ci("vit_finetune_patch_perf_class_s42"), ci("cnn_patch_perf_class_s42")
    ci_n, ci_r, ci_cn = ci("vit_finetune_noaug_patch_perf_class_s42"), ci("vit_finetune_revised_patch_perf_class_s42"), ci("cnn_noaug_patch_perf_class_s42")
    p_nc = st["mcnemar_finetune_noaug_vs_cnn_perf_class_s42"]["exact_p_value"]
    p_nl = st["mcnemar_finetune_noaug_vs_linearprobe_perf_class_s42"]["exact_p_value"]
    fin = R["final"]["val_macro_f1"]
    pc = v42["splits"]["test"]["per_class"]
    runs = pd.read_csv(ROOT / "results" / "metrics" / "all_runs.csv")
    pbs = prep["patch_perf_by_split"]
    dist = lambda s: "/".join(str(pbs[c][s]) for c in ["0", "1", "2"])
    high_share = pbs["2"]["train"] / sum(pbs[c]["train"] for c in ["0", "1", "2"])
    chg_blur = 1 - rob["gaussian_blur_r1"]["agreement_with_clean"]
    chg_col = 1 - rob["copper_to_grey"]["agreement_with_clean"]
    nparams = v42["total_params"] / 1e6
    sc = prep["patch_split_counts"]
    thr = prep["radpat_hpbw_thresholds_deg"]

    B = []
    a = B.append

    # ------------------------------------------------------------------ Abstract
    a(("h1", "Abstract"))
    a(("p", "This report documents the implementation and evaluation of Module 1 of a two-module applied-AI framework for "
            "hybrid-slotted defected ground structure (DGS) compact patch antennas: an image-based Vision Transformer (ViT) that "
            "classifies antenna performance from standardised images and explains which image regions drive its predictions. "
            "No licensed public dataset of Ansys HFSS images of hybrid-slotted DGS antennas exists, and HFSS was not available, so "
            "the module was built and executed as a proof of concept on two verified public simulation datasets: 2,051 "
            "openEMS-simulated rectangular patch antennas (MIT licence) and 3,000 RadPat-50K radiation-pattern plots (CC BY 4.0). "
            "Every patch design was rendered as a text-free, standardised two-panel geometry image and labelled from its simulated "
            "S11 response using rules frozen before training. A ViT-S/16 pretrained on ImageNet-21k was partially fine-tuned on a CPU "
            f"and compared with a majority-class baseline, a compact CNN and a frozen-backbone linear probe on a locked design-level split. "
            f"On the three-class impedance-matching task the pre-registered fine-tuned ViT reached a test macro-F1 of {fmt(vit)} "
            f"(three seeds), against {fmt(cnn)} for the CNN, {lp.macro_f1:.3f} for the linear probe and {maj.macro_f1:.3f} for the "
            "majority baseline. Training curves showed underfitting, and the augmentation ablation identified the cause: small "
            "rotations and shifts blurred the one-to-two-pixel feed offset that determines matching. Without augmentation the same "
            f"ViT reached {noa.macro_f1:.3f} (exploratory, one seed). Attention rollout placed {e['median_feed_enrichment']:.2f} times more "
            "attribution on the feed region than its area share in the median design, but also revealed background attention sinks. "
            "Robustness tests showed sensitivity to blur and colour rendering. The results establish a reproducible, honest image "
            "pipeline that is ready to be applied to the HFSS dataset; they do not validate hybrid-slotted DGS performance."))
    a(("p", "**Keywords:** Vision Transformer; patch antenna; defected ground structure; explainability; attention rollout; 5G/6G."))

    # ------------------------------------------------------------------ 1 Introduction
    a(("h1", "1. Introduction and Recap of the Research Problem"))
    a(("p", "Compact printed patch antennas are central to 5G and emerging 6G devices. Slots and defected ground structures reshape "
            "surface-current paths and can tune resonance, impedance matching and bandwidth [4], but small geometric changes can produce "
            "large electromagnetic differences. Full-wave solvers such as Ansys HFSS give physically grounded answers, yet large "
            "design sweeps are slow. The Module 1 research report proposed a complementary tool: a pretrained Vision Transformer [1] "
            "that reads standardised antenna geometry and electromagnetic field images, predicts a documented performance class, and "
            "shows which regions of the image influence each prediction. The tool is a rapid screening aid, not a replacement for "
            "full-wave simulation."))
    a(("p", "The research objective of this module is unchanged from that report: **to investigate whether standardised antenna "
            "geometry and electromagnetic images can support antenna-performance classification and explainable visual analysis.** "
            "Module 1 is the image branch of the overall framework. The numerical-text surrogate, the multimodal fusion stage and the "
            "NSGA-II optimisation form Module 2, which is documented in a separate report and repository "
            f"({COMPANION_URL}). The only artefact Module 1 passes to Module 2 is the 384-dimensional image embedding of each design."))
    a(("table", C.tab("Implementation status of Module 1."), ["Component", "Status", "Evidence in repository"], [
        ["Dataset discovery and licence verification", "Completed", "docs/DATASET_SURVEY.md"],
        ["Preprocessing, standardised images, labels, locked splits", "Completed and executed", "src/preprocessing.py, src/dataset.py, data/"],
        ["ViT-S/16 partial fine-tuning and frozen linear probe", "Completed and executed", "src/train_vit.py, results/metrics/"],
        ["Majority and compact CNN baselines", "Completed and executed", "src/train_baseline.py"],
        ["Statistical comparison (McNemar, bootstrap CI)", "Completed and executed", "src/evaluate.py"],
        ["Attention rollout, robustness and OOD tests", "Completed and executed", "src/explainability.py"],
        ["Embedding export for Module 2 fusion", "Completed and executed", "results/features/"],
        ["HFSS hybrid-slotted DGS images (geometry, current, E-field)", "Proposed (HFSS unavailable)", "Section 11"],
        ["Block diagram (Figure 1)", "Illustrative", "results/figures/module1_architecture.png"],
    ]))
    a(("figure", FIG / "module1_architecture.png", C.fig("Module 1 block diagram. Solid boxes were implemented and executed; dashed orange boxes are proposed work that requires Ansys HFSS. The thumbnails are real model inputs."), 6.3))

    # ------------------------------------------------------------------ 2 Research design
    a(("h1", "2. Research Design and Hypotheses"))
    a(("p", "The study follows an applied, experimental machine-learning design. The independent variables are the model family "
            "(majority class, compact CNN, frozen ViT with a linear probe, partially fine-tuned ViT), the image modality (geometry "
            "drawing or radiation-pattern plot) and the evaluation distribution (in-distribution test split or out-of-distribution "
            "hold-out). The dependent variables are classification metrics on simulated performance classes. The research "
            "questions, hypotheses, label rules, split and seed were committed to Git before any fine-tuned result existed "
            "(docs/HYPOTHESES.md). One frozen linear-probe test score had been printed beforehand; this is disclosed in that file."))
    a(("table", C.tab("Research questions and their operationalisation."), ["RQ", "Question", "Operationalisation"], [
        ["RQ1", "Can standardised antenna images predict documented performance classes?", "Geometry image → matching class and band class"],
        ["RQ2", "Does a pretrained ViT outperform a compact CNN?", "Same split, three seeds each, exact McNemar test, bootstrap CI"],
        ["RQ3", "Which available image modality gives the strongest signal?", "Geometry vs radiation-pattern images (descriptive only: different datasets)"],
        ["RQ4", "Do attention maps highlight physically relevant regions?", "Rollout mass inside analytic patch and feed masks vs their area share"],
        ["RQ5", "How robust are the predictions?", "OOD hold-out (W/L > 1.95) and seven rendering perturbations"],
    ]))
    a(("p", "The directional hypotheses tested on the locked test split were:"))
    a(("bullets", [f"**{h[0]}**: {h[1]}." for h in H]))
    a(("p", "A sixth pre-registered hypothesis (H6: fusing image embeddings with numerical features improves on the image-only "
            "classifier) concerns the fusion stage and is reported in the Module 2 report."))

    # ------------------------------------------------------------------ 3 Dataset
    a(("h1", "3. Dataset Description and Source Verification"))
    a(("p", "Repositories were searched in the order set by the brief (Hugging Face, Kaggle, Mendeley Data, IEEE DataPort, GitHub), "
            "with Zenodo added. Eleven candidates were recorded with title, repository, version, authors, licence, size, features "
            "and limitations (docs/DATASET_SURVEY.md). Candidates were judged on relevance to the research question. Self-declared "
            "AI-generated datasets, text-only corpora, non-commercial licences and sets too small for a 70/15/15 split were rejected. "
            "No public, licensed dataset of HFSS images of hybrid-slotted DGS antennas with linked performance labels was found. "
            "Following the brief's dataset rule, tabular data without images (such as the Mendeley Antenna Parameters Dataset) were "
            "assigned to Module 2 and are not presented here as images."))
    a(("table", C.tab("Datasets used by Module 1 (verified 25 October 2026)."), ["Dataset", "Source / version", "Licence", "Content", "Role in Module 1"], [
        ["Rectangular Patch Antenna Frequency Response [2]", "Hugging Face becklabash/rectangular-patch-antenna-freq-response, rev. 4be72a2", "MIT",
         "2,051 coax-fed patches; L, W, feed offset; 1,000-point S11 (1–10 GHz), openEMS FDTD; εr 3.68, h 1.61 mm", "Geometry images, matching and band labels"],
        ["RadPat-50K [3]", "Zenodo 10.5281/zenodo.17067881, v1.1", "CC BY 4.0",
         "Uniform-linear-array polar plots with metadata; 10K polar subset, 3,000 sampled", "Radiation-pattern image modality, beamwidth labels"],
    ]))
    a(("p", "**Nature of the evidence.** The patch data are full-wave FDTD simulations produced with openEMS [11]. RadPat consists of analytically computed array "
            "factors with injected noise, so it is synthetic. None of the data is measured, and no HFSS output is claimed."))
    a(("p", "**Discrepancies recorded during verification.** The patch dataset README gives L ∈ [7.5, 52.5] mm and p ∈ [−6, 0) mm; the "
            "observed values are L ∈ [7.5, 47.5] mm and p ∈ [−15.83, 0] mm, including 100 centre-fed designs. The README states "
            f"εr = 3.68 while the repository diagram shows 3.38. {prep['patch_positive_s11_designs']} S11 curves exceed 0 dB by at most "
            f"0.2 dB, a passive-port numerical artefact. In RadPat, {prep['radpat_D_sentinel_minus120']:,} of 10,000 rows carry a −120 dBi "
            "directivity sentinel, so directivity was not used as a label."))

    # ------------------------------------------------------------------ 4 Preprocessing
    a(("h1", "4. Data Preprocessing and Preparation"))
    a(("h2", "4.1 Targets and label rules"))
    a(("p", "Each simulated S11 curve was reduced to the global minimum S11 (S11min), the **first matched resonance** f_res (the "
            "lowest-frequency local minimum below −10 dB), S11 at f_res and the contiguous −10 dB bandwidth. The first matched "
            "resonance was used because the deepest dip is often a higher-order mode; for example, a 22.5 mm patch showed its deepest "
            "dip at 9.6 GHz. Labels were defined by explicit rules frozen before training:"))
    a(("bullets", ["**Matching class** (main task): Poor if S11min > −10 dB; Acceptable if −20 dB < S11min ≤ −10 dB; High if S11min ≤ −20 dB. "
                   "−10 dB (VSWR ≈ 1.9) is the conventional impedance-bandwidth criterion. The −20 dB boundary is a **project assumption** "
                   "that requires engineering confirmation before the classes are treated as physically validated.",
                   "**Band class** (secondary task): f_res < 5 GHz (covers NR n77/n78/n79), 5–7.125 GHz (upper FR1), ≥ 7.125 GHz (FR3, a 6G "
                   "candidate range), with edges taken from 3GPP TS 38.101-1 [5].",
                   f"**Beamwidth class** (RadPat): half-power beamwidth terciles computed on the training split only (≤ {thr[0]:.2f}°, "
                   f"≤ {thr[1]:.2f}°, above)."]))
    a(("h2", "4.2 Standardised geometry images"))
    a(("p", "Each design was rendered as a 224 × 224 RGB image with two panels. The top panel is an absolute-scale top view at a fixed "
            "1.962 px/mm, so physical size, which governs resonance, is visible. The bottom panel shows the same patch scaled to fill "
            "the frame, so the relative feed position, which governs matching, can be resolved at the 16-pixel patch size of the ViT. "
            "Background, copper and feed colours are fixed, and there is no text, legend, axis or filename, which removes label "
            "leakage through annotations. Rendering at 4× and downsampling with a Lanczos filter preserves sub-pixel feed position "
            "through anti-aliasing. These images are drawings of the simulated geometry, not field exports. RadPat polar plots were "
            "verified, converted to RGB and resized to 224 × 224."))
    a(("figure", FIG / "dataset_overview.png", C.fig("Class distribution by split, design space (patch length vs first matched resonance; dashed lines are band edges) and example standardised inputs."), 6.3))
    a(("figure", FIG / "radpat_samples.png", C.fig("Example RadPat-50K polar plots from the training split with their beamwidth class."), 5.6))
    a(("h2", "4.3 Cleaning, duplicates and leakage control"))
    a(("bullets", [f"Patch: {prep['patch_raw_rows']:,} rows, 0 non-finite curves, {prep['patch_duplicate_parameter_rows']} duplicate parameter triples and "
                   f"{prep['patch_exact_duplicate_images']} identical rendered images (MD5). Identical images would have been forced into one split.",
                   f"Perceptual hashing produced {prep['patch_identical_phash_pairs']:,} colliding pairs because the images are simple rectangles, "
                   "so it was judged unsuitable for near-duplicate grouping. Splitting relies on exact design identity; near-neighbour "
                   "interpolation is discussed as a limitation.",
                   f"RadPat: all {sum(prep['radpat_split_counts'].values()):,} sampled images passed PIL verification and a blank-image check; "
                   f"{prep['radpat_exact_duplicates']} exact duplicates.",
                   f"Split: every design with W/L > 1.95 (n = {sc['ood']}) was removed first as an out-of-distribution hold-out. The rest was "
                   f"split by design, stratified on matching class, 70/15/15 with seed 42 (train {sc['train']:,}, validation {sc['val']}, "
                   f"test {sc['test']}). Each design has exactly one image, so no design can appear in two splits. Manifests are in data/splits.",
                   "Normalisation used the checkpoint's statistics (mean = std = 0.5 for the AugReg ViT). No statistic was computed on the test split.",
                   "Training-only augmentation: horizontal mirror (justified because the patch is symmetric about its length axis), rotation "
                   "of ±3° and translation of up to 2%. Scale augmentation was excluded because it would change the apparent physical size "
                   "and therefore the label."]))
    a(("p", "The data dictionary is in Appendix A. Every count and threshold is logged in data/processed/preprocessing_log.json, and "
            "the rules are enforced by 18 automated tests (Section 10)."))

    # ------------------------------------------------------------------ 5 Model
    a(("h1", "5. Proposed Model: Vision Transformer"))
    a(("p", f"The primary model is ViT-S/16 (timm checkpoint vit_small_patch16_224.augreg_in21k_ft_in1k; {nparams:.1f} M parameters "
            "with the three-class head) [1, 6]. A 224 × 224 image is split into 14 × 14 = 196 non-overlapping 16 × 16 patches. These "
            "are linearly embedded to 384 dimensions, prepended with a classification (CLS) token, given learned positional "
            "embeddings and processed by 12 pre-norm transformer blocks with six-head self-attention. A new linear head (dropout 0.1) "
            "maps the CLS token to class logits. Self-attention lets the feed region attend directly to the patch edges, which are "
            "not adjacent in the image; the reference report identified this as the property that makes a ViT attractive for "
            "antenna layouts. ViT-Small (the size class of DeiT-Small [10]) was chosen over ViT-Base because the dataset is small and specialised and training ran on a CPU."))
    a(("p", "**Outputs.** For each image the model returns temperature-scaled class probabilities, the predicted class, its "
            "confidence and the 384-dimensional CLS embedding. Predictions are saved to results/predictions and embeddings to "
            "results/features; the embeddings of the pre-registered seed-42 model are the input to Module 2's fusion stage."))
    a(("p", "**Explainability.** Attention rollout [7] multiplies the head-averaged attention matrices (plus identity) across all "
            "12 blocks and reads the CLS row. For a quantitative check, an analytic patch mask and a 6-pixel feed halo are computed "
            "from each design's parameters. Enrichment is the share of rollout mass inside a region divided by that region's area "
            "share; a value of 1 means no preference. Attention is treated as model attribution, never as evidence of "
            "electromagnetic causation."))

    # ------------------------------------------------------------------ 6 Training
    a(("h1", "6. Training Strategy and Experimental Setup"))
    a(("table", C.tab("Pre-registered ViT training configuration (configs/default.yaml)."), ["Setting", "Value", "Note"], [
        ["Stage A", "2 epochs, head only, AdamW, lr 1e-3", "Head warm-up"],
        ["Stage B", "≤ 10 epochs, last 4 blocks + norm + head", "Backbone lr 3e-5, head lr 3e-4, one-cycle cosine"],
        ["Batch size / weight decay", "16 / 0.01", "As recommended in the Module 1 report"],
        ["Loss", "Class-weighted cross-entropy", "Weights from the training split only"],
        ["Early stopping", "Patience 3 on validation macro-F1", "Reduced from 8 because of the CPU budget"],
        ["Calibration", "Temperature scaling on validation logits [8]", "Applied before test evaluation"],
        ["Seeds", "42, 43, 44 (main task); 42 elsewhere", "Split fixed at seed 42"],
        ["Hardware", "Intel i7-10610U CPU, 4 threads, PyTorch [9], no GPU", f"≈ {v42['train_seconds'] / 60:.0f} min for the seed-42 run"],
    ]))
    a(("p", "Two deviations from the reference plan were made for feasibility and are recorded here: only the upper four transformer "
            "blocks were unfrozen, and early-stopping patience was 3 epochs instead of 8. A full hyperparameter grid (learning rate × "
            "weight decay × dropout) was not run because one fine-tuning run takes about half an hour on the CPU; the recommended "
            "defaults were used. Models were selected on validation macro-F1, and the test split was evaluated once per final model."))
    a(("p", "**Amendment 1 (disclosed deviation).** The pre-registered seed-42 run underfitted (training accuracy about 0.70 and still "
            f"rising; best validation macro-F1 {fin['preregistered']:.3f}, below the linear probe's {fin['linear_probe']:.3f}). A revised "
            "configuration that changes only optimisation settings (backbone learning rate 1e-4, six unfrozen blocks, 15 epochs, "
            "patience 5) was added with a validation-only adoption rule: it would replace the pre-registered model only if its "
            "validation macro-F1 beat both values. It did not, so the pre-registered model remained final. The augmentation ablation "
            "planned in the reference report was also run. Both follow-ups are labelled exploratory."))

    # ------------------------------------------------------------------ 7 Baselines
    a(("h1", "7. Baseline Methods and Comparative Models"))
    a(("bullets", ["**Majority class**: always predicts the most frequent training class; the minimum useful result.",
                   "**Compact CNN** (trained from scratch, 0.25 M parameters): stride-2 5 × 5 stem, four 3 × 3 conv–BN–ReLU–max-pool blocks "
                   "(32–64–128–128 channels), global average pooling and dropout 0.2; AdamW lr 1e-3, cosine schedule, ≤ 25 epochs, patience 5. "
                   "Same split, augmentation, class weights and calibration as the ViT.",
                   "**Frozen ViT linear probe**: CLS features of the pretrained backbone with multinomial logistic regression; C selected on "
                   "validation macro-F1. It separates what the pretrained representation already knows from what fine-tuning adds.",
                   "**Ablations**: the ViT and the CNN retrained without augmentation (seed 42), and the Amendment 1 optimisation profile."]))

    # ------------------------------------------------------------------ 8 Metrics
    a(("h1", "8. Evaluation Metrics and Analysis Plan"))
    a(("p", "Classification is reported with accuracy, macro and weighted precision, recall and F1, balanced accuracy, one-vs-rest "
            "macro ROC-AUC, the confusion matrix and expected calibration error (ECE, 10 bins). Macro-F1 is the primary metric because "
            "the classes are imbalanced. Results are mean ± SD over seeds where several seeds were run. The paired ViT–CNN comparison "
            "uses an exact McNemar test on test-set correctness [12], and a 95% percentile bootstrap interval [13] (2,000 resamples of test designs) is "
            "given for macro-F1. Robustness is the change in test macro-F1 under each perturbation, together with agreement with the "
            "clean-image predictions. The analysis plan was fixed in docs/HYPOTHESES.md before results were produced."))

    # ------------------------------------------------------------------ 9 Results
    a(("h1", "9. Results and Discussion"))
    a(("p", "Every number in this section comes from executed runs stored in results/metrics. No value is estimated or illustrative."))
    a(("h2", "9.1 Main task: geometry image to matching class (RQ1, RQ2)"))
    rows = []
    for mode, lab in [("majority", "Majority baseline"), ("cnn", "Compact CNN"), ("linear_probe", "ViT-S/16 linear probe"), ("finetune", "ViT-S/16 fine-tuned")]:
        r = row("patch", "perf_class", mode)
        rows.append([lab, int(r.n_seeds), fmt(r, "accuracy"), fmt(r), f"{r.balanced_accuracy:.3f}",
                     "–" if pd.isna(r.roc_auc) else f"{r.roc_auc:.3f}", f"{r.ece:.3f}"])
    a(("table", C.tab(f"Locked test split (n = {sc['test']} designs), matching class. Mean ± SD over seeds."),
       ["Model", "Seeds", "Accuracy", "Macro-F1", "Bal. acc.", "ROC-AUC", "ECE"], rows))
    a(("p", f"All learned models clearly beat the majority baseline (macro-F1 {maj.macro_f1:.3f}), so **H1** holds for the main task. "
            f"The fine-tuned ViT reached {fmt(vit)} against {fmt(cnn)} for the compact CNN (**H2** – {H[1][2]}). The bootstrap 95% "
            f"intervals for seed 42 are [{ci_v[0]:.3f}, {ci_v[1]:.3f}] for the ViT and [{ci_c[0]:.3f}, {ci_c[1]:.3f}] for the CNN. In "
            f"the exact McNemar test (seed 42) the ViT alone was correct on {mc['a_correct_b_wrong']} designs and the CNN alone on "
            f"{mc['a_wrong_b_correct']}. Fine-tuning versus the frozen probe: **H3** – {H[2][2]}."))
    a(("table", C.tab("Per-class test results, pre-registered fine-tuned ViT (seed 42)."), ["Class", "Precision", "Recall", "F1", "Support"],
       [[k, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", v["support"]] for k, v in pc.items()]))
    a(("figure", FIG / "confusion_matrices.png", C.fig("Test confusion matrices (counts and row-normalised recall) for the ViT on the three tasks and the CNN on the main task."), 6.3))
    a(("figure", FIG / "training_curves.png", C.fig("Training curves for seed 42. The dotted line marks the start of partial fine-tuning."), 6.0))
    a(("h2", "9.2 Diagnosing the underperformance: Amendment 1 and augmentation ablation"))
    a(("table", C.tab("Exploratory follow-up runs on the matching task (seed 42, same split; not part of the pre-registered test)."),
       ["Run", "What changed", "Val macro-F1", "Test macro-F1", "Test 95% CI"], [
        ["ViT revised (Amendment 1)", "lr 1e-4, 6 blocks, 15 epochs", f"{fin['revised']:.3f}", f"{rev.macro_f1:.3f}", f"[{ci_r[0]:.3f}, {ci_r[1]:.3f}]"],
        ["ViT, no augmentation", "pre-registered optimiser, augmentation off", f"{R['noaug']['best_val_macro_f1']:.3f}", f"{noa.macro_f1:.3f}", f"[{ci_n[0]:.3f}, {ci_n[1]:.3f}]"],
        ["CNN, no augmentation", "augmentation off", f"{R['cnn_noaug']['best_val_macro_f1']:.3f}", f"{cna.macro_f1:.3f}", f"[{ci_cn[0]:.3f}, {ci_cn[1]:.3f}]"],
    ]))
    a(("p", "Under the pre-registered configuration the fine-tuned ViT scored below both the compact CNN and the frozen linear probe, "
            f"so **H2 and H3 are not supported**; for seed 42 the CNN was significantly better (McNemar p = {mc['exact_p_value']:.4f}). "
            "The training curves explain why: training accuracy plateaued near 0.70 while training loss stayed at or above validation "
            "loss, which indicates underfitting rather than overfitting. Two exploratory runs isolate the cause. (i) The revised "
            f"optimisation settings of Amendment 1 raised test macro-F1 to {rev.macro_f1:.3f} but failed the validation-only adoption "
            f"rule (validation {fin['revised']:.3f} against {fin['linear_probe']:.3f} for the probe). (ii) The augmentation ablation "
            "planned in the Module 1 report was decisive. With the pre-registered optimiser and augmentation switched off, test "
            f"macro-F1 rose to **{noa.macro_f1:.3f}** (95% CI [{ci_n[0]:.3f}, {ci_n[1]:.3f}]). This is statistically indistinguishable "
            f"from the linear probe (McNemar p = {p_nl:.2f}) and numerically, but not significantly, above the CNN (p = {p_nc:.2f}). The "
            f"CNN changed little without augmentation ({cna.macro_f1:.3f}). The most likely mechanism is that ±3° rotation and 2% "
            "translation, applied with bilinear resampling, blur the one-to-two-pixel feed offset in the normalised panel, and that "
            "offset carries most of the matching information. Augmentations chosen because they are physically plausible were "
            "therefore not label-preserving at the image resolution that matters. For every model the Acceptable class was the "
            f"hardest (ViT F1 {pc['Acceptable']['f1']:.3f}); it is a narrow band bounded by two thresholds on S11min, a quantity that "
            "changes sharply with feed position. The Module 2 report independently supports the underfitting diagnosis: an MLP "
            "trained on the frozen embeddings of this same ViT outperforms the ViT's own linear head."))
    a(("h2", "9.3 Secondary task and modality comparison (RQ3)"))
    a(("table", C.tab("Test macro-F1 for the secondary image tasks (seed 42)."), ["Task / input", "Majority", "Compact CNN", "ViT linear probe", "ViT fine-tuned"], [
        ["Geometry → band class", f"{mb.macro_f1:.3f}", f"{cb.macro_f1:.3f}", f"{lb.macro_f1:.3f}", f"{vb.macro_f1:.3f}"],
        ["Radiation pattern → beamwidth class", f"{mr.macro_f1:.3f}", f"{cr.macro_f1:.3f}", f"{lr_.macro_f1:.3f}", f"{vr.macro_f1:.3f}"],
    ]))
    a(("figure", FIG / "model_comparison.png", C.fig("Test macro-F1 by model and task (error bars: SD over three seeds where available)."), 6.0))
    a(("p", f"The band class depends mainly on absolute patch size, which is visible in the top panel. The linear probe reached "
            f"{lb.macro_f1:.3f}, ahead of the fine-tuned ViT ({vb.macro_f1:.3f}) and the CNN ({cb.macro_f1:.3f}). The ViT beat the "
            "majority baseline on both patch tasks, so **H1 is supported**. On radiation-pattern images all learned models performed "
            f"strongly (ViT {vr.macro_f1:.3f}, probe {lr_.macro_f1:.3f}, CNN {cr.macro_f1:.3f}), which fits beamwidth being a global "
            "shape property of the plot. These results do **not** show that radiation patterns are an intrinsically stronger modality: "
            "the RadPat task is a different, easier problem on a different dataset, so RQ3 can only be answered descriptively here. A "
            "controlled comparison needs geometry, current and field images of the same designs, which is what the proposed HFSS "
            "dataset would provide."))
    a(("h2", "9.4 Explainability (RQ4)"))
    a(("figure", FIG / "attention_rollout_examples.png", C.fig("Attention rollout on test images: four correct and four incorrect predictions (model attribution only)."), 6.3))
    a(("figure", FIG / "attention_mean_by_class.png", C.fig("Mean rollout map per true class over the test split."), 5.5))
    a(("p", f"Across the {e['n_test']} test designs the patch covered on average {e['mean_patch_area_share']:.1%} of the image and "
            f"received {e['mean_patch_mass']:.1%} of the rollout mass (median enrichment {e['median_patch_enrichment']:.2f}). The feed halo "
            f"covered {e['mean_feed_area_share']:.2%} and received {e['mean_feed_mass']:.2%} (median enrichment {e['median_feed_enrichment']:.2f}); "
            f"feed enrichment exceeded 1 for {e['share_designs_feed_enrichment_gt1']:.0%} of designs. The top (absolute-scale) panel "
            f"received {e['mean_top_panel_mass']:.0%} of the attribution. Median feed enrichment was {e['feed_enrichment_correct_vs_wrong'][0]:.2f} "
            f"for correct and {e['feed_enrichment_correct_vs_wrong'][1]:.2f} for incorrect predictions (**H4** – {H[3][2]}). The feed "
            "result is the more meaningful one: the halo covers less than 1% of the image yet receives more attribution than its "
            "area share in every test design, which is physically plausible because feed position sets the input impedance. A patch "
            "enrichment close to 1 means attribution over the copper is roughly proportional to its area. The mean maps also show a "
            "systematic hotspot in an empty background corner of the top panel for all classes; such high-attention background "
            "tokens are a documented ViT artefact (attention sinks / register-like tokens [14]) with no electromagnetic meaning. "
            "Together these observations caution against reading attention maps as explanations of physics: they show where the "
            "model looks, not why an antenna performs as it does."))
    a(("h2", "9.5 Robustness and out-of-distribution generalisation (RQ5)"))
    a(("table", C.tab("Robustness of the fine-tuned ViT (seed 42) on the test split."), ["Perturbation", "Macro-F1", "Accuracy", "Agreement with clean"],
       [[k.replace("_", " "), f"{v['macro_f1']:.3f}", f"{v['accuracy']:.3f}", f"{v['agreement_with_clean']:.3f}"] for k, v in rob.items()]))
    a(("p", f"On the OOD hold-out (W/L > 1.95, n = {sc['ood']}, never seen in training) macro-F1 was {fmt(vood)} for the ViT, "
            f"{fmt(cood)} for the CNN and {lood.macro_f1:.3f} for the linear probe; band-class OOD macro-F1 was {vbood.macro_f1:.3f}. "
            f"**H5** – {H[4][2]}. The comparison mixes two effects. The OOD set is far more class-balanced than the test split "
            f"(Poor/Acceptable/High = {dist('ood')} versus {dist('test')}), and macro-F1 rewards balanced recall, so the ViT's relatively "
            "good Poor and Acceptable recall counts for more on the OOD set. Covariate shift (wider patches) is therefore confounded "
            "with label shift, and the higher OOD score should not be read as better generalisation. Among the rendering "
            "perturbations, a 1-pixel Gaussian blur and recolouring the copper to grey lowered macro-F1 the most (to "
            f"{rob['gaussian_blur_r1']['macro_f1']:.3f} and {rob['copper_to_grey']['macro_f1']:.3f}) and changed {chg_blur:.0%} and "
            f"{chg_col:.0%} of predictions. The model therefore relies on fine edges and on the exact colour rendering, while contrast "
            "change and JPEG compression had little effect. Applying it to real solver exports would require the fixed rendering "
            "standard set out in the Module 1 report."))
    a(("h2", "9.6 Summary of hypotheses"))
    a(("table", C.tab("Module 1 hypothesis outcomes on the locked test split."), ["ID", "Hypothesis", "Outcome and evidence"], [list(h) for h in H]))

    # ------------------------------------------------------------------ 10 Ethics
    a(("h1", "10. Ethics, Bias, Privacy, and Reproducibility"))
    a(("p", "**Data rights and privacy.** Both datasets are public under licences that permit redistribution with attribution (MIT, "
            "CC BY 4.0) and contain no personal data. Attribution is given in the LICENSE file, the data card and the references. "
            "Raw archives and resized RadPat images are not redistributed; a script re-downloads them with checksums."))
    a(("p", "**Bias.** (i) Topology and material bias: a single coax-fed rectangular patch on one substrate. (ii) Frequency bias: band "
            f"classes are imbalanced. (iii) Class imbalance: {high_share:.0%} of training designs are High. (iv) Rendering bias: one colour "
            "scheme; probed with a colour-shift test, but the images are synthetic drawings, so real HFSS screenshot artefacts are not "
            "represented. (v) Simulation-to-reality gap: no measurements and no fabrication tolerances. (vi) Selection bias: no "
            "simulation failures were present in, or removed from, the patch data."))
    a(("p", "**Responsible use.** The classifier is a screening aid. Its confidence scores are calibrated on simulated validation data and "
            "are not guarantees of real-world performance. Any design selected with it must be verified by full-wave simulation and, "
            "for deployment, by fabrication and measurement. Attention maps are model attribution, not proof of electromagnetic "
            "causation. Training energy was kept low by using CPU-scale models, early stopping and partial fine-tuning."))
    a(("p", "**Reproducibility.** Fixed seeds; pinned dependency versions; dataset revisions, DOIs and SHA-256 checksums; locked split "
            "manifests; one configuration file; scripts for every stage; a data card and a model card; and a record of exclusions, "
            "discrepancies and deviations. The 18 automated tests cover label rules, S11 feature extraction, rendering determinism, "
            "split disjointness, the OOD rule, physical ranges and the training-only threshold derivation. All 18 pass (Appendix D)."))

    # ------------------------------------------------------------------ 11 Limitations
    a(("h1", "11. Limitations and Future Work"))
    a(("bullets", ["**Topology mismatch.** The image data are plain rectangular patches without slots or DGS, so the results do not transfer "
                   "to hybrid-slotted DGS antennas.",
                   "**No HFSS and no field images.** Surface-current and E-field modalities could not be tested, so the geometry-versus-field "
                   "comparison remains open. The geometry images are deterministic drawings, which favours the image models compared with "
                   "noisy screenshots.",
                   "**Labels.** The −20 dB boundary is an assumption, and the classes come from simulation, not measurement.",
                   "**Compute.** Partial fine-tuning (4 of 12 blocks), patience 3, no hyperparameter grid, and three seeds for the main task only.",
                   "**Near-neighbour interpolation.** The patch sweep is dense, so test designs have close neighbours in training; the OOD hold-out partly addresses this.",
                   "**Modality comparison is confounded.** The radiation-pattern task uses a different dataset and label, so RQ3 is descriptive only."]))
    a(("p", "**Future work.** (1) Build the HSDGS-ViT dataset described in the reference report: at least 1,000 parameterised "
            "hybrid-slotted DGS designs simulated in HFSS and automated with PyAEDT, exporting geometry, surface-current and E-field "
            "images with a fixed camera, colour map and scale and no legends. (2) Retrain the same pipeline unchanged, with "
            "augmentation limited to transforms verified to preserve the feed geometry, and add per-modality and multi-view "
            "experiments. (3) Hold out a whole slot topology and a substrate as OOD tests. (4) Obtain engineering sign-off on the −20 dB "
            "label boundary. (5) With GPU access, run full fine-tuning, the planned hyperparameter grid and five or more seeds."))

    # ------------------------------------------------------------------ 12 Conclusion
    a(("h1", "12. Conclusion"))
    a(("p", "Module 1 implements and executes a complete, reproducible image pipeline for the proposed framework: verified data, "
            "standardised leakage-free images, pre-registered labels and hypotheses, a pretrained ViT with three baselines, statistical "
            "comparison, quantitative explainability and robustness tests. Standardised geometry images carry a learnable signal for "
            f"simulated antenna-performance classes (RQ1): every model beat the majority baseline, the best ViT variant reached a test "
            f"macro-F1 of {noa.macro_f1:.3f} on the matching task (no-augmentation ablation) and frozen ImageNet-21k features reached "
            f"{lb.macro_f1:.3f} on the band task. Under the pre-registered CPU-limited fine-tuning, however, the ViT did not outperform the "
            "compact CNN (RQ2); the decisive factor was augmentation that blurred fine feed geometry, not the architecture. Attention "
            "rollout consistently emphasised the feed region but also showed background artefacts (RQ4), and robustness tests revealed "
            "sensitivity to blur and colour rendering, with the OOD comparison confounded by label shift (RQ5). The module offers three "
            "methodological lessons for the full HFSS study: check that every augmentation preserves the label at the resolution of the "
            "relevant physical feature; always report a frozen-feature baseline; and quantify attribution against known geometry rather "
            "than inspecting heat maps. The pipeline is ready to be pointed at the HSDGS-ViT dataset, which is the step needed to test "
            "the original research claim."))

    # ------------------------------------------------------------------ References
    a(("h1", "References"))
    a(("numbered", REFERENCES))

    # ------------------------------------------------------------------ Appendices
    a(("pagebreak",))
    a(("h1", "Appendix A. Data Dictionary"))
    dd = pd.read_csv(ROOT / "data" / "data_dictionary.csv")
    a(("table", C.tab("Data dictionary (data/data_dictionary.csv)."), ["Dataset", "Variable", "Type", "Unit", "Description"], dd.values.tolist()))
    a(("h1", "Appendix B. Configuration"))
    a(("code", (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")))
    a(("h1", "Appendix C. Experimental Details"))
    allr = runs[runs.split == "test"].copy()
    order = {"majority": 0, "cnn": 1, "cnn_noaug": 2, "linear_probe": 3, "finetune": 4, "finetune_revised": 5, "finetune_noaug": 6}
    allr = allr.assign(o=allr["mode"].map(order)).sort_values(["dataset", "task", "o", "seed"])
    a(("table", C.tab("Every executed classification run (test split)."), ["Dataset", "Task", "Mode", "Seed", "Macro-F1", "Accuracy", "ECE", "Train time (min)"],
       [[r.dataset, r.task, r["mode"], "–" if pd.isna(r.seed) else int(r.seed), f"{r.macro_f1:.3f}", f"{r.accuracy:.3f}", f"{r.ece:.3f}",
         "–" if pd.isna(r.train_seconds) else f"{r.train_seconds / 60:.1f}"] for _, r in allr.iterrows()]))
    a(("p", "Failed or abandoned runs: the first CNN design (four double-convolution blocks at full resolution, about 60 s per epoch) was "
            "stopped after three epochs to save compute and replaced by the stride-2 stem version reported here. A first training "
            "queue was restarted as a detached, resumable process; partial runs were discarded and re-run from scratch. Because of a "
            "launcher quoting error, the pre-registered ViT seeds 43 and 44 and CNN seed 43 ran concurrently with other jobs; their "
            "results are unaffected, but their training times are inflated and should not be compared. The first definition of "
            "resonance (the deepest S11 dip) was replaced before training by the first matched resonance. No test-set result was used "
            "for any of these decisions."))
    a(("h1", "Appendix D. Reproducibility Instructions"))
    a(("code", "python -m venv .venv && .venv\\Scripts\\activate\n"
               "pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu\n"
               "python -m src.download_data          # ~510 MB (add --skip-radpat for ~10 MB)\n"
               "python -m src.preprocessing\npytest -q\n"
               "powershell -File scripts\\run_experiments.ps1      # or: bash scripts/run_experiments.sh  (~4 h on CPU)\n"
               "python scripts/make_deliverables.py"))
    a(("p", f"Repository: {REPO_URL}. The pytest output is stored in results/test_report.txt; environment, runtimes and determinism "
            "notes are in docs/REPRODUCIBILITY.md."))
    return B


REFERENCES = [
    "A. Dosovitskiy et al., \"An image is worth 16x16 words: Transformers for image recognition at scale,\" in Proc. ICLR, 2021, arXiv:2010.11929.",
    "B. LaBash et al., \"Improving generative inverse design of rectangular patch antennas with test time optimization,\" arXiv:2505.18188, 2025. Dataset: huggingface.co/datasets/becklabash/rectangular-patch-antenna-freq-response (MIT), rev. 4be72a2.",
    "C. M. Bhuma, P. Sure, S. K. Painam, J. Chandrasekhar Rao and U. Srinivasa Rao, \"RadPat-50K: A large-scale benchmark for antenna array radiation patterns,\" Zenodo, v1.1, 2025, doi:10.5281/zenodo.17067881 (CC BY 4.0).",
    "C. A. Balanis, Antenna Theory: Analysis and Design, 4th ed. Hoboken, NJ: Wiley, 2016.",
    "3GPP, \"NR; User Equipment (UE) radio transmission and reception; Part 1: Range 1 Standalone,\" TS 38.101-1.",
    "R. Wightman, \"PyTorch Image Models (timm),\" GitHub repository, 2019–2026; A. Steiner et al., \"How to train your ViT? Data, augmentation, and regularization in vision transformers,\" TMLR, 2022.",
    "S. Abnar and W. Zuidema, \"Quantifying attention flow in transformers,\" in Proc. ACL, 2020, pp. 4190–4197.",
    "C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, \"On calibration of modern neural networks,\" in Proc. ICML, 2017.",
    "A. Paszke et al., \"PyTorch: An imperative style, high-performance deep learning library,\" in Proc. NeurIPS, 2019.",
    "H. Touvron et al., \"Training data-efficient image transformers & distillation through attention,\" in Proc. ICML, 2021.",
    "T. Liebig et al., openEMS – a free and open electromagnetic field solver, https://openems.de.",
    "Q. McNemar, \"Note on the sampling error of the difference between correlated proportions or percentages,\" Psychometrika, vol. 12, pp. 153–157, 1947.",
    "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall, 1993.",
    "T. Darcet, M. Oquab, J. Mairal and P. Bojanowski, \"Vision transformers need registers,\" in Proc. ICLR, 2024.",
]

META = {
    "kicker": "Applied AI Research Workshop - Module 1 Methodology and Implementation Report",
    "title": TITLE,
    "subtitle": SUBTITLE,
    "doc_type": "Module 1: Image-Based Vision Transformer for Antenna Performance Classification and Explainable Visual Analysis",
    "fields": [("Student name", "[to be completed by student]"), ("Student ID", "[to be completed by student]"),
               ("Programme / Module", "[to be completed by student]"), ("Institution", "[to be completed by student]"),
               ("Submission date", "[to be completed by student]"), ("Report generated", "26 October 2026"),
               ("Code repository", REPO_URL)],
    "note": "Bracketed fields are deliberate placeholders for personal details that were not provided. All results are from executed "
            "experiments on public simulation data. Ansys HFSS was not available and was not run.",
    "author": "",
}


def main():
    R = load()
    blocks = build(R)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    to_docx(blocks, META, out / "module1_report.docx")
    to_tex(blocks, META, out / "module1_report.tex")
    print("report written")


if __name__ == "__main__":
    main()
