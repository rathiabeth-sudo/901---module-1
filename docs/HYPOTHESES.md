# Pre-registered research questions and hypotheses

> **Module 1 repository note.** This is the original pre-registration, committed before any fine-tuned result existed, kept unchanged. H1-H5 are evaluated in this repository. H6 (fusion) is evaluated in the companion Module 2 repository (`902---Module-2`).

Committed to Git **before** any fine-tuned ViT, CNN, fusion or explainability result existed
(see the commit timestamp of this file). One result had already been printed at that point: the frozen
linear-probe test score for the main task (macro-F1 0.709). It is disclosed here for transparency.
The labels, thresholds, split and seed in `configs/default.yaml` were frozen in the same commit.

## Research questions (adapted from the Module 1 report)

| ID | Question | Operationalisation in this prototype |
|----|----------|--------------------------------------|
| RQ1 | Can standardised antenna geometry images predict documented performance classes? | Patch geometry image -> 3-class matching class (Poor / Acceptable / High) and 3-class band class, labels from openEMS S11 |
| RQ2 | Does a pretrained ViT outperform a compact CNN under the same design-level split? | ViT-S/16 partial fine-tune vs compact CNN, 3 seeds each, same split, McNemar + bootstrap CI |
| RQ3 | Which available image modality provides the strongest signal? | Geometry images (patch, two tasks) vs radiation-pattern images (RadPat, beamwidth class). Different datasets and tasks, so this is a descriptive comparison only |
| RQ4 | Do attention maps focus on physically meaningful regions? | Attention-rollout share inside the analytically known patch area and feed halo, compared with their area share (enrichment ratio) |
| RQ5 | How robust are predictions to geometry / rendering changes? | OOD hold-out (W/L > 1.95, never seen in training) + seven rendering perturbations |

## Hypotheses (directional, tested on the locked test split)

* **H1** The fine-tuned ViT exceeds the majority-class baseline in test macro-F1 on both patch tasks.
* **H2** The fine-tuned ViT's mean test macro-F1 over 3 seeds is at least as high as the compact CNN's mean over 3 seeds on the main task.
* **H3** Partial fine-tuning beats the frozen linear probe in test macro-F1 on the main task.
* **H4** Median attention enrichment is above 1 for the patch region, and also above 1 for the feed halo.
* **H5** Macro-F1 on the OOD hold-out is lower than on the in-distribution test split.
* **H6** Fusing ViT embeddings with numerical design features raises test macro-F1 above the image-only classifier.

Exploratory, with no directional hypothesis: the band task versus the matching task, the radiation-pattern modality,
calibration (ECE) before and after temperature scaling, and the NSGA-II surrogate proposals.

## Decision rules

* Models are selected using validation macro-F1 only. The test split is evaluated once per final model.
* If a hypothesis is not supported, it is reported as not supported. No threshold, label or split is changed after viewing test results.

## Amendment 1 (25 Oct 2026, ~21:55): post-hoc revised fine-tuning configuration

The pre-registered fine-tuning run (seed 42) **underfitted**. Training accuracy was about 0.70 and still rising when training stopped,
and its best validation macro-F1 was 0.660, below the frozen linear probe's validation macro-F1 of 0.702. Its test macro-F1 (0.581) had
already been printed when this amendment was written, which we disclose here.

Actions taken:
* The pre-registered result is kept. **H2 and H3 are judged on the pre-registered configuration** (`mode = finetune`).
* We add one revised configuration, `vit_revised` in `configs/default.yaml` (`mode = finetune_revised`). It changes only optimisation settings:
  learning rate 3e-5 to 1e-4, unfrozen blocks 4 to 6, epochs 10 to 15, patience 3 to 5. Data, labels, split and augmentation are unchanged.
* Adoption rule, fixed now: the revised configuration becomes the "final" ViT for explainability, fusion and the secondary tasks only
  if its seed-42 **validation** macro-F1 exceeds both 0.660 (pre-registered) and 0.702 (linear probe). Otherwise the linear probe is used.
* Revised-configuration test results are labelled **exploratory (post-hoc)** everywhere. They carry a risk of test-set adaptivity because one
  test score had been seen before the revision.
