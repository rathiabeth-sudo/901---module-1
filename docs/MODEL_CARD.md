# Model card: ViT-S/16 antenna-geometry classifier (Module 1)

## Model details
| Item | Value |
|---|---|
| Architecture | ViT-S/16 (`timm/vit_small_patch16_224.augreg_in21k_ft_in1k`), 21.7 M parameters with a 3-class head |
| Fine-tuning (final, pre-registered) | 2 head-only epochs, then the last 4 blocks + norm + head (7.1 M trainable); AdamW, lr 3e-5 (head 3e-4), wd 0.01, batch 16, class-weighted CE, early stopping on validation macro-F1 (patience 3), temperature scaling |
| Inputs | 224×224 standardised two-panel geometry image (absolute-scale + normalised top view, no text) |
| Outputs | Calibrated probabilities over {Poor, Acceptable, High} matching classes (or band / beamwidth classes), a 384-d CLS embedding, attention-rollout maps |
| Baselines | Majority class; compact CNN (0.25 M parameters); frozen-backbone linear probe |
| Developed | October 2026, applied AI research workshop prototype, CPU only |

## Intended use
Research and teaching: rapid **screening** of simulated patch-antenna designs, and a methodological template for the planned
HFSS hybrid-slotted DGS study. **Not** for certifying, procuring or deploying antennas, and not for any safety-relevant decision.

## Evaluation (locked test split, n = 262 designs; see `results/metrics/summary_table.csv`)
| Model (matching class) | Seeds | Macro-F1 | Balanced acc. |
|---|---|---|---|
| Majority baseline | 1 | 0.269 | 0.333 |
| Compact CNN | 3 | 0.643 ± 0.022 | 0.669 |
| ViT-S/16 linear probe | 1 | 0.709 | 0.729 |
| **ViT-S/16 fine-tuned (pre-registered, final)** | 3 | **0.582 ± 0.008** | 0.649 |
| ViT-S/16 fine-tuned, no augmentation (exploratory ablation) | 1 | 0.725 | 0.737 |
| ViT-S/16 revised optimiser (exploratory, Amendment 1) | 1 | 0.646 | 0.660 |

Other tasks with the pre-registered ViT (seed 42): band class 0.685 (linear probe 0.811); RadPat beamwidth 0.881 (linear probe 0.884).
OOD hold-out (W/L > 1.95): 0.679 ± 0.027, confounded by a more balanced class mix in the OOD set.
In Module 2, an MLP trained on this model's frozen embeddings reached 0.775, which supports the underfitting diagnosis.

## Known failure modes
* **Underfitting** under the pre-registered settings: training-time rotation and translation blur the 1–2 px feed offset that carries the matching information.
* The "Acceptable" class is weak (F1 0.37) because it is a narrow band between two S11 thresholds.
* Sensitive to blur and to the copper colour (macro-F1 drops to 0.53 and 0.52; about 17% of predictions change).
* Attention shows background "sink" tokens unrelated to physics. The feed region is enriched ×1.41, the patch only ×1.03.

## Ethical considerations
Simulation data only, no personal data. Confidence is calibrated on simulated validation data and does not guarantee real-world
performance. Attention maps are attribution, not causation. The intended scope is civilian communications research.

## Training data
See `docs/DATA_CARD.md`: openEMS rectangular patch sweep (MIT) and RadPat-50K subset (CC BY 4.0).
