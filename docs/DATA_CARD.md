# Data card - Module 1 (image module)

| Item | Patch geometry (primary) | RadPat polar plots |
|---|---|---|
| Source | Hugging Face `becklabash/rectangular-patch-antenna-freq-response` @ `4be72a2` | Zenodo 10.5281/zenodo.17067881 (v1.1), `Polar_10K.zip` + `Metadata_10K.csv` |
| Licence | MIT | CC BY 4.0 |
| Evidence type | **Simulated** (openEMS FDTD, εr = 3.68, h = 1.61 mm) | **Synthetic / analytical** (uniform linear array factor with amplitude, phase and steering noise) |
| Unit of analysis | One antenna design = one rendered geometry image | One array configuration = one polar plot |
| Samples used | 2,051 designs (0 excluded) | 3,000 sampled at random (seed 42) from 10,000 (0 corrupted, 0 duplicates) |
| Inputs | 224×224 standardised two-panel geometry image | 750×750 JPG resized to 224×224 PNG (LANCZOS) |
| Labels | `perf_class` (main task), `band_class` (rules below) | `hpbw_class` (training-split HPBW terciles: Narrow ≤ 6.48°, Medium 6.48–17.82°, Wide > 17.82°) |
| Split | Design level, stratified 70/15/15 (seed 42) after the OOD hold-out W/L > 1.95: train 1,220 / val 262 / test 262 / OOD 307 | Random 70/15/15: 2,100 / 450 / 450 |
| Redistributed in repo | Derived CSV + rendered PNGs (MIT) | Metadata-derived CSV only; images regenerated locally |

The tabular Mendeley Antenna Parameters Dataset has no images, so (following the brief's dataset rule) it is used only in Module 2.

## Label rules (frozen before training; see `configs/default.yaml`)

* **perf_class** (from the global minimum of simulated S11 over 1–10 GHz)
  * 0 *Poor*: S11min > −10 dB, so no impedance match anywhere in band
  * 1 *Acceptable*: −20 dB < S11min ≤ −10 dB
  * 2 *High*: S11min ≤ −20 dB
  * −10 dB (VSWR ≈ 1.92) is the conventional impedance-bandwidth criterion. The −20 dB boundary is a **project
    assumption** and needs engineering confirmation before the classes are treated as physically validated.
* **band_class** (from the first matched resonance, i.e. the lowest-frequency local S11 minimum below −10 dB; falls back to the global minimum if the design is never matched)
  * 0: < 5.0 GHz (covers 5G NR n77/n78/n79 and lower)
  * 1: 5.0–7.125 GHz (upper FR1; 7.125 GHz is the FR1 upper edge in 3GPP TS 38.101-1)
  * 2: ≥ 7.125 GHz (FR3 region, discussed as a 6G candidate band)

## Standardised geometry image

Two panels with no text, legend, axis or label:

* **Top** (224×112): absolute-scale top view at a fixed 1.962 px/mm for every design. Patch size, which drives resonance, is visible.
* **Bottom** (224×112): the same patch scaled to fill the panel. The relative feed position, which drives matching, can be resolved.

Background (RGB 236/239/243), copper patch (196/134/52) and black feed (2.2 px radius) are identical for every design.
Images are rendered at 4× resolution and downsampled with LANCZOS, so anti-aliasing preserves sub-pixel feed position.
Each image is a deterministic **drawing of the simulated geometry**. It is not an exported field plot and is not HFSS output.

## Quality control performed

* Non-finite or incomplete S11 curves: 0. Duplicate parameter triples: 0. Identical rendered images (MD5): 0.
* Perceptual-hash collisions are frequent (21,561 pairs) because the images are simple rectangles, so pHash was judged unsuitable for
  near-duplicate grouping; the split relies on exact design identity.
* RadPat: every image passed PIL `verify()` and a blank-image check; MD5 found 0 duplicates.
* All counts are in `data/processed/preprocessing_log.json`; tests are in `tests/`.

## Intended use and out-of-scope use

Research and teaching on image-based antenna screening. **Not** suitable for certifying any antenna design.
The patch dataset contains one topology (a coax-fed rectangular patch) and one substrate, with no DGS or slots,
so conclusions do not transfer to hybrid-slotted DGS antennas without new HFSS data.
