# Dataset discovery and verification (Phase 2)

> Shared survey for both modules. Module 1 uses candidates selected for images (patch geometry, RadPat); Module 2 uses the patch data and the Mendeley Antenna Parameters Dataset.

The search covered Hugging Face, Kaggle, Mendeley Data, IEEE DataPort, Zenodo and GitHub, in the order
the project brief prescribes. Search and verification date: **25 October 2026**. All metadata below was read
directly from each repository's API or landing page on that date. Fields that could not be verified
are marked "not verified".

## Selection criteria

1. Relevance to the research question: images, or exact geometry, linked to simulated electromagnetic outputs at the level of individual designs.
2. A clear licence that permits research use and redistribution of derived files.
3. Provenance: named authors, a version or DOI, and a documented solver.
4. Enough samples for a design-level 70/15/15 split.
5. Labels that can be derived from verified simulation outputs using explicit rules.

## Candidates

| # | Official title / ID | Repository & version | Authors / publisher | Licence | Size & format | Features / targets | Decision |
|---|---|---|---|---|---|---|---|
| 1 | **Rectangular Patch Antenna Frequency Response Dataset** (`becklabash/rectangular-patch-antenna-freq-response`) | Hugging Face, revision `4be72a22e55fbf46966c59a5d6b3ec95c836379b` | LaBash et al., released with arXiv:2505.18188 | MIT | 2,051 designs; one Parquet file (9.8 MB) | Patch L, W and feed offset; full S11 curve, 1,000 points from 1–10 GHz; openEMS FDTD; fixed substrate | **Selected.** Module 1 primary dataset (geometry images rendered from exact parameters), Module 2 and fusion |
| 2 | **RadPat-50K: A Large-Scale Benchmark for Antenna Array Radiation Patterns** | Zenodo record 17067881, v1.1, DOI 10.5281/zenodo.17067881 (published 2025-09-06) | Bhuma C. M., Sure P., Painam S. K., Chandrasekhar Rao J., Srinivasa Rao U. | CC BY 4.0 | 50,000 patterns; polar and rectangular JPG plots plus CSV metadata (10K subset: 495 MB polar) | ULA parameters (N, spacing, steering, taper, noise); directivity, HPBW, main-lobe angle | **Selected.** Radiation-pattern modality (beamwidth class); 3,000-sample subset |
| 3 | **Antenna Parameters Dataset** | Mendeley Data `3gxr2vvd9n`, **version 2**, DOI 10.17632/3gxr2vvd9n.2 (2024-06-25). The reference report cites v1 | Mihad M. F. H., Islam M. M., Sojib F. H. (Hajee Mohammad Danesh S&T University) | CC BY 4.0 | 1,648 rows = 103 designs × 16 frequencies; XLSX (SHA-256 `7c15f2a3…`) | Substrate, patch, feed and two slot dimensions; gain, directivity, S11, VSWR, radiation and total efficiency | **Selected.** Module 2, dual-slot patch regression |
| 4 | Antenna-Image-Semantics | GitHub `DENGFEIYANG/Antenna-Image-Semantics`, commit `88239f9f` | DENGFEIYANG | MIT | 59 PNG images, 58 parsable JSON files | Concept or mask images across many topologies; relative EM classes (gain, FBW, pattern) | Rejected. Too few samples for a 70/15/15 split; heterogeneous concept renders; labels are qualitative |
| 5 | `infinite-dataset-hub/*` (MSPA_Antenna_Review, AntennaPatch5G, AntennaParameterTrainingSet) | Hugging Face | infinite-dataset-hub | MIT | < 1K rows, CSV | Text and tabular | Rejected. Self-declared "AI-generated … may be inaccurate or false" |
| 6 | `CJJones/LLM_Training_Antenna_Design` | Hugging Face | CJJones | CC BY-NC-SA 4.0 | 1K–10K rows, text | Text Q&A | Rejected. Text only, non-commercial licence |
| 7 | `moondream/ishape-antenna` | Hugging Face | moondream (subset of iShape) | not stated on card | 370 images, Parquet | Instance masks of antenna objects in scenes | Rejected. Object segmentation with no RF performance labels |
| 8 | Dataset for Antenna Array Pattern with Sidelobe Level Control Using Deep Learning | Zenodo 14533762 (2024-12-20) | Abdullah M. A. | CC BY 4.0 | 2.9 GB CSV | Array pattern samples | Not used. Tabular array data, no geometry images; size exceeds the CPU budget |
| 9 | Measurement and simulation data for Direct and Residual Surrogate Models for a Triple-Band Koch Monopole Antenna | Zenodo 23088565 (2026-10-01) | Göçen C. | CC BY 4.0 | files not inspected | Measured and simulated monopole data | Not used. Different topology; noted for future measured-vs-simulated validation |
| 10 | Kaggle antenna-parameter dataset (cited in the reference report) | Kaggle | not verified | Reference report lists the licence as "unknown" | not verified | HFSS patch and slot dimensions, S11 | Not accessed. Requires an authenticated Kaggle API; unclear licence |
| 11 | IEEE DataPort microstrip S-parameter / 2-D pattern collections | IEEE DataPort | not verified | Access-controlled | not verified | S-parameters, radiation patterns | Not accessed. Requires an institutional login or subscription |

## Why no HFSS image dataset was used

No public, licensed dataset was found that contains HFSS-exported geometry, surface-current or E-field
images of **hybrid-slotted DGS** patch antennas with linked performance labels. This matches the reference report's own conclusion, which recommends generating the dataset locally.

Ansys HFSS was **not available** in this environment and **was not run**. Following the project's
dataset rule, the prototype therefore:

* uses real simulation data (openEMS FDTD; analytically computed array factors) instead of HFSS,
* renders a standardised geometry image of each *simulated* design from its exact parameters. This is a
  drawing of the geometry, not a field image, and is never presented as an HFSS export,
* documents the additional HFSS data needed for full validation (see `LIMITATIONS.md`).

## Discrepancies found during verification (recorded, not corrected at source)

* Patch dataset: the README states L ∈ [7.5, 52.5] mm and p ∈ [−6, 0) mm. The observed values are L ∈ [7.5, 47.5] mm and
  p ∈ [−15.83, 0] mm, and 100 designs are centre-fed (p = 0). The README gives εr = 3.68, while the diagram in the same
  repository shows εr = 3.38. The README value is the one we adopted.
* Patch dataset: 1,170 designs have a maximum S11 slightly above 0 dB (at most +0.20 dB). This is a numerical artefact of the
  passive-port calculation. The rows were kept because the targets rely on the minima.
* RadPat: 5,015 of the 10,000 rows report `D_peak_dBi = -120`, a sentinel value that coincides with grating-lobe cases.
  Directivity was therefore **not** used as a label; HPBW was used instead.
* Mendeley: 1 non-numeric `Total_eff` entry (`"7.695490e"`, a truncated number). 155 rows have directivity above 20 dBi,
  90 rows have radiation efficiency above 1, and 90 rows have gain above directivity. In total, **245 rows** are flagged as
  physically implausible solver artefacts for a ~6 mm antenna and are excluded from the gain target. The version cited in the reference report (v1) has been superseded by v2.
