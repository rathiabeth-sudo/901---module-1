# Known limitations - Module 1

## Scope and evidence
1. **No HFSS.** Ansys HFSS was not available and was not run. No result in this repository is an HFSS result.
2. **Topology mismatch.** The image dataset contains plain coax-fed rectangular patches on one substrate (εr 3.68, h 1.61 mm),
   with no slots and no DGS. Conclusions do **not** transfer to hybrid-slotted DGS antennas without new data.
3. **Simulation-only evidence.** openEMS FDTD (patch) and analytical array factors (RadPat). No measurements, fabrication tolerances,
   connector losses or material variability.
4. **Geometry images are drawings.** They are rendered deterministically from the exact simulated parameters and contain none of the noise,
   legends or viewpoint changes of real solver screenshots, which makes the task easier than a real HFSS-export pipeline.
5. **No field images.** Surface-current and E-field modalities, central to the original plan, could not be tested.

## Labels and data
6. The **−20 dB** boundary between "Acceptable" and "High" is a project assumption and needs engineering confirmation.
7. Class imbalance: about two-thirds of the in-distribution designs are "High".
8. The patch sweep is dense, so test designs have close neighbours in training (interpolation). The OOD hold-out (W/L > 1.95) is the only
   extrapolation test; there is no topology or substrate hold-out because the data contain only one of each.
9. The dataset documentation differs from the observed data (parameter ranges, εr); both are recorded in `DATASET_SURVEY.md`.
10. RadPat is a different task (beamwidth class) on a different dataset, so the modality comparison (RQ3) is descriptive, not causal.

## Modelling and compute
11. CPU only. Partial fine-tuning (last 4 of 12 blocks), early-stopping patience 3 (planned: 8), no hyperparameter grid search.
12. Three seeds for the main task only; the other tasks have one seed.
13. Attention rollout is model attribution, **not** electromagnetic causation.

## What full validation requires
* ≥ 1,000 parameterised hybrid-slotted DGS designs simulated in HFSS (PyAEDT automation), with converged meshes and solver logs.
* Geometry, surface-current and E-field images exported at fixed frequency, camera, colour map and magnitude range, with no legends.
* Engineering sign-off on the label thresholds and target bands.
* Hold-out slot topologies and substrates.
