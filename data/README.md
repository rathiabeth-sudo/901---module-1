# data/

| Path | Content | In Git? |
|---|---|---|
| `raw/` | Original downloads (Parquet, Zenodo ZIP/CSV). Recreate with `python -m src.download_data` | no |
| `processed/patch_designs.csv` | One row per patch design: parameters, S11-derived targets, labels, split, image path, MD5 | yes |
| `processed/patch_images/` | 2,051 standardised 224×224 geometry images (derived from MIT data) | yes |
| `processed/radpat_designs.csv` | 3,000 sampled RadPat rows with `hpbw_class` and split | yes |
| `processed/radpat_images/` | Resized polar plots (CC BY 4.0). Regenerated locally | no |
| `processed/preprocessing_log.json` | Every exclusion, duplicate and artefact count, plus thresholds | yes |
| `splits/*.csv` | Locked split manifests (design IDs and labels) | yes |
| `data_dictionary.csv` | Variable definitions, types and units | yes |

See `docs/DATA_CARD.md` for label rules and `docs/DATASET_SURVEY.md` for source verification.
