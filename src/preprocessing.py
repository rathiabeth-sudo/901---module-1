"""Module 1 data preparation for the two image datasets.

* Patch (openEMS) : derive S11 targets, define labels, render standardised geometry images,
                    check duplicates, create OOD hold-out and design-level 70/15/15 split.
* RadPat          : sub-sample, quality-check, resize polar plots, split, derive beamwidth labels
                    from training-split terciles only.
* Data dictionary : written to data/data_dictionary.csv.

Usage:  python -m src.preprocessing [--skip-radpat]
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .dataset import render_geometry, RENDER_CONSTANTS
from .utils import load_config, p, save_json

# --------------------------------------------------------------------------------------
# Patch dataset
# --------------------------------------------------------------------------------------


def s11_features(freq_hz: np.ndarray, s11_db: np.ndarray, thr: float = -10.0) -> dict:
    """Global minimum S11 plus the FIRST matched resonance and its -10 dB bandwidth.

    The deepest dip is often a higher-order mode, so the resonant frequency is defined as the
    lowest-frequency local minimum below `thr` (fundamental matched resonance). If the design is
    never matched, the global minimum is used and the bandwidth is 0.
    """
    g = int(np.argmin(s11_db))
    smin = float(s11_db[g])
    is_min = np.r_[False, (s11_db[1:-1] <= s11_db[:-2]) & (s11_db[1:-1] <= s11_db[2:]), False]
    cand = np.where(is_min & (s11_db < thr))[0]
    j = int(cand[0]) if len(cand) else g
    f_res = float(freq_hz[j] / 1e9)
    bw_mhz = 0.0
    if s11_db[j] < thr:
        lo = j
        while lo > 0 and s11_db[lo - 1] < thr:
            lo -= 1
        hi = j
        while hi < len(s11_db) - 1 and s11_db[hi + 1] < thr:
            hi += 1
        bw_mhz = float((freq_hz[hi] - freq_hz[lo]) / 1e6)
    return {"f_res_ghz": f_res, "s11_min_db": smin, "s11_at_res_db": float(s11_db[j]),
            "f_deepest_ghz": float(freq_hz[g] / 1e9), "bw10_mhz": bw_mhz,
            "fbw10_pct": 100.0 * bw_mhz / (f_res * 1e3) if bw_mhz > 0 else 0.0}


def perf_label(s11_min: float, cfg: dict) -> int:
    if s11_min > cfg["labels"]["s11_matched_db"]:
        return 0  # Poor: no -10 dB match anywhere in 1-10 GHz
    if s11_min > cfg["labels"]["s11_high_db"]:
        return 1  # Acceptable
    return 2      # High


def band_label(f_res: float, cfg: dict) -> int:
    return int(np.searchsorted(cfg["labels"]["band_edges_ghz"], f_res, side="right"))


def prepare_patch(cfg: dict, log: dict) -> pd.DataFrame:
    raw = pd.read_parquet(p(cfg["paths"]["raw"], "patch", "train.parquet"))
    log["patch_raw_rows"] = len(raw)
    rows, excluded = [], []
    for r in raw.itertuples(index=False):
        f = np.asarray(r.frequencies, dtype=float)
        s = np.asarray(r.s11, dtype=float)
        if len(f) != 1000 or not np.all(np.isfinite(s)):
            excluded.append({"id": int(r.id), "reason": "missing or non-finite S11"})
            continue
        feats = s11_features(f, s, cfg["labels"]["s11_matched_db"])
        rows.append({"design_id": f"P{int(r.id):04d}", "source_id": int(r.id),
                     "length_mm": float(r.length), "width_mm": float(r.width), "feed_y_mm": float(r.feed_y),
                     "s11_max_db": float(s.max()), **feats})
    df = pd.DataFrame(rows)
    log["patch_excluded"] = excluded
    log["patch_positive_s11_designs"] = int((df.s11_max_db > 0).sum())  # small solver artefact, documented
    dup = df.duplicated(subset=["length_mm", "width_mm", "feed_y_mm"])
    log["patch_duplicate_parameter_rows"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)

    df["wl_ratio"] = df.width_mm / df.length_mm
    df["feed_ratio"] = -df.feed_y_mm / (df.length_mm / 2)  # 0 = centre, 1 = patch edge
    df["area_mm2"] = df.width_mm * df.length_mm
    df["perf_class"] = [perf_label(v, cfg) for v in df.s11_min_db]
    df["band_class"] = [band_label(v, cfg) for v in df.f_res_ghz]

    # ---- render standardised geometry images
    img_dir = p(cfg["paths"]["processed"], "patch_images", "x").parent
    img_dir.mkdir(parents=True, exist_ok=True)
    hashes = []
    for r in df.itertuples():
        im = render_geometry(r.length_mm, r.width_mm, r.feed_y_mm, cfg["render"]["size"], cfg["render"]["supersample"])
        arr = np.asarray(im)
        assert arr.std() > 5, "blank render"
        path = img_dir / f"{r.design_id}.png"
        im.save(path)
        hashes.append(hashlib.md5(arr.tobytes()).hexdigest())
    df["image_path"] = [f"data/processed/patch_images/{d}.png" for d in df.design_id]
    df["md5"] = hashes
    log["patch_exact_duplicate_images"] = int(df.duplicated("md5").sum())
    try:
        import imagehash

        ph = [imagehash.phash(Image.open(p(x))) for x in df.image_path]
        near = sum(1 for i in range(len(ph)) for j in range(i + 1, len(ph)) if ph[i] - ph[j] == 0)
        log["patch_identical_phash_pairs"] = int(near)
    except ImportError:
        pass

    # ---- split: OOD hold-out first, then stratified design-level split; identical images share a group
    ood = df.wl_ratio > cfg["split"]["ood_wl_ratio_min"]
    df["split"] = ""
    df.loc[ood, "split"] = "ood"
    pool = df[~ood]
    groups = pool.drop_duplicates("md5")
    seed = cfg["seed"]
    tr, rest = train_test_split(groups, test_size=1 - cfg["split"]["train"], stratify=groups.perf_class, random_state=seed)
    va, te = train_test_split(rest, test_size=cfg["split"]["test"] / (cfg["split"]["val"] + cfg["split"]["test"]),
                              stratify=rest.perf_class, random_state=seed)
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        df.loc[df.md5.isin(part.md5) & ~ood, "split"] = name
    assert (df.split != "").all()

    df.to_csv(p(cfg["paths"]["processed"], "patch_designs.csv"), index=False)
    for name in ["train", "val", "test", "ood"]:
        df.loc[df.split == name, ["design_id", "perf_class", "band_class"]].to_csv(
            p(cfg["paths"]["splits"], f"patch_{name}_ids.csv"), index=False)
    log["patch_split_counts"] = df.split.value_counts().to_dict()
    log["patch_perf_by_split"] = pd.crosstab(df.split, df.perf_class).to_dict()
    log["patch_band_by_split"] = pd.crosstab(df.split, df.band_class).to_dict()
    log["render_constants"] = RENDER_CONSTANTS
    return df


# --------------------------------------------------------------------------------------
# RadPat dataset
# --------------------------------------------------------------------------------------


def prepare_radpat(cfg: dict, log: dict) -> pd.DataFrame:
    meta = pd.read_csv(p(cfg["paths"]["raw"], "radpat", cfg["datasets"]["radpat"]["metadata_file"]))
    log["radpat_raw_rows"] = len(meta)
    log["radpat_D_sentinel_minus120"] = int((meta.D_peak_dBi == -120).sum())
    img_root = p(cfg["paths"]["raw"], "radpat", "polar", "polar", "x").parent
    sub = meta.sample(n=cfg["datasets"]["radpat"]["subsample"], random_state=cfg["seed"]).reset_index(drop=True)
    out_dir = p(cfg["paths"]["processed"], "radpat_images", "x").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    keep, md5s, excluded = [], [], []
    for r in sub.itertuples():
        src = img_root / r.image_polar
        try:
            im = Image.open(src)
            im.verify()
            im = Image.open(src).convert("RGB")
        except Exception as e:  # corrupted or missing
            excluded.append({"id": int(r.id), "reason": f"unreadable: {e}"})
            continue
        arr = np.asarray(im)
        if arr.std() < 5:
            excluded.append({"id": int(r.id), "reason": "blank image"})
            continue
        md5s.append(hashlib.md5(arr.tobytes()).hexdigest())
        im.resize((224, 224), Image.LANCZOS).save(out_dir / f"R{int(r.id):05d}.png")
        keep.append(r.Index)
    sub = sub.loc[keep].copy()
    sub["md5"] = md5s
    sub["design_id"] = [f"R{int(i):05d}" for i in sub.id]
    sub["image_path"] = [f"data/processed/radpat_images/{d}.png" for d in sub.design_id]
    log["radpat_excluded"] = excluded
    log["radpat_exact_duplicates"] = int(sub.duplicated("md5").sum())
    sub = sub.drop_duplicates("md5")

    seed = cfg["seed"]
    tr, rest = train_test_split(sub, test_size=0.30, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, random_state=seed)
    sub["split"] = "train"
    sub.loc[va.index, "split"] = "val"
    sub.loc[te.index, "split"] = "test"
    # beamwidth class thresholds from the TRAINING split only
    q = np.quantile(sub.loc[sub.split == "train", "HPBW_deg"], [1 / 3, 2 / 3])
    sub["hpbw_class"] = np.searchsorted(q, sub.HPBW_deg, side="right")
    log["radpat_hpbw_thresholds_deg"] = q.tolist()
    log["radpat_split_counts"] = sub.split.value_counts().to_dict()
    log["radpat_hpbw_by_split"] = pd.crosstab(sub.split, sub.hpbw_class).to_dict()
    sub.to_csv(p(cfg["paths"]["processed"], "radpat_designs.csv"), index=False)
    for name in ["train", "val", "test"]:
        sub.loc[sub.split == name, ["design_id", "hpbw_class"]].to_csv(p(cfg["paths"]["splits"], f"radpat_{name}_ids.csv"), index=False)
    return sub


DATA_DICTIONARY = [
    # dataset, variable, type, unit, description
    ("patch", "design_id", "string", "-", "Unique design key (P + source id); one image per design"),
    ("patch", "length_mm", "float", "mm", "Patch length L (resonant dimension)"),
    ("patch", "width_mm", "float", "mm", "Patch width W"),
    ("patch", "feed_y_mm", "float", "mm", "Coaxial feed offset from patch centre along L (negative toward edge)"),
    ("patch", "f_res_ghz", "float", "GHz", "First matched resonance: lowest-frequency local S11 minimum below -10 dB (global minimum if never matched)"),
    ("patch", "f_deepest_ghz", "float", "GHz", "Frequency of the deepest S11 dip in 1-10 GHz (often a higher-order mode)"),
    ("patch", "s11_min_db", "float", "dB", "Global minimum simulated S11 over 1-10 GHz (openEMS FDTD); defines perf_class"),
    ("patch", "s11_at_res_db", "float", "dB", "Simulated S11 at f_res"),
    ("patch", "bw10_mhz", "float", "MHz", "Contiguous -10 dB bandwidth around resonance (0 if unmatched)"),
    ("patch", "fbw10_pct", "float", "%", "Fractional -10 dB bandwidth"),
    ("patch", "wl_ratio", "float", "-", "W/L aspect ratio"),
    ("patch", "feed_ratio", "float", "-", "Feed offset normalised by L/2 (0 centre, 1 edge)"),
    ("patch", "area_mm2", "float", "mm^2", "Patch area W x L"),
    ("patch", "perf_class", "int", "0/1/2", "Poor (S11min > -10 dB), Acceptable (-20 < S11min <= -10), High (S11min <= -20 dB)"),
    ("patch", "band_class", "int", "0/1/2", "f_res < 5 GHz | 5-7.125 GHz | >= 7.125 GHz (3GPP FR1 / FR3 boundary)"),
    ("patch", "split", "category", "train/val/test/ood", "Locked design-level allocation (seed 42)"),
    ("patch", "image_path", "string", "-", "Rendered standardised geometry image (224x224 PNG)"),
    ("radpat", "design_id", "string", "-", "R + RadPat sample id"),
    ("radpat", "N", "int", "-", "Number of array elements"),
    ("radpat", "spacing_wavelengths", "float", "lambda", "Element spacing"),
    ("radpat", "weights", "category", "-", "Amplitude taper (uniform, binomial, cosine, kaiser, hamming, hann, blackman, exponential)"),
    ("radpat", "steering_deg_nominal", "float", "deg", "Nominal steering angle"),
    ("radpat", "HPBW_deg", "float", "deg", "Half-power beamwidth reported by dataset authors"),
    ("radpat", "D_peak_dBi", "float", "dBi", "Peak directivity; value -120 is a sentinel (not used as label)"),
    ("radpat", "hpbw_class", "int", "0/1/2", "Narrow/Medium/Wide beam from training-split HPBW terciles"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-radpat", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    log: dict = {}
    df = prepare_patch(cfg, log)
    print("patch:", df.shape, log["patch_split_counts"])
    if not args.skip_radpat:
        rp = prepare_radpat(cfg, log)
        print("radpat:", rp.shape, log["radpat_split_counts"])
    pd.DataFrame(DATA_DICTIONARY, columns=["dataset", "variable", "type", "unit", "description"]).to_csv(
        p("data", "data_dictionary.csv"), index=False)
    save_json(log, p(cfg["paths"]["processed"], "preprocessing_log.json"))


if __name__ == "__main__":
    main()
