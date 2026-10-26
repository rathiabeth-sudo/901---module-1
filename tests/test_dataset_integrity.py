"""Integrity tests on the processed datasets and locked split manifests (leakage checks)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import band_label, perf_label
from src.utils import ROOT, load_config

CFG = load_config()
PATCH = ROOT / "data" / "processed" / "patch_designs.csv"
RADPAT = ROOT / "data" / "processed" / "radpat_designs.csv"


@pytest.fixture(scope="module")
def patch():
    if not PATCH.exists():
        pytest.skip("run `python -m src.preprocessing` first")
    return pd.read_csv(PATCH)


def test_unique_design_ids(patch):
    assert patch.design_id.is_unique


def test_splits_are_disjoint_and_complete(patch):
    assert set(patch.split) == {"train", "val", "test", "ood"}
    ids = {s: set(patch.design_id[patch.split == s]) for s in ["train", "val", "test", "ood"]}
    for a in ids:
        for b in ids:
            if a < b:
                assert not ids[a] & ids[b]


def test_split_manifests_match_table(patch):
    for s in ["train", "val", "test", "ood"]:
        m = pd.read_csv(ROOT / "data" / "splits" / f"patch_{s}_ids.csv")
        assert set(m.design_id) == set(patch.design_id[patch.split == s])


def test_no_identical_image_across_splits(patch):
    by = patch.groupby("md5").split.nunique()
    assert (by == 1).all()


def test_ood_rule(patch):
    assert (patch.wl_ratio[patch.split == "ood"] > CFG["split"]["ood_wl_ratio_min"]).all()
    assert (patch.wl_ratio[patch.split != "ood"] <= CFG["split"]["ood_wl_ratio_min"]).all()


def test_split_proportions(patch):
    pool = patch[patch.split != "ood"]
    frac = pool.split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.02
    assert abs(frac["val"] - 0.15) < 0.02
    assert abs(frac["test"] - 0.15) < 0.02


def test_labels_follow_documented_rules(patch):
    assert (patch.perf_class == [perf_label(v, CFG) for v in patch.s11_min_db]).all()
    assert (patch.band_class == [band_label(v, CFG) for v in patch.f_res_ghz]).all()


def test_images_exist_and_are_standardised(patch):
    from PIL import Image

    for path in patch.image_path.sample(50, random_state=0):
        im = Image.open(ROOT / path)
        assert im.size == (224, 224) and im.mode == "RGB"


def test_physical_ranges(patch):
    assert patch.length_mm.between(7.0, 53.0).all()
    # dataset README states p in [-6, 0); observed range is [-15.8, 0] (documented discrepancy)
    assert (patch.feed_y_mm <= 0).all()
    assert (-patch.feed_y_mm < patch.length_mm / 2).all()
    assert patch.f_res_ghz.between(1.0, 10.0).all()


def test_radpat_thresholds_from_train_only():
    if not RADPAT.exists():
        pytest.skip("radpat not prepared")
    r = pd.read_csv(RADPAT)
    log = json.load(open(ROOT / "data" / "processed" / "preprocessing_log.json"))
    q = np.quantile(r.HPBW_deg[r.split == "train"], [1 / 3, 2 / 3])
    assert np.allclose(q, log["radpat_hpbw_thresholds_deg"])
    assert r.design_id.is_unique
