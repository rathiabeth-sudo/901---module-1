"""Unit tests for label rules, S11 feature extraction and geometry rendering."""
import numpy as np

from src.dataset import geometry_masks, render_geometry
from src.preprocessing import band_label, perf_label, s11_features
from src.utils import load_config

CFG = load_config()


def test_s11_features_first_matched_resonance():
    f = np.linspace(1e9, 10e9, 1000)
    # two dips: shallow-but-matched at 3 GHz (-15 dB), deeper at 8 GHz (-30 dB)
    s = -1 - 14 * np.exp(-((f - 3e9) / 0.05e9) ** 2) - 29 * np.exp(-((f - 8e9) / 0.05e9) ** 2)
    r = s11_features(f, s)
    assert abs(r["f_res_ghz"] - 3.0) < 0.02          # first matched dip, not the deepest
    assert abs(r["f_deepest_ghz"] - 8.0) < 0.02
    assert r["s11_min_db"] < -29.5
    assert r["bw10_mhz"] > 0


def test_s11_features_unmatched():
    f = np.linspace(1e9, 10e9, 1000)
    s = -1 - 5 * np.exp(-((f - 5e9) / 0.2e9) ** 2)
    r = s11_features(f, s)
    assert r["bw10_mhz"] == 0 and r["fbw10_pct"] == 0
    assert abs(r["f_res_ghz"] - 5.0) < 0.02


def test_perf_label_thresholds():
    assert perf_label(-5.0, CFG) == 0
    assert perf_label(-10.0, CFG) == 1     # boundary: -10 dB counts as matched
    assert perf_label(-19.9, CFG) == 1
    assert perf_label(-20.0, CFG) == 2


def test_band_label_edges():
    assert band_label(3.5, CFG) == 0
    assert band_label(5.0, CFG) == 1
    assert band_label(7.124, CFG) == 1
    assert band_label(7.125, CFG) == 2


def test_render_deterministic_and_size():
    a = np.asarray(render_geometry(20.0, 30.0, -3.0))
    b = np.asarray(render_geometry(20.0, 30.0, -3.0))
    assert a.shape == (224, 224, 3)
    assert np.array_equal(a, b)


def test_render_absolute_scale_preserves_size_order():
    small = np.asarray(render_geometry(10.0, 15.0, -2.0))[:112]
    large = np.asarray(render_geometry(40.0, 60.0, -2.0))[:112]
    copper = lambda im: (np.abs(im.astype(int) - [196, 134, 52]).sum(-1) < 60).sum()
    assert copper(large) > 10 * copper(small)


def test_feed_position_changes_image():
    a = np.asarray(render_geometry(20.0, 30.0, -1.0)).astype(int)
    b = np.asarray(render_geometry(20.0, 30.0, -5.0)).astype(int)
    assert np.abs(a - b).sum() > 0


def test_masks_inside_frame():
    pm, fm = geometry_masks(30.0, 45.0, -4.0)
    assert pm.shape == (224, 224) and pm.any() and fm.any()
    assert pm.mean() < 0.8
