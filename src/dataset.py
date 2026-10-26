"""Geometry rendering and PyTorch datasets.

Standardised geometry image (224 x 224), two stacked panels:
  * top    (224 x 112): ABSOLUTE-scale top view, fixed 1.962 px/mm for every design, so the
                         physical size of the patch (which sets the resonant frequency) is visible;
  * bottom (224 x 112): NORMALISED detail view, the patch scaled to fill the panel, so the
                         relative feed position (which sets impedance matching) is resolvable.
Fixed background, fixed colours, no text, legends or labels (prevents shortcut learning).
The image is a deterministic drawing of the simulated design parameters (L, W, feed offset);
it is NOT an HFSS/openEMS field export.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

RENDER_CONSTANTS = {
    "abs_max_width_mm": 106.0,
    "abs_max_length_mm": 53.0,
    "panel_height_px": 112,
    "abs_margin_px": 4,
    "norm_margin_px": 12,
    "feed_radius_px": 2.2,
    "background_rgb": [236, 239, 243],
    "patch_rgb": [196, 134, 52],
    "feed_rgb": [20, 20, 20],
    "separator_rgb": [130, 130, 130],
}
BG = tuple(RENDER_CONSTANTS["background_rgb"])


def _layout(L: float, W: float, size: int = 224):
    """Return panel rectangles and feed centres in final-pixel coordinates."""
    c = RENDER_CONSTANTS
    ph = c["panel_height_px"] * size / 224
    s_abs = min((size - 2 * c["abs_margin_px"]) / c["abs_max_width_mm"],
                (ph - 2 * c["abs_margin_px"]) / c["abs_max_length_mm"])
    s_norm = min((size - 2 * c["norm_margin_px"]) / W, (ph - 2 * c["norm_margin_px"]) / L)
    panels = []
    for s, cy in [(s_abs, ph / 2), (s_norm, ph + ph / 2)]:
        cx = size / 2
        rect = (cx - W * s / 2, cy - L * s / 2, cx + W * s / 2, cy + L * s / 2)
        panels.append((s, cx, cy, rect))
    return panels, ph


def render_geometry(L: float, W: float, feed_y: float, size: int = 224, supersample: int = 4) -> Image.Image:
    c = RENDER_CONSTANTS
    big = size * supersample
    im = Image.new("RGB", (big, big), BG)
    d = ImageDraw.Draw(im)
    panels, ph = _layout(L, W, size)
    k = supersample
    for s, cx, cy, rect in panels:
        d.rectangle([v * k for v in rect], fill=tuple(c["patch_rgb"]))
        fx, fy = cx, cy - feed_y * s  # feed_y < 0 -> below centre (image y grows downward)
        r = c["feed_radius_px"] * k
        d.ellipse([fx * k - r, fy * k - r, fx * k + r, fy * k + r], fill=tuple(c["feed_rgb"]))
    d.line([0, ph * k, big, ph * k], fill=tuple(c["separator_rgb"]), width=k)
    return im.resize((size, size), Image.LANCZOS)


def geometry_masks(L: float, W: float, feed_y: float, size: int = 224, feed_halo_px: float = 6.0):
    """Boolean masks (size x size) of the patch area and a halo around the feed, both panels."""
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    patch = np.zeros((size, size), bool)
    feed = np.zeros((size, size), bool)
    panels, _ = _layout(L, W, size)
    for s, cx, cy, (x0, y0, x1, y1) in panels:
        patch |= (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        fy = cy - feed_y * s
        feed |= (xx - cx) ** 2 + (yy - fy) ** 2 <= feed_halo_px ** 2
    return patch, feed


# --------------------------------------------------------------------------------------
# PyTorch datasets
# --------------------------------------------------------------------------------------

def build_transforms(train: bool, mean, std, fill=BG, hflip: bool = True):
    from torchvision import transforms as T

    ops = []
    if train:
        # conservative, physically motivated: the patch is mirror-symmetric about its length axis
        if hflip:
            ops.append(T.RandomHorizontalFlip())
        ops.append(T.RandomAffine(degrees=3, translate=(0.02, 0.02), fill=fill))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


class ImageTable:
    """Minimal torch-style dataset over a DataFrame with image_path and a label column."""

    def __init__(self, df: pd.DataFrame, label_col: str, root, transform):
        self.paths = [root / x for x in df.image_path]
        self.labels = df[label_col].astype(int).to_numpy()
        self.ids = df.design_id.to_numpy()
        self.transform = transform
        self._cache = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = self._cache.get(i)
        if im is None:
            im = Image.open(self.paths[i]).convert("RGB")
            im.load()
            self._cache[i] = im
        return self.transform(im), int(self.labels[i])


def load_table(dataset: str, root) -> pd.DataFrame:
    return pd.read_csv(root / "data" / "processed" / f"{dataset}_designs.csv")
