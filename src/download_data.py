"""Download the two public image-module datasets into data/raw (pinned versions, checksums recorded).

Usage:  python -m src.download_data [--skip-radpat]
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile

from .utils import load_config, p, save_json


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists: {dest.name}")
        return
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "applied-vit-antenna-research"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-radpat", action="store_true", help="skip the 495 MB radiation-pattern archive")
    args = ap.parse_args()
    cfg = load_config()
    raw = cfg["paths"]["raw"]
    record = {}

    c = cfg["datasets"]["patch"]
    url = f"https://huggingface.co/datasets/{c['repo_id']}/resolve/{c['revision']}/{c['file']}"
    dest = p(raw, "patch", "train.parquet")
    fetch(url, dest)
    record["patch"] = {"url": url, "sha256": sha256(dest), "licence": "MIT"}

    c = cfg["datasets"]["radpat"]
    base = f"https://zenodo.org/api/records/{c['record']}/files"
    dest = p(raw, "radpat", c["metadata_file"])
    fetch(f"{base}/{c['metadata_file']}/content", dest)
    record["radpat_metadata"] = {"sha256": sha256(dest), "licence": "CC-BY-4.0"}
    if not args.skip_radpat:
        z = p(raw, "radpat", c["images_zip"])
        fetch(f"{base}/{c['images_zip']}/content", z)
        out = p(raw, "radpat", "polar", "x").parent
        if not any(out.rglob("*.jpg")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
        record["radpat_images"] = {"sha256": sha256(z), "licence": "CC-BY-4.0"}

    save_json(record, p(raw, "download_record.json"))
    print("done")


if __name__ == "__main__":
    main()
