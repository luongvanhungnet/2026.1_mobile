"""Fetch the exact official Multi-Weather checkpoint; never silently overwrite files.

Usage: python tools/download_gridformer.py --output-dir models/source
Only gdown is needed. Conversion is a separate Linux-only step.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen
import zipfile

COMMIT = "d0ec11c2bf3bb0f09e00059006523c4c44c535a0"
FILE_ID = "1aNyEQPHFviCYr_H_rrz8m3mqlRUGqEuR"
EXPECTED_BYTES = 243056049
SOURCE_URL = f"https://raw.githubusercontent.com/TaoWangzj/GridFormer/{COMMIT}/basicsr/archs/GridFormer_arch.py"
CONFIG_URL = f"https://raw.githubusercontent.com/TaoWangzj/GridFormer/{COMMIT}/options/test/GridFormer/Multi_weather_Restoration/GridFormer_mixed.yml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_text(url: str, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}. Use another output directory.")
    with urlopen(Request(url, headers={"User-Agent": "GridFormer-Mobile-BTL/1"}), timeout=60) as response:
        content = response.read()
    if len(content) < 100 or b"<html" in content[:100].lower():
        raise ValueError(f"Unexpected source response from {url}")
    target.write_bytes(content)


def inspect_checkpoint(path: Path) -> dict:
    if path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(f"Checkpoint size {path.stat().st_size:,} != verified official size {EXPECTED_BYTES:,}. It may be incomplete or changed.")
    # torch.save stores ZIP archives. No pickle is loaded or executed here.
    if not zipfile.is_zipfile(path):
        raise ValueError("Expected a PyTorch ZIP checkpoint, received another file format.")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not any(name.endswith("/data.pkl") for name in names):
            raise ValueError("Archive does not contain PyTorch data.pkl.")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC validation failed: {bad}")
    return {"bytes": path.stat().st_size, "sha256": sha256(path), "zip_crc_passed": True, "entries": len(names)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/source"))
    parser.add_argument("--inspect-only", type=Path, help="Validate an existing official checkpoint without downloading")
    args = parser.parse_args()
    if args.inspect_only:
        print(json.dumps(inspect_checkpoint(args.inspect_only), indent=2))
        return 0
    try:
        import gdown
    except ImportError:
        parser.error("Install download dependencies: python -m pip install -r tools/requirements-download.txt")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "net_g_best.pth"
    if checkpoint.exists() or (directory / "download_manifest.json").exists():
        raise FileExistsError("Output directory already has downloaded artifacts. Inspect them or choose another directory.")
    fetch_text(SOURCE_URL, directory / "GridFormer_arch.py")
    fetch_text(CONFIG_URL, directory / "GridFormer_mixed.yml")
    partial = directory / "net_g_best.pth.partial"
    if partial.exists():
        raise FileExistsError(f"Refusing to reuse an unverified partial download: {partial}")
    result = gdown.download(id=FILE_ID, output=str(partial), quiet=False)
    if not result:
        raise RuntimeError("Google Drive did not return a checkpoint. Download manually using the official folder link.")
    validation = inspect_checkpoint(partial)
    partial.replace(checkpoint)
    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "https://github.com/TaoWangzj/GridFormer",
        "source_commit": COMMIT,
        "checkpoint_drive_id": FILE_ID,
        "checkpoint": validation,
        "architecture_sha256": sha256(directory / "GridFormer_arch.py"),
        "config_sha256": sha256(directory / "GridFormer_mixed.yml"),
        "strict_pytorch_load_verified": False,
        "tflite_conversion_verified": False,
    }
    (directory / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"DOWNLOAD FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
