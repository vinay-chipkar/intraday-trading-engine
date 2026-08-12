"""The Parquet warehouse's integrity ledger.

Two small JSON files live at the warehouse root:

- `_schema_version.json` -- which SCHEMA_VERSION wrote this warehouse.
- `manifest.json` -- one entry per partition file ever written: table, zone,
  partition label, row count, and a sha256 of the file's bytes. Both
  persist and restore treat this file as authoritative: persist skips
  re-exporting a partition whose row count already matches, and restore
  refuses to load anything whose file is missing or whose checksum/row
  count disagrees with what's recorded here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION_FILENAME = "_schema_version.json"


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path) -> dict[str, dict]:
    path = root / MANIFEST_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_manifest(root: Path, manifest: dict[str, dict]) -> None:
    path = root / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def load_schema_version(root: Path) -> dict | None:
    path = root / SCHEMA_VERSION_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_schema_version(root: Path, version: int) -> None:
    path = root / SCHEMA_VERSION_FILENAME
    path.write_text(json.dumps({"schema_version": version}, indent=2))
