#!/usr/bin/env python3
"""Hash de código, configuración, documentación y resultados entregados."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from noprop_rl.io_utils import atomic_write_json, sha256_file, utc_now


def combined_hash(entries: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    output = PROJECT / "results" / "artifact_manifest.json"
    entries = []
    for path in sorted(PROJECT.rglob("*")):
        if not path.is_file() or path == output or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PROJECT)
        entries.append(
            {"path": str(relative), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    source_entries = [
        entry
        for entry in entries
        if str(entry["path"]).endswith(".py")
        or str(entry["path"]).startswith("configs/")
    ]
    result_entries = [entry for entry in entries if str(entry["path"]).startswith("results/")]
    payload = {
        "generated_at_utc": utc_now(),
        "num_files": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "code_and_config_combined_sha256": combined_hash(source_entries),
        "results_combined_sha256": combined_hash(result_entries),
        "files": entries,
    }
    atomic_write_json(output, payload)
    print(
        f"Manifest: {len(entries)} archivos; código/config {payload['code_and_config_combined_sha256']}"
    )


if __name__ == "__main__":
    main()
