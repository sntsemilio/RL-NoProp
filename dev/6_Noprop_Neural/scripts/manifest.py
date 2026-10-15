import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.io import ROOT, digest, json_write, now

files = []
for d in ["results", "checkpoints", "docs", "configs", "neural", "scripts", "tests"]:
    for p in sorted((ROOT / d).rglob("*")):
        if (
            not p.is_file()
            or p.name in ("artifact_manifest.json", "campaign.lock")
            or "__pycache__" in str(p)
            or p.suffix == ".tmp"
        ):
            continue
        try:
            files.append(
                {
                    "path": str(p.relative_to(ROOT)),
                    "bytes": p.stat().st_size,
                    "sha256": digest(p),
                }
            )
        except FileNotFoundError:
            pass
json_write(
    ROOT / "results/artifact_manifest.json",
    {
        "utc": now(),
        "snapshot_note": "Mutable active-run files can change after this snapshot; final manifest regenerated at campaign end.",
        "files": files,
    },
)
print(f"Manifest: {len(files)} files")
