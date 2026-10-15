import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.io import REPO, ROOT, digest, json_write, now

old = REPO / "dev/4_Noprop_fst_Advance"
manifest = json.loads((old / "results/artifact_manifest.json").read_text())
checks = []
for entry in manifest["files"]:
    if (
        entry["path"].startswith(("results/", "checkpoints/"))
        and entry["path"] != "results/artifact_manifest.json"
    ):
        p = old / entry["path"]
        checks.append(
            {
                "path": str(p.relative_to(REPO)),
                "exists": p.exists(),
                "match": p.exists() and digest(p) == entry["sha256"],
            }
        )
summary = {}
for stage in ["e0", "pilot", "e1", "e2_smoke"]:
    path = old / f"results/raw/{stage}/summary.json"
    summary[stage] = json.loads(path.read_text())
counts = {}
for p in (old / "results/raw").rglob("*.csv"):
    with p.open() as f:
        counts[str(p.relative_to(old))] = sum(1 for _ in csv.DictReader(f))
json_write(
    ROOT / "results/historical_audit.json",
    {
        "utc": now(),
        "checks": checks,
        "matched": sum(x["match"] for x in checks),
        "checked": len(checks),
        "row_counts": counts,
        "summaries": summary,
        "E2_confirmatory_exists": (old / "results/raw/e2").exists(),
        "E3_exists": (old / "results/raw/e3").exists(),
    },
)
print(
    {
        "checked": len(checks),
        "matched": sum(x["match"] for x in checks),
        "row_counts": counts,
    }
)
