import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.io import ROOT

for name in ["process", "status"]:
    p = ROOT / f"results/{name}.json"
    if p.exists():
        value = json.loads(p.read_text())
        print(
            name,
            json.dumps(
                value
                if name == "status"
                else {k: v for k, v in value.items() if k != "provenance"},
                indent=2,
            ),
        )
        if name == "process":
            try:
                os.kill(value["pid"], 0)
                print("Supervisor PID exists")
            except ProcessLookupError:
                print("Supervisor is no longer active")
for stage, expected in [("N1", 1980), ("pilot", 1680), ("N2", 3360)]:
    print(
        stage,
        "completed",
        len(list((ROOT / "results/raw" / stage).glob("*/result.json"))),
        "/",
        expected,
        "failed",
        len(list((ROOT / "results/raw" / stage).glob("*/failure.json"))),
    )
print("STOP flag:", (ROOT / "results/STOP").exists())
