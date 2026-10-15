import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def show():
    result = {}
    for name in ("launcher", "status"):
        p = ROOT / f"results/{name}.json"
        if p.exists():
            result[name] = json.loads(p.read_text())
    pid = result.get("launcher", {}).get("pid")
    cmd = Path(f"/proc/{pid}/cmdline")
    result["process_visible_active"] = bool(
        pid
        and cmd.exists()
        and b"7_Noprop_N2A_N3/scripts/run_campaign.py" in cmd.read_bytes()
    )
    result["counts"] = {
        stage: {
            "completed": len(
                list((ROOT / "results/raw" / stage).glob("*/result.json"))
            ),
            "failures": len(
                list((ROOT / "results/raw" / stage).glob("*/failure.json"))
            ),
            "expected": n,
        }
        for stage, n in (("N2A", 480), ("N3pilot", 1920), ("N3", 1440))
    }
    active = result.get("status", {}).get("active", [])
    result["progress"] = []
    for jid in active:
        p = ROOT / "results/raw" / result["status"]["stage"] / jid / "progress.json"
        if p.exists():
            result["progress"].append(json.loads(p.read_text()))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        show()
        if not args.watch:
            break
        time.sleep(10)
