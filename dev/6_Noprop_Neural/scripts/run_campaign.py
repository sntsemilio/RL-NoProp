import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.campaign import freeze, n0_gate, stage_run
from neural.io import REPO, ROOT, event, json_write, now, provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["all", "N1", "pilot", "N2"], default="all")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    lock = open(ROOT / "results/campaign.lock", "w")  # noqa: SIM115 -- hold exclusive lock for supervisor lifetime
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    n0_gate()
    origin = provenance()
    path = ROOT / "results/campaign_origin.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["source_sha256"] != origin["source_sha256"]:
            raise RuntimeError(
                "Source changed after campaign launch. Explicit amendment and a distinct campaign required."
            )
        origin = previous
    else:
        json_write(path, origin)
    snapshot = ROOT / "results/source_snapshot"
    for relative in origin["source_files"]:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(REPO / relative, target)
    (snapshot / "tracked.diff").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=REPO)
    )
    json_write(
        ROOT / "results/process.json",
        {
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "utc": now(),
            "command": sys.argv,
            "provenance": origin,
        },
    )

    def stop(signum, frame):
        (ROOT / "results/STOP").touch()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for stage in ["N1", "pilot", "N2"] if args.stage == "all" else [args.stage]:
            rates = None
            if stage == "N2":
                frozen = freeze(origin)
                rates = frozen["selected_rates"]
                if provenance()["source_sha256"] != origin["source_sha256"]:
                    raise RuntimeError("Code changed before N2")
            if (
                stage == "pilot"
                and json.loads((ROOT / "results/N1_summary.json").read_text())["status"]
                != "completed"
            ):
                raise RuntimeError("N1 incomplete")
            stage_run(stage, args.workers, origin, rates)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/analyze.py")], check=True
            )
        event(ROOT / "results/ledger.jsonl", event="campaign_completed")
    except Exception as exc:
        event(ROOT / "results/ledger.jsonl", event="campaign_stopped", error=str(exc))
        raise
    finally:
        subprocess.run([sys.executable, str(ROOT / "scripts/manifest.py")], check=False)


if __name__ == "__main__":
    main()
