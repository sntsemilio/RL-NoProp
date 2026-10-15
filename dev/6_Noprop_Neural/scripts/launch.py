"""Launch/resume a detached local supervisor, or request a checkpoint-safe stop."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.io import REPO, ROOT, json_write, now


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "resume", "stop"])
    args = p.parse_args()
    process = ROOT / "results/launcher.json"
    if args.action == "stop":
        (ROOT / "results/STOP").touch()
        print("Stop requested. Active fits finish/checkpoint; no new fits will start.")
        return
    if process.exists():
        old = json.loads(process.read_text())
        pid = old["pid"]
        cmd = Path(f"/proc/{pid}/cmdline")
        if cmd.exists() and "run_campaign.py" in cmd.read_text():
            raise RuntimeError(f"Campaign already active: PID {pid}")
    if (ROOT / "results/STOP").exists():
        if args.action != "resume":
            raise RuntimeError("STOP exists. Use resume deliberately.")
        (ROOT / "results/STOP").unlink()
    stamp = now().replace(":", "-")
    log = ROOT / "results" / f"campaign_{stamp}.log"
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/tmp/matplotlib-noprop-neural",
        "PYTHONUNBUFFERED": "1",
    }
    command = [
        sys.executable,
        str(ROOT / "scripts/run_campaign.py"),
        "--stage",
        "all",
        "--workers",
        "4",
    ]
    with log.open("a") as out:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            cwd=REPO,
            env=env,
            start_new_session=True,
        )
    record = {
        "pid": child.pid,
        "pgid": child.pid,
        "utc": now(),
        "log": str(log),
        "command": command,
    }
    json_write(process, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
