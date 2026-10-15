import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extension import ROOT
from extension.io import json_write, now

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--stage", default="all", choices=["all", "N2A", "N3pilot", "N3"]
    )
    args = parser.parse_args()
    folder = ROOT / "results"
    folder.mkdir(exist_ok=True)
    prior = folder / "launcher.json"
    if prior.exists():
        pid = json.loads(prior.read_text())["pid"]
        cmd = Path(f"/proc/{pid}/cmdline")
        if (
            cmd.exists()
            and b"7_Noprop_N2A_N3/scripts/run_campaign.py" in cmd.read_bytes()
        ):
            raise RuntimeError(f"Campaign already active PID {pid}")
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/run_campaign.py"),
        "--workers",
        str(args.workers),
        "--stage",
        args.stage,
    ]
    env = {
        **os.environ,
        **{
            k: "1"
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    with (folder / "campaign.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            cwd=ROOT.parents[1],
        )
    result = {
        "pid": process.pid,
        "utc": now(),
        "command": command,
        "log": str(folder / "campaign.log"),
    }
    json_write(prior, result)
    print(json.dumps(result, indent=2))
