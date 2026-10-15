import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.io import ROOT, event, json_write, now, provenance

stamp = now().replace(":", "-")
path = ROOT / "results/n0" / stamp
path.mkdir(parents=True)
t = time.perf_counter()
env = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(ROOT / "tests"),
        "--junitxml=" + str(path / "junit.xml"),
    ],
    capture_output=True,
    text=True,
    env=env,
    check=False,
)
(path / "pytest.log").write_text(result.stdout + "\n" + result.stderr)
report = {
    "status": "passed" if result.returncode == 0 else "failed",
    "utc": now(),
    "seconds": time.perf_counter() - t,
    "provenance": provenance(),
    "tolerances": json.loads((ROOT / "configs/n0.json").read_text()),
    "log": str(path / "pytest.log"),
}
json_write(path / "summary.json", report)
json_write(ROOT / "results/n0/latest.json", report)
event(
    ROOT / "results/ledger.jsonl",
    stage="N0",
    **{k: report[k] for k in ("status", "seconds", "log")},
)
print(result.stdout, result.stderr, flush=True)
sys.exit(result.returncode)
