import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extension import OLD, ROOT
from extension.io import json_write, now, provenance, verify_history
from extension.sequential import plan

if __name__ == "__main__":
    stamp = now().replace(":", "-")
    folder = ROOT / "results/validation" / stamp
    folder.mkdir(parents=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(OLD / "tests"),
        str(ROOT / "tests"),
        "-q",
        f"--junitxml={folder / 'junit.xml'}",
    ]
    with (folder / "pytest.log").open("w") as log:
        result = subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    summary = {
        "utc": now(),
        "returncode": result.returncode,
        "command": command,
        "provenance": provenance(),
        "log": str((folder / "pytest.log").relative_to(ROOT)),
    }
    if result.returncode == 0:
        summary["history"] = verify_history(full=True)
        json_write(ROOT / "results/sequential_plan.json", plan())
    json_write(folder / "summary.json", summary)
    json_write(ROOT / "results/validation/latest.json", summary)
    print((folder / "pytest.log").read_text(), flush=True)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2),
        flush=True,
    )
    sys.exit(result.returncode)
