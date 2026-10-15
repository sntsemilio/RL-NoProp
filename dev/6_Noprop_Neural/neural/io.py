import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def event(path, **fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"utc": now(), **fields}, allow_nan=False) + "\n")


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def provenance():
    files = {
        str(p.relative_to(REPO)): digest(p)
        for directory in ("neural", "scripts", "tests", "configs")
        for p in sorted((ROOT / directory).rglob("*"))
        if p.suffix in (".py", ".json") and "frozen" not in p.name
    }
    files["pyproject.toml"] = digest(REPO / "pyproject.toml")
    files["uv.lock"] = digest(REPO / "uv.lock")
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=REPO)
    return {
        "utc": now(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO)
        .decode()
        .strip(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_files": files,
        "source_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
