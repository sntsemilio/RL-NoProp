import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from neural.io import digest, event, json_write, now, save_checkpoint

__all__ = ["digest", "event", "json_write", "now", "save_checkpoint"]

from . import OLD, ROOT

REPO = ROOT.parents[1]


def config(name):
    return json.loads((ROOT / "configs" / f"{name}.json").read_text())


def job_id(job):
    h = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:12]
    return f"{job['stage']}_{job['family']}_s{job['seed']}_{job['method']}_{h}"


def paths(job):
    jid = job_id(job)
    folder = ROOT / "results/raw" / job["stage"] / jid
    cp = ROOT / "checkpoints" / job["stage"] / f"{jid}.pt"
    return jid, folder, cp


def provenance():
    files = {}
    for root, directories in (
        (ROOT, ("extension", "scripts", "tests", "configs")),
        (OLD, ("neural",)),
    ):
        for directory in directories:
            for p in sorted((root / directory).rglob("*")):
                if p.suffix in (".py", ".json") and "frozen" not in p.name:
                    files[str(p.relative_to(REPO))] = digest(p)
    for filename in ("uv.lock", "pyproject.toml"):
        files[filename] = digest(REPO / filename)
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


def snapshot_origin():
    origin = provenance()
    path = ROOT / "results/origin.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["source_sha256"] != origin["source_sha256"]:
            raise RuntimeError(
                "Source changed after launch; preserve campaign and register amendment"
            )
        return previous
    json_write(path, origin)
    for name in origin["source_files"]:
        target = ROOT / "results/source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    (ROOT / "results/source_snapshot/tracked.diff").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=REPO)
    )
    return origin


def verify_history(full=False):
    sources = config("sources")["files"]
    for p, expected in sources.items():
        if digest(REPO / p) != expected:
            raise RuntimeError(f"Historical input changed: {p}")
    checked = len(sources)
    if full:
        manifest = json.loads((OLD / "results/artifact_manifest.json").read_text())
        for entry in manifest["files"]:
            p, expected = entry["path"], entry["sha256"]
            if digest(OLD / p) != expected:
                raise RuntimeError(f"Historical artifact changed: {p}")
            checked += 1
    return {"utc": now(), "checked": checked, "status": "passed"}


def manifest():
    files = {}
    for directory in ("results", "checkpoints", "configs", "docs"):
        for p in sorted((ROOT / directory).rglob("*")):
            if (
                p.is_file()
                and p.name
                not in (
                    "artifact_manifest.json",
                    "campaign.log",
                    "process.json",
                    "status.json",
                )
                and p.suffix != ".tmp"
            ):
                files[str(p.relative_to(ROOT))] = digest(p)
    json_write(
        ROOT / "results/artifact_manifest.json",
        {
            "utc": now(),
            "files": files,
            "excluded_mutable": [
                "campaign.log",
                "process.json",
                "status.json",
                "artifact_manifest.json",
            ],
        },
    )
    return len(files)


def completed(stage):
    return [
        json.loads(p.read_text())
        for p in sorted((ROOT / "results/raw" / stage).glob("*/result.json"))
    ]


def npz_write(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".tmp"), "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(path.with_suffix(".tmp"), path)
