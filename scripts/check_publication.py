"""Reject private documents/artifacts in the Git index or unpublished history.

Usage: python scripts/check_publication.py [--base origin/master]
The root README is the sole documentation exception. Previously published
history is not rewritten; this check covers the current index and new commits.
"""

import argparse
import subprocess
from pathlib import PurePosixPath

from export_public_results import E_RAW, FIGURES, STAGES, TABLES

PUBLIC_RESULTS = {
    "published_results/manifest.json",
    "published_results/status.json",
    "published_results/E/frozen_rates.json",
    "published_results/N/n2_frozen.json",
    "published_results/N2A_N3/n3_frozen.json",
    "published_results/N2A_N3/audits.json",
}
for stage in STAGES:
    PUBLIC_RESULTS.update(
        f"published_results/{stage}/tables/{s}.csv" for s in TABLES[stage]
    )
    PUBLIC_RESULTS.update(
        f"published_results/{stage}/figures/{s}.png" for s in FIGURES[stage]
    )
for stage, names in E_RAW.items():
    PUBLIC_RESULTS.update(f"published_results/E/{stage}/{s}.csv" for s in names)

PRIVATE_PARTS = {
    "docs",
    "documents",
    "reports",
    "results",
    "checkpoints",
    "archive",
    ".private",
    "__pycache__",
    ".venv",
}
PRIVATE_NAMES = {
    "sources.json",
    "hardware_selected.json",
    "n2_frozen.json",
    "n3_frozen.json",
    "frozen_rates.json",
}


def allowed(name):
    path = PurePosixPath(name)
    if name in {"README.md", ".gitignore", ".gitattributes", "pyproject.toml", "uv.lock"}:
        return True
    if name.startswith("published_results/"):
        return name in PUBLIC_RESULTS
    if PRIVATE_PARTS.intersection(p.lower() for p in path.parts):
        return False
    if path.name in PRIVATE_NAMES:
        return False
    # The historical prototype plots and scientific JSON configs are results/code.
    return path.suffix.lower() in {
        ".py",
        ".json",
        ".png",
        ".toml",
        ".yml",
        ".yaml",
        ".sh",
    }


def git(*args):
    return subprocess.check_output(["git", *args])


def check(base):
    index = git("ls-files", "-s", "-z").decode().split("\0")
    failures = []
    count = 0
    for record in filter(None, index):
        meta, name = record.split("\t", 1)
        count += 1
        if not allowed(name) or meta.split()[0] not in {"100644", "100755"}:
            failures.append(f"index: {name}")
    # Tree paths for every outgoing commit also catch a file added then deleted.
    commits = git("rev-list", f"{base}..HEAD").decode().splitlines()
    for commit in commits:
        for name in git("ls-tree", "-rz", "--name-only", commit).decode().split("\0"):
            if name and not allowed(name):
                failures.append(f"commit {commit[:12]}: {name}")
    if failures:
        raise SystemExit(
            "Private/unreviewed files would be published:\n" + "\n".join(failures)
        )
    print(
        f"Publication boundary verified: {count} indexed files, {len(commits)} outgoing commits"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/master")
    check(parser.parse_args().base)
