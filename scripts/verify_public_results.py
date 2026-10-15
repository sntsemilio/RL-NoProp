"""Verify exported hashes and reproduce the N2/N3 primary seed bootstraps.

Works in a public checkout without documents, checkpoints or private metadata.
No files are written and no training is performed.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "published_results"


def read(name):
    return json.loads((ROOT / name).read_text())


def rows(name):
    with (ROOT / name).open(newline="") as f:
        return list(csv.DictReader(f))


def verify_manifest():
    manifest = read("manifest.json")["files"]
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()}
    if actual != set(manifest) | {"manifest.json"}:
        raise ValueError("Missing or unexpected publication files")
    for name, expected in manifest.items():
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Unsafe manifest path: {name}")
        data = path.read_bytes()
        if (
            len(data) != expected["bytes"]
            or hashlib.sha256(data).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"Hash/size mismatch: {name}")
    print(f"SHA-256 verified: {len(manifest)} artifacts")


def index_unique(records, fields):
    result = {}
    for r in records:
        key = tuple(r[k] for k in fields)
        if key in result:
            raise ValueError(f"Duplicate experimental cell: {key}")
        result[key] = r
    return result


def compare(values, recorded, bootstrap_seed, resamples, adjusted=False):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite seed outcome")
    idx = np.random.default_rng(bootstrap_seed).integers(
        len(values), size=(resamples, len(values))
    )
    means = values[idx].mean(1)
    expected = {
        "mean": values.mean(),
        "ci95_low": np.quantile(means, 0.025),
        "ci95_high": np.quantile(means, 0.975),
    }
    if adjusted:
        expected.update(
            ci9875_low=np.quantile(means, 0.00625),
            ci9875_high=np.quantile(means, 0.99375),
        )
    for key, value in expected.items():
        # CSV decimal round trips and the order of four-family means can differ
        # at FP64 roundoff; this is not a change to scientific tolerances.
        np.testing.assert_allclose(value, float(recorded[key]), rtol=0, atol=1e-12)
    count_key = "n_seeds" if adjusted else "seeds"
    if int(recorded[count_key]) != len(values):
        raise ValueError("Seed count differs from frozen protocol")
    return expected


def verify_n2():
    frozen = read("N/n2_frozen.json")
    spec, analysis = frozen["N2"], frozen["base"]["analysis"]
    records = [
        r
        for r in rows("N/tables/n2_per_replica.csv")
        if r["readout"] == "value" and r["method"] in {"structured", "direct"}
    ]
    data = index_unique(records, ("n", "seed", "family", "method"))
    expected_cells = {
        (str(n), str(s), f, m)
        for n in spec["train_triplets"]
        for s in spec["seeds"]
        for f in spec["families"]
        for m in ("structured", "direct")
    }
    if set(data) != expected_cells:
        raise ValueError("Incomplete or unexpected N2 primary grid")
    table = index_unique(rows("N/tables/n2_paired_contrasts.csv"), ("n", "comparison"))
    for n in spec["train_triplets"]:
        values = []
        for seed in spec["seeds"]:
            means = [
                np.mean(
                    [
                        float(data[str(n), str(seed), f, m]["primary"])
                        for f in spec["families"]
                    ]
                )
                for m in ("structured", "direct")
            ]
            values.append(means[0] - means[1])
        result = compare(
            values,
            table[str(n), "structured/value minus direct/value"],
            analysis["bootstrap_seed"],
            analysis["bootstrap"],
            adjusted=True,
        )
        print(
            f"N2 n={n}: S - direct = {result['mean']:.9f}; 98.75% CI "
            f"[{result['ci9875_low']:.9f}, {result['ci9875_high']:.9f}]"
        )


def verify_n3():
    spec = read("N2A_N3/n3_frozen.json")["N3"]
    records = rows("N2A_N3/tables/n3_runs.csv")
    data = index_unique(records, ("regime", "seed", "family", "method"))
    expected = {
        (r, str(s), f, m)
        for r in spec["regimes"]
        for s in spec["seeds"]
        for f in spec["families"]
        for m in spec["methods"]
    }
    if set(data) != expected:
        raise ValueError("Incomplete or unexpected N3 grid")
    for row in records:
        if (
            row["regime"] == "interactions"
            and int(row["interactions"]) != spec["interactions"]
        ):
            raise ValueError("Wrong confirmatory interaction horizon")
        if row["time_censored"].lower() != "false":
            raise ValueError("Censored run requires separate interpretation")
    table = index_unique(
        rows("N2A_N3/tables/n3_contrasts.csv"),
        ("regime", "first", "second", "metric", "family"),
    )
    # Verify both regimes, but only the fixed-interaction comparison is primary.
    for regime in spec["regimes"]:
        values = [
            np.mean(
                [
                    float(data[regime, str(s), f, "structured"]["mean_regret"])
                    - float(data[regime, str(s), f, "direct"]["mean_regret"])
                    for f in spec["families"]
                ]
            )
            for s in spec["seeds"]
        ]
        recorded = table[regime, "structured", "direct", "mean_regret", "all"]
        result = compare(
            values, recorded, spec["bootstrap_seed"], spec["bootstrap_resamples"]
        )
        print(
            f"N3 {regime}: S - direct = {result['mean']:.9f}; 95% CI "
            f"[{result['ci95_low']:.9f}, {result['ci95_high']:.9f}]"
        )


if __name__ == "__main__":
    verify_manifest()
    verify_n2()
    verify_n3()
