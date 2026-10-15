"""Export reviewed numerical artifacts without modifying private runs.

Run in the original research workspace. Tables are byte-for-byte copies;
frozen configurations are scientific-field extracts, not replacements for the
private originals. No checkpoint, log or document is exported.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "published_results"
STAGES = {
    "E": ROOT / "dev/4_Noprop_fst_Advance",
    "N": ROOT / "dev/6_Noprop_Neural",
    "N2A_N3": ROOT / "dev/7_Noprop_N2A_N3",
}
TABLES = {
    "E": [
        "e1_primary_by_instance",
        "e1_across_instances",
        "e1_paired_differences",
        "e1_paired_across_instances",
        "e1_per_arm_aggregate",
    ],
    "N": [
        "costs",
        "n1_curves",
        "n1_summary",
        "n2_aggregates",
        "n2_paired_contrasts",
        "n2_per_family",
        "n2_per_replica",
        "pilot_selection",
    ],
    "N2A_N3": [
        "n2a_contrasts",
        "n2a_families",
        "n2a_readouts",
        "n2a_summary",
        "n3_contrasts",
        "n3_families",
        "n3_readouts",
        "n3_runs",
        "n3_same_checkpoint_contrasts",
        "n3_summary",
        "pilot_scores",
    ],
}
FIGURES = {
    "E": ["e0_sampler_variance", "e1_local_bound", "e1_value_error"],
    "N": ["n1_learning", "n2_primary"],
    "N2A_N3": ["n2a_value_decisions", "n3_diagnostics", "n3_cost_regret"],
}
E_RAW = {
    "e0": [
        "locality_checks",
        "bounds_checks",
        "gradient_checks",
        "transition_checks",
        "monte_carlo_checks",
        "analytic_checks",
    ],
    "e1": ["loss_history", "per_arm"],
    "e2_smoke": ["per_run", "per_round"],
    "pilot": ["aggregate", "per_seed"],
}
FROZEN_FIELDS = {
    "E": (
        "frozen_rates",
        [
            "batch_size",
            "candidate_rates",
            "criterion",
            "frozen_at_utc",
            "pilot_instances",
            "pilot_seeds",
            "pilot_training_observations_per_arm",
            "rates",
            "rho",
            "status",
            "steps_per_fit",
            "true_means_used_for_selection",
            "validation_rewards_per_arm",
        ],
    ),
    "N": (
        "n2_frozen",
        [
            "N2",
            "base",
            "pilot",
            "selected_rates",
            "selection_bce",
            "selection_rule",
            "status",
            "utc",
        ],
    ),
    "N2A_N3": (
        "n3_frozen",
        [
            "N3",
            "base",
            "pilot",
            "scores",
            "selected_rates",
            "status",
            "time_budget_seconds",
            "utc",
        ],
    ),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def encoded(data):
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def export():
    # Construct everything before writing, so missing private inputs fail early.
    artifacts = {}
    for stage, root in STAGES.items():
        for name in TABLES[stage]:
            artifacts[f"{stage}/tables/{name}.csv"] = (
                root / "results/aggregated" / f"{name}.csv"
            ).read_bytes()
        for name in FIGURES[stage]:
            artifacts[f"{stage}/figures/{name}.png"] = (
                root / "results/figures" / f"{name}.png"
            ).read_bytes()
        name, fields = FROZEN_FIELDS[stage]
        source = root / "configs" / f"{name}.json"
        original = read(source)
        public = {k: original[k] for k in fields}
        public["publication"] = {
            "kind": "scientific_fields_from_private_frozen_original",
            "original_sha256": sha(source.read_bytes()),
            "omitted": "local provenance, machine inventory, raw artifact paths",
            "embedded_statuses": "retained as recorded before execution; see status.json for completion",
        }
        artifacts[f"{stage}/{name}.json"] = encoded(public)

    for stage, names in E_RAW.items():
        for name in names:
            artifacts[f"E/{stage}/{name}.csv"] = (
                STAGES["E"] / "results/raw" / stage / f"{name}.csv"
            ).read_bytes()

    audits = {}
    for stage in ("N2A", "N3pilot", "N3"):
        source = read(STAGES["N2A_N3"] / "results" / f"audit_{stage}.json")
        audits[stage] = {
            k: source[k]
            for k in (
                "stage",
                "status",
                "verified",
                "expected",
                "max_recomputed_metric_difference",
                "utc",
            )
        }
    artifacts["N2A_N3/audits.json"] = encoded(audits)

    counts = {}
    for stage, filename in (
        ("N1", "N1_summary"),
        ("pilot_N2", "pilot_summary"),
        ("N2", "N2_summary"),
    ):
        d = read(STAGES["N"] / "results" / f"{filename}.json")
        counts[stage] = {k: d[k] for k in ("status", "completed", "total", "failed")}
    for stage, audit in audits.items():
        counts[stage] = {
            "status": "completed",
            "completed": audit["verified"],
            "total": audit["expected"],
            "audit": audit["status"],
        }
    finished = read(STAGES["N2A_N3"] / "results/status.json")
    artifacts["status.json"] = encoded(
        {
            "as_of": "2026-10-14",
            "campaign_finished_utc": finished["utc"],
            "campaigns": counts,
            "E0": "completed",
            "E1": "completed",
            "E2": "smoke_only",
            "E2_confirmatory": "pending",
            "E3": "pending",
            "N0": "passed_28_tests",
            "extension_validation": "passed_48_tests",
            "N2A_inference": "exploratory; original N2 conclusions unchanged",
            "S0_S1": "prepared_not_executed",
            "N4": "pending",
            "S2_S3": "pending",
            "Atari_A0_A4": "pending",
            "scope": "numerical tables, figures, scientific configuration extracts; full runs remain local",
            "direct_MSE": "structured_mse.d1 is a same-training readout, not an independent fit",
        }
    )
    artifacts["manifest.json"] = encoded(
        {
            "schema": 1,
            "algorithm": "sha256",
            "files": {
                name: {"sha256": sha(data), "bytes": len(data)}
                for name, data in sorted(artifacts.items())
            },
        }
    )
    if OUT.exists():
        unexpected = {
            str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()
        } - artifacts.keys()
        if unexpected:
            raise RuntimeError(
                f"Unexpected files in publication directory: {sorted(unexpected)}"
            )
    for name, data in sorted(artifacts.items()):
        path = OUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(
        f"Exported {len(artifacts)} files, {sum(map(len, artifacts.values())):,} bytes"
    )


if __name__ == "__main__":
    export()
