"""Independent post-run checks of artifacts, environment and reported outcomes."""

import hashlib
import json

import numpy as np
import torch
from neural.core import seed_for
from neural.environment import means
from neural.evaluation import value_metrics

from . import ROOT
from .io import completed, config, digest, json_write, now, paths, provenance
from .runs import n2a_jobs, online_jobs


def audit(stage):
    results = completed(stage)
    if stage == "N2A":
        expected = list(n2a_jobs())
    else:
        expected = list(
            online_jobs(stage, config("n3_frozen") if stage == "N3" else None)
        )
    want = {paths(j)[0]: j for j in expected}
    found = {r["id"]: r for r in results}
    if stage != "N3pilot" and found.keys() != want.keys():
        raise RuntimeError(f"{stage}: missing or unexpected runs")
    origin = json.loads((ROOT / "results/origin.json").read_text())
    if origin["source_sha256"] != provenance()["source_sha256"]:
        raise RuntimeError("Source changed during campaign")
    largest = 0.0
    for jid, r in found.items():
        job = r["config"]
        if job != want[jid] or r["provenance"] != origin:
            raise RuntimeError(f"Config/provenance mismatch {jid}")
        cp = ROOT / r["checkpoint"]
        if digest(cp) != r["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed {jid}")
        saved = torch.load(cp, weights_only=False, map_location="cpu")
        folder = ROOT / "results/raw" / stage / jid
        if stage == "N2A":
            assert saved["learner"]["step"] == 5000
            datahash = hashlib.sha256(
                b"".join(x.tobytes() for x in saved["data"])
            ).hexdigest()
            assert datahash == job["source_data_sha256"] == r["data_sha256"]
            with np.load(folder / "predictions.npz") as z:
                actual = value_metrics(z["value"], z["mu"], z["rewards"])
            error = abs(actual["primary"] - r["metrics"]["value"]["primary"])
        else:
            with np.load(folder / "trajectory.npz") as z:
                c, a, rewards = (z[k] for k in ("contexts", "actions", "rewards"))
                t = len(a)
                assert r["updates"] == t - 1 == saved["agent"]["learner"]["step"]
                assert a[0] == 0 and a[1] == 1 and saved["agent"]["size"] == t
                crng = np.random.default_rng(
                    seed_for(job["seed"], job["family"], job["beta"], "online_context")
                )
                np.testing.assert_array_equal(
                    c, crng.uniform(-1, 1, (t, 2)).astype(np.float32)
                )
                urng = np.random.default_rng(
                    seed_for(
                        job["seed"], job["family"], job["beta"], "potential_rewards"
                    )
                )
                uniforms = urng.random((t, 2))
                chosen = means(c.astype(float), a, job["family"], job["beta"])
                np.testing.assert_array_equal(
                    rewards, (uniforms[np.arange(t), a] < chosen).astype(np.float32)
                )
                for key, buffer in zip(
                    ("contexts", "actions", "rewards"), saved["agent"]["buffer"]
                ):
                    np.testing.assert_array_equal(z[key], buffer.numpy())
                if stage == "N3":
                    all_mu = np.stack(
                        [
                            means(
                                c.astype(float),
                                np.full(t, j),
                                job["family"],
                                job["beta"],
                            )
                            for j in (0, 1)
                        ],
                        1,
                    )
                    regrets = all_mu.max(1) - chosen
                    np.testing.assert_allclose(regrets, z["regret"], rtol=0, atol=1e-12)
                    error = abs(float(regrets.mean()) - r["metrics"]["mean_regret"])
                    assert job["frozen_sha256"] == digest(
                        ROOT / "configs/n3_frozen.json"
                    )
                    if job["regime"] == "interactions":
                        assert t == 20000 and saved["done_evals"] == [
                            100,
                            1000,
                            5000,
                            20000,
                        ]
                    else:
                        assert (
                            r["deployed_seconds"] >= job["time_budget"]
                            or r["time_censored"]
                        )
                else:
                    assert (
                        set(r["metrics"]) == {"observed_reward", "interactions"}
                        and t == 5000
                    )
                    error = abs(float(rewards.mean()) - r["metrics"]["observed_reward"])
        largest = max(largest, error)
        if error > 1e-12:
            raise RuntimeError(f"Reported metric differs {jid}: {error}")
    result = {
        "utc": now(),
        "stage": stage,
        "status": "passed",
        "verified": len(found),
        "expected": len(want),
        "max_recomputed_metric_difference": largest,
        "source_sha256": origin["source_sha256"],
        "checks": "grid, config, source, checkpoint hash, real feedback reconstruction, RNG pairing, steps and metric reconstruction",
    }
    json_write(ROOT / f"results/audit_{stage}.json", result)
    return result
