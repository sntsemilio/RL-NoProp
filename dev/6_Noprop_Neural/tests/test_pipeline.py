import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.campaign import job_id, jobs
from neural.core import METHODS
from neural.environment import triples
from neural.evaluation import bce, value_metrics
from neural.future import project_target


def test_complete_campaign_grid_and_partition_isolation():
    expected = {"N1": 1980, "pilot": 1680, "N2": 3360}
    seeds = []
    for stage, count in expected.items():
        grid = list(jobs(stage, {m: 0.01 for m in METHODS}))
        assert len(grid) == count
        assert len(set(map(job_id, grid))) == count
        seeds.append({j["seed"] for j in grid})
    assert not (seeds[0] & seeds[1] or seeds[0] & seeds[2] or seeds[1] & seeds[2])
    train = triples(83000, "linear", 1.5, 2048, "train")
    val = triples(83000, "linear", 1.5, 4096, "validation")
    assert not np.array_equal(train[0][:10], val[0][:10])
    assert train[0].shape == (2048, 2)
    assert len(train[1]) == 2048


def test_metric_is_context_mean_of_action_max():
    p = np.array([0.4, 0.2, 0.8, 0.1])
    mu = np.array([0.5, 0.5, 0.5, 0.5])
    assert abs(value_metrics(p, mu, np.zeros(4))["primary"] - 0.35) < 1e-14
    assert np.all(np.isfinite(bce(np.array([0.0, 1.0]), np.array([1.0, 0.0]))))


def test_projection_interface_mass_mean_terminal():
    atoms = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    mass = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]], dtype=torch.float64)
    r = torch.tensor([0.0, 0.3], dtype=torch.float64)
    d = torch.tensor([0.0, 1.0], dtype=torch.float64)
    p = project_target(r, d, mass, atoms, 0.95)
    torch.testing.assert_close(
        p.sum(1), torch.ones(2, dtype=torch.float64), atol=1e-14, rtol=0
    )
    torch.testing.assert_close(
        p @ atoms, torch.tensor([0.285, 0.3], dtype=torch.float64), atol=1e-14, rtol=0
    )
    assert torch.all(p >= 0)


def test_worker_resume_after_evaluation_failure(tmp_path, monkeypatch):
    from neural import campaign
    from neural.io import provenance

    origin = provenance()
    base = campaign.config("base")
    base.update(updates=12, save_every=6, batch=16)
    job = {
        "stage": "pilot",
        "family": "interaction",
        "n": 64,
        "seed": 81090,
        "method": "structured",
        "beta": 1.5,
        "dim": 2,
        "lr": 0.01,
        "base": base,
    }
    monkeypatch.setattr(campaign, "ROOT", tmp_path / "interrupted")
    original = campaign.readouts

    def fail(*a, **kw):
        raise ArithmeticError("Injected evaluation failure for resume test")

    monkeypatch.setattr(campaign, "readouts", fail)
    result = campaign.run_job(job, origin)
    assert result["status"] == "failed"
    monkeypatch.setattr(campaign, "readouts", original)
    resumed = campaign.run_job(job, origin)
    assert resumed["status"] == "completed"
    jid = job_id(job)
    first = torch.load(
        tmp_path / "interrupted/checkpoints/pilot" / f"{jid}.pt", weights_only=False
    )
    first_result = json.loads(
        (tmp_path / "interrupted/results/raw/pilot" / jid / "result.json").read_text()
    )
    assert (
        tmp_path / "interrupted/results/raw/pilot" / jid / "failures.jsonl"
    ).exists()
    monkeypatch.setattr(campaign, "ROOT", tmp_path / "continuous")
    assert campaign.run_job(job, origin)["status"] == "completed"
    second = torch.load(
        tmp_path / "continuous/checkpoints/pilot" / f"{jid}.pt", weights_only=False
    )
    second_result = json.loads(
        (tmp_path / "continuous/results/raw/pilot" / jid / "result.json").read_text()
    )
    assert first_result["validation_value_bce"] == second_result["validation_value_bce"]
    assert first_result["data_sha256"] == second_result["data_sha256"]
    for x, y in zip(first["learner"]["blocks"], second["learner"]["blocks"]):
        assert all(torch.equal(x[k], y[k]) for k in x)
    assert torch.equal(first["learner"]["batch_rng"], second["learner"]["batch_rng"])
    assert all(
        torch.equal(x, y)
        for x, y in zip(first["learner"]["noise_rng"], second["learner"]["noise_rng"])
    )
