import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extension.agents import InteractiveAgent, MSELearner
from extension.environment import PrivateEnvironment
from extension.evaluation import decisions, diagnostic
from extension.io import config
from extension.sequential import frozen_bellman_targets, kernel, policy_reference
from neural.core import Learner, generator
from neural.quadrature import NumericBlock

torch.set_num_threads(1)


def identical(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            identical(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            identical(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_direct_mse_is_exact_first_structured_block(dtype):
    s = Learner("structured_mse", 90000, 0.1, dtype=dtype)
    d = MSELearner("direct_mse", 90000, 0.1, dtype=dtype)
    d.blocks[0].load_state_dict(s.blocks[0].state_dict())
    g = generator(90000, "data")
    c = torch.randn((131, 2), generator=g, dtype=dtype)
    a = torch.arange(131) % 2
    r = torch.randint(2, (131,), generator=g).to(dtype)
    for _ in range(120):
        s.update(c, a, r, 128)
        d.update(c, a, r, 128)
        identical(s.blocks[0].state_dict(), d.blocks[0].state_dict())
        identical(s.optimizers[0].state_dict(), d.optimizers[0].state_dict())
        identical(s.batch_rng.get_state(), d.batch_rng.get_state())
    before = copy.deepcopy(s.blocks[1].state_dict())
    handle = s.blocks[1].register_forward_pre_hook(
        lambda *_: (_ for _ in ()).throw(AssertionError("foreign block executed"))
    )
    noise = torch.zeros(len(r), dtype=dtype)
    ls = s.local_update(0, c, a, r, noise)
    ld = d.local_update(0, c, a, r, noise)
    handle.remove()
    identical(ls, ld)
    identical(s.blocks[1].state_dict(), before)
    identical(s.blocks[0].state_dict(), d.blocks[0].state_dict())
    assert all(p.grad is None for p in d.blocks[0].biases) is False


def test_ensemble_mse_is_two_independent_direct_models():
    l = MSELearner("ensemble_mse", 90001)
    assert not l.composed and l.loss_name == "mse"
    assert len(l.blocks) == 2 and all(b.kind == "direct" for b in l.blocks)
    assert l.optimizers[0].state is not l.optimizers[1].state
    assert not {p.data_ptr() for p in l.blocks[0].parameters()} & {
        p.data_ptr() for p in l.blocks[1].parameters()
    }
    assert not torch.equal(l.blocks[0].weights[0], l.blocks[1].weights[0])


def advance(agent, env, count):
    for _ in range(count):
        c = env.context()
        a = agent.choose(c)
        r = env.step(a)
        agent.observe(c, a, r)


@pytest.mark.parametrize("method", config("base")["methods"])
def test_interactive_exact_resume_and_no_global_rng(method, tmp_path):
    global_state = torch.get_rng_state().clone()
    agent = InteractiveAgent(method, 90002, 0.1, capacity=180)
    env = PrivateEnvironment(90002, "interaction", 2)
    advance(agent, env, 65)
    saved = agent.snapshot()
    es = env.snapshot()
    advance(agent, env, 75)
    resumed = InteractiveAgent(method, 90002, 0.1, capacity=180)
    er = PrivateEnvironment(90002, "interaction", 2)
    resumed.restore(saved)
    er.restore(es)
    # Independent diagnostic cannot consume any training/policy/environment RNG.
    diagnostic(
        resumed.learner,
        {"seed": 90002, "family": "interaction", "beta": 2, "evaluation_contexts": 4},
        tmp_path,
        65,
    )
    identical(resumed.snapshot(), saved)
    advance(resumed, er, 75)
    identical(agent.snapshot(), resumed.snapshot())
    identical(env.snapshot(), er.snapshot())
    identical(global_state, torch.get_rng_state())
    assert agent.learner.step == 139


def test_initial_buffer_and_unseen_rewards_contract():
    a = InteractiveAgent("structured", 90003, 0.1, 10)
    e = PrivateEnvironment(90003, "radial", 2)
    for t in (1, 2):
        c = e.context()
        choice = a.choose(c)
        assert choice == t - 1
        reward = e.step(choice)
        assert isinstance(reward, float)
        a.observe(c, choice, reward)
        assert a.learner.step == t - 1
        with pytest.raises(RuntimeError):
            e.step(choice)
    assert a.size == 2


def test_environment_common_random_numbers_different_feedback():
    e0 = PrivateEnvironment(90004, "linear", 2)
    e1 = PrivateEnvironment(90004, "linear", 2)
    different = 0
    for _ in range(100):
        np.testing.assert_array_equal(e0.context(), e1.context())
        r0 = e0.step(0)
        r1 = e1.step(1)
        different += r0 != r1
        identical(e0.snapshot(), e1.snapshot())
    assert different > 0


def test_buffers_are_owned_and_can_diverge():
    a = InteractiveAgent("direct", 90005, 0.1, 10)
    b = InteractiveAgent("ensemble", 90005, 0.1, 10)
    assert a.c.data_ptr() != b.c.data_ptr()
    c = np.array([0.2, 0.3], np.float32)
    a.observe(c, 0, 1.0)
    b.observe(c, 1, 0.0)
    assert a.a[0] != b.a[0] and a.r[0] != b.r[0]


def test_average_decision_matches_same_weight_propensities():
    agent = InteractiveAgent("structured_average", 90006, 0.1)
    c = np.array([0.2, -0.4], np.float32)
    expected = []
    for action in (0, 1):
        ps = [
            1 / (1 + np.exp(-NumericBlock(b).base(c, action)))
            for b in agent.learner.blocks
        ]
        expected.append(np.mean(ps))
    np.testing.assert_allclose(agent.values(c).numpy(), expected, atol=1e-7, rtol=0)


def test_decision_regret_ties_and_wrong_gap():
    p = np.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])
    mu = np.array([[0.1, 0.9], [0.2, 0.7], [0.5, 0.5]])
    d = decisions(p, mu)
    np.testing.assert_array_equal(d["wrong_probability"], [0.5, 0, 0])
    np.testing.assert_allclose(d["regret"], [0.4, 0, 0], atol=1e-15)


def test_reserved_seed_disjointness_and_n3_grid():
    pilot = config("pilot_preregistered")
    spec = config("n3_preregistered")
    assert not set(pilot["seeds"]) & set(spec["seeds"])
    assert not (set(pilot["seeds"]) | set(spec["seeds"])) & set(range(81000, 85030))
    assert len(spec["seeds"]) == 30 and spec["interactions"] == 20000
    assert (
        len(pilot["learning_rates"])
        * len(pilot["seeds"])
        * len(pilot["families"])
        * len(pilot["beta"])
        * len(pilot["methods"])
        == 1920
    )


def test_small_mdp_reference_and_frozen_target_mass():
    p, _r = kernel()
    ref = policy_reference()
    assert ref["residual"] < 1e-10
    np.testing.assert_allclose(np.array(ref["values"])[[0, 2, 4]], 0, atol=1e-14)
    np.testing.assert_allclose(p[1:4].sum(2), 1, atol=1e-15)
    atoms = torch.linspace(-1, 1, 51, dtype=torch.float64)
    mass = torch.full((3, 2, 51), 1 / 51, dtype=torch.float64)
    policy = torch.full((3, 2), 0.5, dtype=torch.float64)
    target = frozen_bellman_targets(
        torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64),
        mass,
        policy,
        atoms,
    )
    torch.testing.assert_close(
        target.sum(1), torch.ones(3, dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        target @ atoms,
        torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    with pytest.raises(ValueError):
        frozen_bellman_targets(
            torch.zeros(3), torch.zeros(3), mass.requires_grad_(), policy, atoms
        )


def test_runner_recovers_failed_diagnostic_without_retraining(tmp_path, monkeypatch):
    from extension import io, runs

    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    job = {
        "stage": "N3",
        "method": "structured",
        "seed": 90009,
        "lr": 0.1,
        "family": "radial",
        "beta": 2,
        "regime": "interactions",
        "interactions": 12,
        "capacity": 12,
        "time_budget": None,
        "diagnostic_interactions": [5, 12],
        "evaluation_contexts": 4,
    }
    origin = {"source_sha256": "test"}
    actual = runs.diagnostic
    monkeypatch.setattr(
        runs,
        "diagnostic",
        lambda *_: (_ for _ in ()).throw(RuntimeError("injected diagnostic failure")),
    )
    assert runs.worker(job, origin)["status"] == "failed"
    _, folder, cp = io.paths(job)
    saved = torch.load(cp, weights_only=False)
    assert saved["agent"]["size"] == 5 and saved["agent"]["learner"]["step"] == 4
    monkeypatch.setattr(runs, "diagnostic", actual)
    assert runs.worker(job, origin)["status"] == "completed"
    resumed = torch.load(cp, weights_only=False)
    agent = InteractiveAgent("structured", 90009, 0.1, 12)
    env = PrivateEnvironment(90009, "radial", 2)
    advance(agent, env, 12)
    identical(agent.snapshot(), resumed["agent"])
    identical(env.snapshot(), resumed["environment"])
    assert (folder / "failure.json").exists()
    assert resumed["done_evals"] == [5, 12]


def test_pilot_result_has_no_oracle_metrics(tmp_path, monkeypatch):
    import json

    from extension import io, runs

    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    monkeypatch.setattr(
        runs,
        "trajectory_metrics",
        lambda *_: (_ for _ in ()).throw(AssertionError("pilot requested oracle")),
    )
    job = {
        "stage": "N3pilot",
        "method": "direct",
        "seed": 90010,
        "lr": 0.1,
        "family": "linear",
        "beta": 1.5,
        "regime": "interactions",
        "interactions": 9,
        "capacity": 9,
        "time_budget": None,
        "diagnostic_interactions": [],
        "evaluation_contexts": 4,
    }
    assert runs.worker(job, {"source_sha256": "test"})["status"] == "completed"
    r = json.loads((io.paths(job)[1] / "result.json").read_text())
    assert set(r["metrics"]) == {"observed_reward", "interactions"}
    assert r["updates"] == 8 and not r["history"]


def test_offline_control_reuses_actual_historical_data(tmp_path, monkeypatch):
    from extension import io, runs

    job = next(runs.n2a_jobs())
    job["updates"] = 12
    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    assert runs.worker(job, {"source_sha256": "test"})["status"] == "completed"
    saved = torch.load(io.paths(job)[2], weights_only=False)
    historic = torch.load(runs.OLD / job["source_checkpoint"], weights_only=False)
    identical(saved["data"], historic["data"])
    assert saved["learner"]["step"] == 12
    assert len(saved["learner"]["blocks"]) == 2


def test_complete_grid_and_seed_cluster_bootstrap(monkeypatch):
    from extension import runs
    from extension.analysis import contrast_rows

    assert len(list(runs.online_jobs("N3pilot"))) == 1920
    monkeypatch.setattr(runs, "digest", lambda _: "test_frozen")
    jobs = list(
        runs.online_jobs(
            "N3",
            {
                "selected_rates": {m: 0.1 for m in config("base")["methods"]},
                "time_budget_seconds": 25.0,
            },
        )
    )
    assert len(jobs) == 1440
    assert sum(j["regime"] == "interactions" for j in jobs) == 720
    rows = []
    for seed in (0, 1, 2):
        for family, offset in (("a", 0), ("b", 2)):
            rows.extend(
                [
                    {
                        "n": 128,
                        "seed": seed,
                        "family": family,
                        "method": "first",
                        "error": seed + offset,
                    },
                    {
                        "n": 128,
                        "seed": seed,
                        "family": family,
                        "method": "second",
                        "error": 0,
                    },
                ]
            )
    stats = contrast_rows(
        rows, [("first", "second")], ["error"], [128], [0, 1, 2], ["a", "b"], 90011
    )[0]
    assert stats["family"] == "all" and stats["seeds"] == 3
    assert stats["mean"] == 2 and stats["paired_sd"] == 1 and stats["dz"] == 2
