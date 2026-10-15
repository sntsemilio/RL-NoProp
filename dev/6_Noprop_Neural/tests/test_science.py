import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.core import Learner, generator, sample
from neural.quadrature import NumericBlock, integral, readouts

TOL = json.loads((Path(__file__).resolve().parents[1] / "configs/n0.json").read_text())
torch.set_num_threads(1)


def data(dtype=torch.float64):
    g = generator(81000, "test_data")
    return (
        torch.randn((7, 2), generator=g, dtype=dtype),
        torch.arange(7) % 2,
        (torch.arange(7) % 3 == 0).to(dtype),
        torch.randn(7, generator=g, dtype=dtype),
    )


@pytest.mark.parametrize("method", ["structured", "flexible", "structured_mse"])
def test_locality_instrumented_and_optimizer_disjoint(method):
    l = Learner(method, 81001, 0.01, dtype=torch.float64)
    c, a, r, z = data()
    assert l.optimizers[0].state is not l.optimizers[1].state
    assert not (
        {p.data_ptr() for p in l.blocks[0].parameters()}
        & {p.data_ptr() for p in l.blocks[1].parameters()}
    )
    assert {id(p) for p in l.optimizers[0].param_groups[0]["params"]} == {
        id(p) for p in l.blocks[0].parameters()
    }
    assert {id(p) for p in l.optimizers[1].param_groups[0]["params"]} == {
        id(p) for p in l.blocks[1].parameters()
    }
    for k in range(2):
        other = 1 - k
        before = copy.deepcopy(l.blocks[other].state_dict())

        def forbidden(*args):
            raise AssertionError("Another block was executed in local update")

        handle = l.blocks[other].register_forward_pre_hook(forbidden)
        l.local_update(k, c, a, r, z)
        handle.remove()
        assert all(
            torch.equal(v, before[n]) for n, v in l.blocks[other].state_dict().items()
        )
        # Clear pre-existing gradients, then inspect the actual graph.
        for b in l.blocks:
            b.zero_grad(set_to_none=True)
        u = l.blocks[k](c, a, z)
        loss = F.binary_cross_entropy_with_logits(u, r)
        cross = torch.autograd.grad(
            loss, list(l.blocks[other].parameters()), allow_unused=True
        )
        assert all(g is None for g in cross)
    # Perturb all foreign parameters; fixed local inputs must give identical update.
    twin = copy.deepcopy(l)
    with torch.no_grad():
        for p in twin.blocks[1].parameters():
            p.add_(10)
    l.local_update(0, c, a, r, z)
    twin.local_update(0, c, a, r, z)
    assert all(
        torch.equal(p, q)
        for p, q in zip(l.blocks[0].parameters(), twin.blocks[0].parameters())
    )


@pytest.mark.parametrize("method", ["structured", "flexible", "structured_mse"])
@pytest.mark.parametrize("k", [0, 1])
def test_finite_difference_every_parameter(method, k):
    l = Learner(method, 81002, 0.01, dtype=torch.float64)
    # Shift contexts until all hidden preactivations are away from ReLU kinks.
    c, a, r, z = data()
    b = l.blocks[k]
    for attempt in range(100):
        x = b.features(c, a, z)
        margin = float("inf")
        for w, bias in zip(b.weights[:-1], b.biases[:-1]):
            x = x @ w + bias
            margin = min(margin, x.abs().min().item())
            x = x.relu()
        if margin > TOL["relu_min_margin"]:
            break
        c = c + 0.001
    assert margin > TOL["relu_min_margin"]

    def loss():
        u = b(c, a, z)
        return (
            F.binary_cross_entropy_with_logits(u, r)
            if l.loss_name == "bce"
            else 0.5 * (u.sigmoid() - r).square().mean()
        )

    loss().backward()
    epsilon = TOL["finite_difference_epsilon"]
    worst = 0
    for p in b.parameters():
        if not p.requires_grad:
            continue
        grad = p.grad.flatten().clone()
        for i in range(p.numel()):
            with torch.no_grad():
                old = p.flatten()[i].item()
                p.flatten()[i] = old + epsilon
                plus = loss().item()
                p.flatten()[i] = old - epsilon
                minus = loss().item()
                p.flatten()[i] = old
            numerical = (plus - minus) / (2 * epsilon)
            worst = max(
                worst,
                abs(numerical - grad[i].item())
                / max(1, abs(numerical), abs(grad[i].item())),
            )
    assert worst < TOL["gradient_normalized_tolerance"], worst


def test_first_structured_equals_direct_updates():
    s = Learner("structured", 81000, 0.1)
    d = Learner("direct", 81001, 0.1)
    d.blocks[0].load_state_dict(s.blocks[0].state_dict())
    c, a, r, z = data(torch.float32)
    for _ in range(25):
        ls = s.local_update(0, c, a, r, z)
        ld = d.local_update(0, c, a, r, z + 42)
        assert torch.equal(ls, ld)
        assert all(
            torch.equal(p, q)
            for p, q in zip(s.blocks[0].parameters(), d.blocks[0].parameters())
        )


@pytest.mark.parametrize("method", ["structured", "flexible"])
def test_piecewise_represents_real_relu_network(method):
    l = Learner(method, 81001, 0.1, dtype=torch.float64)
    c, a, _r, _z = data()
    for block in l.blocks:
        b = NumericBlock(block)
        for ci, ai in zip(c, a):
            left, right, s, t = b.pieces(ci.numpy(), ai.item(), -10, 10)
            points = left + 0.371 * (right - left)
            truth = (
                block(
                    ci.expand(len(points), -1),
                    ai.expand(len(points)),
                    torch.from_numpy(points),
                )
                .detach()
                .numpy()
            )
            np.testing.assert_allclose(s * points + t, truth, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("method", ["structured", "flexible", "structured_mse"])
def test_network_identity_bounds_quadrature_and_mc(method):
    l = Learner(method, 81000, 0.1, dtype=torch.float64)
    c, a, r, _z = data()
    # Includes genuinely trained networks, not insertion of ideal analytic functions.
    for _ in range(80):
        l.update(c, a, r, batch=32)
    c, a = c[:3], a[:3]
    values, meta = readouts(l, c.numpy(), a.numpy(), TOL["quadrature_tolerance"])
    assert meta["max_refinement"] <= TOL["quadrature_tolerance"]
    refined, _ = readouts(l, c.numpy(), a.numpy(), 1e-11)
    np.testing.assert_allclose(
        values["value"], refined["value"], atol=TOL["identity_tolerance"], rtol=0
    )
    tau = math.erf(math.sqrt(0.5) / (2 * math.sqrt(0.5) * math.sqrt(2)))
    if method.startswith("structured"):
        k = values["kappa"]
        assert np.all(k >= 0) and np.all(k <= tau + TOL["identity_tolerance"])
        np.testing.assert_allclose(
            values["value"],
            k * values["d1"] + (1 - k) * values["d2"],
            atol=TOL["identity_tolerance"],
            rtol=0,
        )
    for j, (ci, ai) in enumerate(zip(c.numpy(), a.numpy())):
        mu = [0.05, 0.47, 0.93][j]
        v1, _, _ = integral(NumericBlock(l.blocks[0]), ci, ai, mu=mu)
        v20, _, _ = integral(
            NumericBlock(l.blocks[1]), ci, ai, 0, math.sqrt(0.5), mu=mu
        )
        v21, _, _ = integral(
            NumericBlock(l.blocks[1]), ci, ai, math.sqrt(0.5), math.sqrt(0.5), mu=mu
        )
        v2 = (1 - mu) * v20 + mu * v21
        error = abs(values["value"][j] - mu)
        assert (
            error
            <= tau * math.sqrt(max(0, v1[1]) / 2)
            + math.sqrt(max(0, v2[1]) / 2)
            + TOL["identity_tolerance"]
        )
        assert (
            error
            <= tau * math.sqrt(max(0, 2 * v1[2]))
            + math.sqrt(max(0, 2 * v2[2]))
            + TOL["identity_tolerance"]
        )
    streams = [generator(81000, "mc", j) for j in range(3)]
    pred, inter, p2 = sample(l, c, a, TOL["mc_samples"], streams, trace=True)
    se = p2.std(1, unbiased=True) / math.sqrt(TOL["mc_samples"])
    assert torch.all(
        torch.abs(pred - torch.from_numpy(values["value"]))
        < TOL["mc_standard_errors"] * se + TOL["quadrature_tolerance"]
    )
    assert set(inter.unique().tolist()) == {0.0, 1.0}
    streams = [generator(81000, "mc", j) for j in range(3)]
    wrong, inter_wrong, _ = sample(
        l, c, a, TOL["mc_samples"], streams, mean_substitute=True, trace=True
    )
    assert torch.any((inter_wrong > 0) & (inter_wrong < 1))
    assert torch.max(torch.abs(wrong - pred)) > 1e-4


@pytest.mark.parametrize("method", ["structured", "flexible", "ensemble"])
def test_resume_rng_and_evaluation_isolation(method, tmp_path):
    from neural.io import save_checkpoint

    l = Learner(method, 81002, 0.01)
    c, a, r, _z = data(torch.float32)
    global_t = torch.get_rng_state().clone()
    global_n = np.random.get_state()
    for _ in range(11):
        l.update(c, a, r)
    before = l.snapshot()
    readouts(l, c.numpy(), a.numpy())
    if l.composed:
        sample(l, c, a, 32, [generator(99, "inference", j) for j in range(3)])
    assert torch.equal(before["batch_rng"], l.batch_rng.get_state())
    assert all(
        torch.equal(x, g.get_state()) for x, g in zip(before["noise_rng"], l.noise_rng)
    )
    assert not torch.equal(l.noise_rng[0].get_state(), l.noise_rng[1].get_state())
    assert torch.equal(global_t, torch.get_rng_state())
    np.testing.assert_array_equal(global_n[1], np.random.get_state()[1])
    save_checkpoint(
        tmp_path / "resume.pt",
        {"state": before, "triples": (c, a, r), "config": {"method": method}},
    )
    twin = Learner(method, 81002, 0.01)
    twin.restore(torch.load(tmp_path / "resume.pt", weights_only=False)["state"])
    for _ in range(17):
        assert torch.equal(l.update(c, a, r), twin.update(c, a, r))
    assert l.step == twin.step
    for b, b2 in zip(l.blocks, twin.blocks):
        assert all(torch.equal(p, q) for p, q in zip(b.parameters(), b2.parameters()))
    assert torch.equal(l.batch_rng.get_state(), twin.batch_rng.get_state())


def test_stability_counts_no_oracle_in_core():
    expected = {
        "structured": 2498,
        "flexible": 2562,
        "direct": 1249,
        "wide": 2539,
        "ensemble": 2498,
        "logistic": 6,
    }
    for method, n in expected.items():
        l = Learner(method, 81000, 0.1)
        assert (
            sum(p.numel() for b in l.blocks for p in b.parameters() if p.requires_grad)
            == n
        )
    u = torch.tensor([-1e4, 1e4, -100.0, 100.0], requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(u, torch.tensor([1.0, 0.0, 0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss) and torch.all(torch.isfinite(u.grad))
    assert torch.all((u.sigmoid() >= 0) & (u.sigmoid() <= 1))
    from neural import core

    source = Path(core.__file__).read_text()
    assert "from .environment" not in source and "means(" not in source


@pytest.mark.parametrize(
    "method,k",
    [
        ("legacy_mse", 0),
        ("legacy_mse", 1),
        ("legacy_bce", 1),
        ("legacy_d1_mse", 0),
        ("legacy_d2_mse", 0),
    ],
)
def test_logistic_controls_match_historical_updates(method, k):
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "4_Noprop_fst_Advance")
    )
    from noprop_rl.model import LogisticBlock

    learner = Learner(method, 81000, 0.1, dim=0)
    c = torch.empty((7, 0), dtype=torch.float64)
    _, a, r, z = data()
    old = LogisticBlock()
    for _ in range(5):
        inputs = z.numpy()
        if learner.blocks[k].index == 2:
            inputs = math.sqrt(0.5) * r.numpy() + math.sqrt(0.5) * inputs
        expected = old.update(inputs, a.numpy(), r.numpy(), learner.loss_name, 0.1)
        actual = learner.local_update(k, c, a, r, z).item()
        assert abs(expected - actual) < 1e-14
        np.testing.assert_allclose(
            old.parameters,
            learner.blocks[k].weights[0].detach().numpy()[:, 0],
            atol=1e-14,
            rtol=0,
        )
