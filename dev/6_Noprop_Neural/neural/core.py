"""Equations 3.3--3.11. No environment parameters or evaluator imports here."""

import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


def seed_for(seed, *labels):
    return int.from_bytes(
        hashlib.sha256(repr((seed, labels)).encode()).digest()[:8], "big"
    ) % (2**63 - 1)


def generator(seed, *labels, device="cpu"):
    return torch.Generator(device=device).manual_seed(seed_for(seed, *labels))


class Block(nn.Module):
    def __init__(
        self, kind, index, seed, dim=2, width=32, rho=0.5, dtype=torch.float32
    ):
        super().__init__()
        self.kind, self.index, self.dim, self.rho = kind, index, dim, rho
        self.calls = 0
        p = dim + 2 + (kind in ("flexible", "legacy"))
        if kind == "logistic":
            shapes = [(2 * (dim + 1), 1)]
        elif kind == "legacy":
            shapes = [(3, 1)]
        else:
            shapes = [(p, width), (width, width), (width, 1)]
        # No nn.Linear constructor: it would consume the process-global RNG.
        rng = generator(seed, "weights")
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i, (din, dout) in enumerate(shapes):
            gain = math.sqrt(2) if i < len(shapes) - 1 else 1.0
            w = torch.randn((din, dout), generator=rng, dtype=dtype) * (
                gain / math.sqrt(din)
            )
            if kind in ("legacy", "logistic"):
                w.zero_()
            self.weights.append(nn.Parameter(w))
            # Historical logistic has only a slope and two intercepts (no extra bias).
            self.biases.append(
                nn.Parameter(
                    torch.zeros(dout, dtype=dtype),
                    requires_grad=kind not in ("legacy", "logistic"),
                )
            )

    def features(self, c, a, z):
        onehot = F.one_hot(a, 2).to(c.dtype)
        if self.kind == "logistic":
            return torch.cat(
                [onehot] + [c[..., j : j + 1] * onehot for j in range(self.dim)], -1
            )
        if self.kind == "legacy":
            return torch.cat((z[..., None], onehot), -1)
        x = torch.cat((c, onehot), -1)
        return torch.cat((z[..., None], x), -1) if self.kind == "flexible" else x

    def forward(self, c, a, z):
        self.calls += 1
        x = self.features(c, a, z)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = F.relu(x)
        u = x.squeeze(-1)
        if self.kind == "structured" and self.index == 2:
            u = u + (math.sqrt(self.rho) * z - self.rho / 2) / (1 - self.rho)
        return u


METHODS = (
    "logistic",
    "direct",
    "wide",
    "ensemble",
    "structured",
    "flexible",
    "structured_mse",
)


class Learner:
    """Owns only observed triples, local blocks, optimizers and training RNGs."""

    def __init__(
        self, method, seed, lr, dim=2, rho=0.5, device="cpu", dtype=torch.float32
    ):
        if method.startswith("legacy"):
            dtype = torch.float64
        self.method, self.seed, self.lr = method, seed, lr
        self.dim, self.rho, self.device, self.dtype = dim, rho, device, dtype
        self.loss_name = "mse" if method.endswith("mse") else "bce"
        self.composed = method in (
            "structured",
            "structured_mse",
            "flexible",
            "legacy_bce",
            "legacy_mse",
        )
        count = 2 if self.composed or method == "ensemble" else 1
        kind = {
            "wide": "direct",
            "ensemble": "direct",
            "structured_mse": "structured",
            "legacy_bce": "legacy",
            "legacy_mse": "legacy",
            "legacy_d1_mse": "legacy",
            "legacy_d2_mse": "legacy",
        }.get(method, method)
        self.blocks = [
            Block(
                kind,
                2 if method == "legacy_d2_mse" else k + 1,
                seed_for(seed, method, "init", k),
                dim,
                47 if method == "wide" else 32,
                rho,
                dtype,
            ).to(device)
            for k in range(count)
        ]
        self.optimizers = [
            torch.optim.SGD(b.parameters(), lr=lr, momentum=0, weight_decay=0)
            for b in self.blocks
        ]
        self.batch_rng = generator(seed, "minibatch", device=device)
        self.noise_rng = [
            generator(seed, method, "train_noise", k, device=device)
            for k in range(count)
        ]
        self.step = 0

    def local_update(self, k, c, a, r, noise):
        b, opt = self.blocks[k], self.optimizers[k]
        z = noise
        if b.index == 2 and (self.composed or b.kind == "legacy"):
            z = math.sqrt(self.rho) * r + math.sqrt(1 - self.rho) * noise
        opt.zero_grad(set_to_none=True)
        u = b(c, a, z)
        loss = (
            F.binary_cross_entropy_with_logits(u, r)
            if self.loss_name == "bce"
            else 0.5 * (u.sigmoid() - r).square().mean()
        )
        loss.backward()
        opt.step()
        return loss.detach()

    def update(self, c, a, r, batch=128):
        ix = torch.randint(
            len(r), (batch,), generator=self.batch_rng, device=self.device
        )
        cb, ab, rb = c[ix], a[ix], r[ix]
        losses = [
            self.local_update(
                k,
                cb,
                ab,
                rb,
                torch.randn(batch, generator=rng, device=self.device, dtype=self.dtype),
            )
            for k, rng in enumerate(self.noise_rng)
        ]
        self.step += 1
        return torch.stack(losses)

    def snapshot(self):
        return copy.deepcopy(
            {
                "blocks": [b.state_dict() for b in self.blocks],
                "optimizers": [o.state_dict() for o in self.optimizers],
                "batch_rng": self.batch_rng.get_state(),
                "noise_rng": [g.get_state() for g in self.noise_rng],
                "step": self.step,
            }
        )

    def restore(self, state):
        for b, s in zip(self.blocks, state["blocks"]):
            b.load_state_dict(s)
        for o, s in zip(self.optimizers, state["optimizers"]):
            o.load_state_dict(s)
        self.batch_rng.set_state(state["batch_rng"])
        for g, s in zip(self.noise_rng, state["noise_rng"]):
            g.set_state(s)
        self.step = state["step"]


@torch.no_grad()
def sample(learner, c, a, replicas, streams, mean_substitute=False, trace=False):
    """RNGs for z1, Bernoulli and z2 are supplied separately; never training RNGs."""
    cc = c[:, None, :].expand(-1, replicas, -1)
    aa = a[:, None].expand(-1, replicas)
    shape = aa.shape
    z1 = torch.randn(shape, generator=streams[0], dtype=c.dtype, device=c.device)
    p1 = learner.blocks[0](cc, aa, z1).sigmoid()
    intermediate = (
        p1
        if mean_substitute
        else (torch.rand(shape, generator=streams[1], device=c.device) < p1).to(c.dtype)
    )
    noise = torch.randn(shape, generator=streams[2], dtype=c.dtype, device=c.device)
    z2 = math.sqrt(learner.rho) * intermediate + math.sqrt(1 - learner.rho) * noise
    p2 = learner.blocks[1](cc, aa, z2).sigmoid()
    return (p2.mean(1), intermediate, p2) if trace else p2.mean(1)
