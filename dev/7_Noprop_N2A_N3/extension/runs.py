import hashlib
import itertools
import json
import os
import time
import traceback

import numpy as np
import torch
from neural.evaluation import value_metrics
from neural.quadrature import readouts

from . import OLD, ROOT
from .agents import InteractiveAgent, MSELearner
from .environment import PrivateEnvironment
from .evaluation import diagnostic, trajectory_metrics
from .io import (
    config,
    digest,
    event,
    json_write,
    now,
    npz_write,
    paths,
    save_checkpoint,
)


def n2a_jobs():
    spec = config("n2a")
    old_results = {}
    for p in sorted((OLD / "results/raw/N2").glob("*/result.json")):
        d = json.loads(p.read_text())
        c = d["config"]
        if c["method"] == "structured_mse":
            old_results[c["n"], c["seed"], c["family"]] = (p, d)
    for n, seed, family in itertools.product(
        spec["n"], spec["seeds"], spec["families"]
    ):
        p, d = old_results[n, seed, family]
        yield {
            "stage": "N2A",
            "n": n,
            "seed": seed,
            "family": family,
            "beta": 2,
            "method": "ensemble_mse",
            "lr": 0.1,
            "batch": 128,
            "updates": 5000,
            "architecture": "2 independent MLP; inputs context2 + action_onehot2; hidden [32,32], ReLU, scalar logit",
            "initialization": "independent He hidden / output sqrt(1/fan_in), zero biases",
            "loss": "0.5 * mean((sigmoid(logit) - real_reward)**2)",
            "optimizer": "SGD(momentum=0,weight_decay=0), no clipping, FP32",
            "source_frozen_protocol_sha256": config("sources")["files"][
                "dev/6_Noprop_Neural/configs/n2_frozen.json"
            ],
            "source_result": str(p.relative_to(OLD)),
            "source_checkpoint": d["checkpoint"],
            "source_checkpoint_sha256": d["checkpoint_sha256"],
            "source_data_sha256": d["data_sha256"],
            "source_prediction_sha256": digest(p.parent / "predictions.npz"),
        }


def online_jobs(stage, frozen=None):
    spec = config("pilot_preregistered" if stage == "N3pilot" else "n3_preregistered")
    betas = spec["beta"] if isinstance(spec["beta"], list) else [spec["beta"]]
    regimes = ["interactions"] if stage == "N3pilot" else spec["regimes"]
    for regime, seed, family, beta, method in itertools.product(
        regimes, spec["seeds"], spec["families"], betas, spec["methods"]
    ):
        rates = (
            spec["learning_rates"]
            if stage == "N3pilot"
            else [frozen["selected_rates"][method]]
        )
        for lr in rates:
            yield {
                "stage": stage,
                "regime": regime,
                "seed": seed,
                "family": family,
                "beta": beta,
                "method": method,
                "lr": lr,
                "base": config("base"),
                "loss": "BCEWithLogits on real rewards, separate local blocks",
                "interactions": spec["interactions"],
                "capacity": spec.get("time_max_interactions", spec["interactions"])
                if regime == "time"
                else spec["interactions"],
                "time_budget": frozen["time_budget_seconds"]
                if regime == "time"
                else None,
                "diagnostic_interactions": spec.get("diagnostic_interactions", []),
                "evaluation_contexts": spec.get("evaluation_contexts", 4096),
                "frozen_sha256": digest(ROOT / "configs/n3_frozen.json")
                if stage == "N3"
                else None,
            }


def worker(job, origin):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    jid, folder, _cp = paths(job)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        result = json.loads((folder / "result.json").read_text())
        if (
            result["config"] != job
            or result["provenance"]["source_sha256"] != origin["source_sha256"]
        ):
            raise RuntimeError("Existing result config/source differs")
        return {"id": jid, "status": "completed", "cached": True}
    json_write(folder / "config.json", job)
    event(folder / "events.jsonl", event="start_or_resume", pid=os.getpid())
    try:
        return (
            run_offline(job, origin)
            if job["stage"] == "N2A"
            else run_online(job, origin)
        )
    except Exception as exc:  # noqa: BLE001 - preserve scientific failures, never discard a seed
        failure = {
            "id": jid,
            "utc": now(),
            "config": job,
            "provenance": origin,
            "status": "failed",
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
        json_write(folder / "failure.json", failure)
        event(folder / "events.jsonl", event="failed", exception=repr(exc))
        return {"id": jid, "status": "failed", "exception": repr(exc)}


def load_saved(cp, job, origin):
    saved = torch.load(cp, weights_only=False, map_location="cpu")
    if (
        saved["config"] != job
        or saved["provenance"]["source_sha256"] != origin["source_sha256"]
    ):
        raise RuntimeError("Refusing resume under different source/config")
    return saved


def run_offline(job, origin):
    jid, folder, cp = paths(job)
    source_cp = OLD / job["source_checkpoint"]
    if digest(source_cp) != job["source_checkpoint_sha256"]:
        raise RuntimeError("Historical checkpoint changed")
    historic = torch.load(source_cp, weights_only=False, map_location="cpu")
    data = historic["data"]
    data_hash = hashlib.sha256(b"".join(x.tobytes() for x in data)).hexdigest()
    if data_hash != job["source_data_sha256"]:
        raise RuntimeError("Historical data identity failed")
    learner = MSELearner(job["method"], job["seed"], job["lr"])
    elapsed, history = 0.0, []
    if cp.exists():
        saved = load_saved(cp, job, origin)
        learner.restore(saved["learner"])
        elapsed, history = saved["train_seconds"], saved["history"]
    c, a, r = map(torch.from_numpy, data)
    while learner.step < job["updates"]:
        start = time.perf_counter()
        end = min(learner.step + 500, job["updates"])
        while learner.step < end:
            loss = learner.update(c, a, r, 128)
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("Nonfinite N2A loss")
        elapsed += time.perf_counter() - start
        history.append(
            {"step": learner.step, "loss": loss.tolist(), "train_seconds": elapsed}
        )
        save_checkpoint(
            cp,
            {
                "config": job,
                "provenance": origin,
                "learner": learner.snapshot(),
                "data": data,
                "train_seconds": elapsed,
                "history": history,
                "utc": now(),
            },
        )
        json_write(
            folder / "progress.json",
            {
                "id": jid,
                "t": learner.step,
                "total": 5000,
                "pid": os.getpid(),
                "utc": now(),
            },
        )
    start = time.perf_counter()
    source_pred = OLD / job["source_result"]
    source_pred = source_pred.parent / "predictions.npz"
    if digest(source_pred) != job["source_prediction_sha256"]:
        raise RuntimeError("Historical prediction changed")
    with np.load(source_pred) as d:
        c, a, mu, r = (d[k].copy() for k in ("contexts", "actions", "mu", "rewards"))
    pred, quad = readouts(learner, c, a)
    npz_write(
        folder / "predictions.npz", contexts=c, actions=a, mu=mu, rewards=r, **pred
    )
    result = {
        "id": jid,
        "utc": now(),
        "status": "completed",
        "config": job,
        "provenance": origin,
        "train_seconds": elapsed,
        "evaluation_seconds": time.perf_counter() - start,
        "data_sha256": data_hash,
        "metrics": {k: value_metrics(v, mu, r) for k, v in pred.items()},
        "quadrature": quad,
        "history": history,
        "checkpoint": str(cp.relative_to(ROOT)),
        "checkpoint_sha256": digest(cp),
    }
    json_write(folder / "result.json", result)
    event(folder / "events.jsonl", event="completed")
    return {"id": jid, "status": "completed"}


def run_online(job, origin):
    jid, folder, cp = paths(job)
    agent = InteractiveAgent(
        job["method"], job["seed"], job["lr"], capacity=job["capacity"]
    )
    env = PrivateEnvironment(job["seed"], job["family"], job["beta"])
    clocks = {
        "decision": 0.0,
        "update": 0.0,
        "environment": 0.0,
        "diagnostic": 0.0,
        "checkpoint": 0.0,
    }
    history, done_evals = [], []
    if cp.exists():
        saved = load_saved(cp, job, origin)
        agent.restore(saved["agent"])
        env.restore(saved["environment"])
        clocks, history, done_evals = (
            saved["clocks"],
            saved["history"],
            saved["done_evals"],
        )

    def deployed():
        return clocks["decision"] + clocks["update"] + clocks["environment"]

    def save():
        start = time.perf_counter()
        save_checkpoint(
            cp,
            {
                "config": job,
                "provenance": origin,
                "agent": agent.snapshot(),
                "environment": env.snapshot(),
                "clocks": clocks,
                "history": history,
                "done_evals": done_evals,
                "utc": now(),
            },
        )
        clocks["checkpoint"] += time.perf_counter() - start
        json_write(
            folder / "progress.json",
            {
                "id": jid,
                "t": agent.size,
                "total": job["interactions"]
                if job["regime"] == "interactions"
                else None,
                "regime": job["regime"],
                "time_budget": job["time_budget"],
                "deployed_seconds": deployed(),
                "updates": agent.learner.step,
                "utc": now(),
                "pid": os.getpid(),
            },
        )

    def evaluate_here():
        if job["stage"] != "N3" or agent.size in done_evals:
            return
        # Save BEFORE diagnostic so a diagnostic crash cannot change the trajectory.
        save()
        record = diagnostic(agent.learner, job, folder, agent.size)
        clocks["diagnostic"] += record["seconds"]
        history.append(record)
        done_evals.append(agent.size)
        save_checkpoint(
            folder / f"checkpoint_t{agent.size}.pt",
            {
                "config": job,
                "provenance": origin,
                "agent": agent.snapshot(),
                "environment": env.snapshot(),
                "clocks": clocks.copy(),
                "history": history,
                "done_evals": done_evals,
                "utc": now(),
            },
        )
        save()

    # Handles resumption after a crash immediately before an evaluation.
    if agent.size in job["diagnostic_interactions"]:
        evaluate_here()
    while agent.size < job["capacity"]:
        if job["regime"] == "time" and deployed() >= job["time_budget"]:
            break
        start = time.perf_counter()
        c = env.context()
        clocks["environment"] += time.perf_counter() - start
        start = time.perf_counter()
        a = agent.choose(c)
        clocks["decision"] += time.perf_counter() - start
        start = time.perf_counter()
        r = env.step(a)
        clocks["environment"] += time.perf_counter() - start
        start = time.perf_counter()
        agent.observe(c, a, r)
        clocks["update"] += time.perf_counter() - start
        if agent.size % 500 == 0:
            save()
        if agent.size in job["diagnostic_interactions"]:
            evaluate_here()
    save()
    evaluate_here()
    c, a, r = [
        x[: agent.size].cpu().numpy().copy() for x in (agent.c, agent.a, agent.r)
    ]
    # Selector sees observed_reward only. Oracle metrics never computed in pilot.
    metrics = {"observed_reward": float(r.mean()), "interactions": agent.size}
    extra = {}
    if job["stage"] == "N3":
        metrics, extra = trajectory_metrics(c, a, r, job["family"], job["beta"])
    npz_write(folder / "trajectory.npz", contexts=c, actions=a, rewards=r, **extra)
    result = {
        "id": jid,
        "utc": now(),
        "status": "completed",
        "config": job,
        "provenance": origin,
        "metrics": metrics,
        "clocks": clocks,
        "deployed_seconds": deployed(),
        "history": history,
        "updates": agent.learner.step,
        "checkpoint": str(cp.relative_to(ROOT)),
        "checkpoint_sha256": digest(cp),
        "time_censored": job["regime"] == "time" and deployed() < job["time_budget"],
        "time_overshoot_seconds": max(0.0, deployed() - job["time_budget"])
        if job["regime"] == "time"
        else None,
    }
    json_write(folder / "result.json", result)
    event(folder / "events.jsonl", event="completed", t=agent.size)
    return {"id": jid, "status": "completed"}
