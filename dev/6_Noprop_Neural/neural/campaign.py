"""Persistent, gated experiment execution. No selection uses test or oracle means."""

import concurrent.futures
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import time
import traceback

import numpy as np
import torch

from .core import METHODS, Learner
from .environment import triples
from .evaluation import bce, evaluate
from .io import ROOT, digest, event, json_write, now, provenance, save_checkpoint
from .quadrature import readouts


def config(name):
    return json.loads((ROOT / "configs" / f"{name}.json").read_text())


def job_id(job):
    return (
        f"{job['stage']}_{job['family']}_n{job['n']}_s{job['seed']}_{job['method']}_"
        + hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:12]
    )


def jobs(stage, rates=None):
    base = config("base")
    spec = config(stage.lower())
    if stage == "N1":
        for i, n, s, m in itertools.product(
            range(9), spec["observations_per_arm"], spec["seeds"], spec["methods"]
        ):
            yield {
                "stage": stage,
                "family": f"constant{i}",
                "instance": spec["instances"][i],
                "n": n,
                "seed": s,
                "method": m,
                "beta": 0,
                "dim": 0,
                "lr": spec["lr"],
                "base": base,
            }
    elif stage == "pilot":
        for f, b, s, m, lr in itertools.product(
            spec["families"],
            spec["beta"],
            spec["seeds"],
            METHODS,
            spec["learning_rates"],
        ):
            yield {
                "stage": stage,
                "family": f,
                "n": 2048,
                "seed": s,
                "method": m,
                "beta": b,
                "dim": 2,
                "lr": lr,
                "base": base,
            }
    elif stage == "N2":
        assert rates is not None
        for n, s, f, m in itertools.product(
            spec["train_triplets"], spec["seeds"], spec["families"], METHODS
        ):
            yield {
                "stage": stage,
                "family": f,
                "n": n,
                "seed": s,
                "method": m,
                "beta": 2,
                "dim": 2,
                "lr": rates[m],
                "base": base,
            }
    else:
        raise ValueError(stage)


def finite(learner, losses):
    if not torch.isfinite(losses).all() or not all(
        torch.isfinite(p).all() for b in learner.blocks for p in b.parameters()
    ):
        raise FloatingPointError(
            "Non-finite loss or parameters; no seed discarded or gradient clipping"
        )


def run_job(job, origin):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    jid = job_id(job)
    path = ROOT / "results/raw" / job["stage"] / jid
    path.mkdir(parents=True, exist_ok=True)
    cp = ROOT / "checkpoints" / job["stage"] / f"{jid}.pt"
    final = path / "result.json"
    if final.exists():
        result = json.loads(final.read_text())
        if result["status"] == "completed":
            return {"id": jid, "status": "completed", "cached": True}
    json_write(path / "config.json", job)
    event(path / "events.jsonl", event="start_or_resume", pid=os.getpid())
    started = time.perf_counter()
    learner = Learner(job["method"], job["seed"], job["lr"], dim=job["dim"])
    history = []
    train_seconds = 0
    eval_seconds = 0
    try:
        if cp.exists():
            saved = torch.load(cp, weights_only=False, map_location="cpu")
            if (
                saved["config"] != job
                or saved["provenance"]["source_sha256"] != origin["source_sha256"]
            ):
                raise RuntimeError(
                    "Checkpoint code/config differs; preserve it and declare an amendment"
                )
            learner.restore(saved["learner"])
            arrays = saved["data"]
            history = saved["history"]
            train_seconds = saved["train_seconds"]
            eval_seconds = saved["evaluation_seconds"]
        else:
            arrays = triples(
                job["seed"],
                job["family"],
                job["beta"],
                job["n"],
                "train",
                job.get("instance"),
            )
        c, a, r = map(torch.from_numpy, arrays)
        c, r = c.to(learner.dtype), r.to(learner.dtype)

        def checkpoint():
            save_checkpoint(
                cp,
                {
                    "learner": learner.snapshot(),
                    "data": arrays,
                    "config": job,
                    "provenance": origin,
                    "history": history,
                    "train_seconds": train_seconds,
                    "evaluation_seconds": eval_seconds,
                    "utc": now(),
                },
            )
            json_write(
                path / "progress.json",
                {
                    "id": jid,
                    "step": learner.step,
                    "total": job["base"]["updates"],
                    "utc": now(),
                    "pid": os.getpid(),
                    "train_seconds": train_seconds,
                    "evaluation_seconds": eval_seconds,
                },
            )

        if not history and job["stage"] == "N1":
            t = time.perf_counter()
            initial, _ = evaluate(learner, job, full=False)
            eval_seconds += time.perf_counter() - t
            history.append({"step": 0, "train_seconds": 0, **initial})
            checkpoint()
        while learner.step < job["base"]["updates"]:
            t = time.perf_counter()
            target = min(
                learner.step + job["base"]["save_every"], job["base"]["updates"]
            )
            while learner.step < target:
                losses = learner.update(c, a, r, batch=job["base"]["batch"])
                if learner.step % 100 == 0:
                    finite(learner, losses)
            train_seconds += time.perf_counter() - t
            row = {
                "step": learner.step,
                "train_seconds": train_seconds,
                "local_loss_last_batch": losses.tolist(),
            }
            if job["stage"] == "N1" and learner.step in job["base"]["checkpoint_steps"]:
                t = time.perf_counter()
                value, _ = evaluate(learner, job, full=False)
                eval_seconds += time.perf_counter() - t
                row.update(value)
            history.append(row)
            checkpoint()
        finite(learner, torch.tensor(0.0))
        t = time.perf_counter()
        if job["stage"] == "pilot":
            vc, va, vr = triples(
                job["seed"], job["family"], job["beta"], 4096, "validation"
            )
            qt = time.perf_counter()
            pred, quad = readouts(learner, vc, va)
            qt = time.perf_counter() - qt
            value = {
                "validation_value_bce": float(bce(pred["value"], vr).mean()),
                "quadrature": quad,
                "quadrature_seconds": qt,
            }
            raw = {"contexts": vc, "actions": va, "rewards": vr, **pred}
        else:
            value, raw = evaluate(learner, job)
            if job["stage"] == "N1":
                tabular = np.array([arrays[2][arrays[1] == i].mean() for i in range(2)])
                value["tabular_primary"] = float(
                    np.max(abs(tabular - np.array(job["instance"])))
                )
        eval_seconds += time.perf_counter() - t
        np.savez_compressed(path / "predictions.npz", **raw)
        checkpoint()
        output = {
            "id": jid,
            "status": "completed",
            "utc": now(),
            "config": job,
            "provenance": origin,
            "history": history,
            "train_seconds": train_seconds,
            "evaluation_seconds": eval_seconds,
            "current_session_wall_seconds": time.perf_counter() - started,
            "parameters": sum(
                p.numel()
                for b in learner.blocks
                for p in b.parameters()
                if p.requires_grad
            ),
            "train_action_counts": np.bincount(arrays[1], minlength=2).tolist(),
            "data_sha256": hashlib.sha256(
                b"".join(x.tobytes() for x in arrays)
            ).hexdigest(),
            "checkpoint": str(cp.relative_to(ROOT)),
            "checkpoint_sha256": digest(cp),
            **value,
        }
        json_write(final, output)
        event(path / "events.jsonl", event="completed", step=learner.step)
        return {
            "id": jid,
            "status": "completed",
            "train_seconds": train_seconds,
            "evaluation_seconds": eval_seconds,
        }
    except Exception as exc:  # noqa: BLE001 -- preserve every scientific failure
        failure = {
            "id": jid,
            "status": "failed",
            "utc": now(),
            "step": learner.step,
            "type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "config": job,
            "provenance": origin,
        }
        event(path / "failures.jsonl", **failure)
        json_write(path / "failure.json", failure)
        return {"id": jid, "status": "failed", "error": str(exc)}


def n0_gate():
    summary = json.loads((ROOT / "results/n0/latest.json").read_text())
    if summary["status"] != "passed":
        raise RuntimeError("N0 did not pass")
    # N0 gates actual scientific source; scripts/report generation can evolve before freezing.
    current = provenance()["source_files"]
    for p, h in summary["provenance"]["source_files"].items():
        if (
            "/neural/" in p or "/tests/" in p or p.endswith(("n0.json", "base.json"))
        ) and current.get(p) != h:
            raise RuntimeError(f"N0 must be rerun after change to {p}")


def stage_run(stage, workers, origin, rates=None):
    alljobs = list(jobs(stage, rates))
    total = len(alljobs)
    # A separate result for every seed/configuration, never gradients shared across workers.
    ledger = ROOT / "results/ledger.jsonl"
    done = []
    todo = []
    for j in alljobs:
        result = ROOT / "results/raw" / stage / job_id(j) / "result.json"
        if result.exists() and json.loads(result.read_text())["status"] == "completed":
            done.append(job_id(j))
        else:
            todo.append(j)
    t = time.perf_counter()
    initial = len(done)
    state = {
        "stage": stage,
        "total": total,
        "completed": len(done),
        "failed": 0,
        "started_utc": now(),
        "pid": os.getpid(),
        "workers": workers,
    }
    json_write(ROOT / "results/status.json", state)
    event(ledger, event="stage_start", **state)
    stopped = False
    with concurrent.futures.ProcessPoolExecutor(
        workers, mp_context=mp.get_context("spawn")
    ) as pool:
        pending = {}
        it = iter(todo)

        def fill():
            while len(pending) < workers:
                j = next(it, None)
                if j is None:
                    break
                pending[pool.submit(run_job, j, origin)] = job_id(j)

        fill()
        while pending:
            ready, _ = concurrent.futures.wait(
                pending, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in ready:
                jid = pending.pop(future)
                try:
                    result = future.result()
                except Exception:  # noqa: BLE001 -- persist worker failures, then stop stage
                    result = {
                        "id": jid,
                        "status": "failed",
                        "error": traceback.format_exc(),
                    }
                event(ledger, stage=stage, **result)
                if result["status"] == "failed":
                    state["failed"] += 1
                    stopped = True
                else:
                    done.append(jid)
            if (ROOT / "results/STOP").exists():
                stopped = True
            state.update(
                completed=len(done),
                active=list(pending.values()),
                updated_utc=now(),
                elapsed_seconds=time.perf_counter() - t,
                eta_seconds=(time.perf_counter() - t)
                / (len(done) - initial)
                * (total - len(done))
                if len(done) > initial
                else None,
            )
            json_write(ROOT / "results/status.json", state)
            if ready:
                print(json.dumps(state), flush=True)
            if not stopped:
                fill()
    state["status"] = (
        "completed"
        if len(done) == total and not state["failed"]
        else ("failed" if state["failed"] else "stopped")
    )
    json_write(ROOT / f"results/{stage}_summary.json", state)
    json_write(ROOT / "results/status.json", state)
    if state["status"] != "completed":
        raise RuntimeError(
            f"{stage}: {state['status']}; inspect ledger; resume explicitly"
        )
    return state


def freeze(origin):
    n0_gate()
    for stage in ("N1", "pilot"):
        if (
            json.loads((ROOT / f"results/{stage}_summary.json").read_text())["status"]
            != "completed"
        ):
            raise RuntimeError(stage + " incomplete")
    results = []
    for j in jobs("pilot"):
        p = ROOT / "results/raw/pilot" / job_id(j) / "result.json"
        r = json.loads(p.read_text())
        if r["provenance"]["source_sha256"] != origin["source_sha256"]:
            raise RuntimeError("Mixed pilot source versions")
        results.append((j, r))
    rates = {}
    selection = {}
    for method in METHODS:
        losses = {
            str(lr): float(
                np.mean(
                    [
                        r["validation_value_bce"]
                        for j, r in results
                        if j["method"] == method and j["lr"] == lr
                    ]
                )
            )
            for lr in config("pilot")["learning_rates"]
        }
        rates[method] = min(
            config("pilot")["learning_rates"], key=lambda lr: (losses[str(lr)], lr)
        )
        selection[method] = losses
    value = {
        "status": "frozen_before_N2",
        "utc": now(),
        "base": config("base"),
        "N2": config("n2"),
        "pilot": config("pilot"),
        "selected_rates": rates,
        "selection_bce": selection,
        "provenance": origin,
        "sources": config("sources"),
        "N0_summary_sha256": digest(ROOT / "results/n0/latest.json"),
        "pilot_results": {
            r["id"]: digest(ROOT / "results/raw/pilot" / r["id"] / "result.json")
            for _, r in results
        },
        "selection_rule": "all 80 validation environments; smallest mean value BCE; no expansion of predeclared grid in this first campaign; boundary choices reported",
        "hardware": {
            "device": "cpu",
            "workers": 4,
            "threads_per_worker": 1,
            "benchmark_cpu_sha256": digest(ROOT / "results/benchmark_cpu.json"),
            "benchmark_cuda_sha256": digest(ROOT / "results/benchmark_cuda.json"),
        },
    }
    path = ROOT / "configs/n2_frozen.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if (
            existing["selected_rates"] != rates
            or existing["provenance"]["source_sha256"] != origin["source_sha256"]
        ):
            raise RuntimeError("Cannot overwrite a frozen confirmatory protocol")
        return existing
    json_write(path, value)
    event(
        ROOT / "results/ledger.jsonl",
        event="protocol_frozen",
        sha256=digest(path),
        rates=rates,
    )
    return value
