import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extension import ROOT
from extension.analysis import analyze_n2a, analyze_n3, freeze_n3, report
from extension.audit import audit
from extension.io import (
    config,
    event,
    json_write,
    manifest,
    now,
    paths,
    snapshot_origin,
    verify_history,
)
from extension.runs import n2a_jobs, online_jobs, worker

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def stopped():
    return STOP or (ROOT / "results/STOP").exists()


def run_stage(stage, jobs, origin, workers):
    jobs = list(jobs)
    remaining = [j for j in jobs if not (paths(j)[1] / "result.json").exists()]
    status = {
        "stage": stage,
        "total": len(jobs),
        "completed": len(jobs) - len(remaining),
        "failed": 0,
        "pid": os.getpid(),
        "utc": now(),
        "state": "running",
    }
    json_write(ROOT / "results/status.json", status)
    if not remaining:
        return True
    iterator = iter(remaining)
    pending = {}
    started = time.perf_counter()
    initial = status["completed"]
    with concurrent.futures.ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:

        def fill():
            while len(pending) < workers and not stopped():
                job = next(iterator, None)
                if job is None:
                    break
                pending[pool.submit(worker, job, origin)] = job

        fill()
        while pending:
            done, _ = concurrent.futures.wait(
                pending, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                pending.pop(future)
                result = future.result()
                event(ROOT / "results/ledger.jsonl", stage=stage, **result)
                if result["status"] == "completed":
                    status["completed"] += 1
                else:
                    status["failed"] += 1
                    if stage != "N3pilot":
                        request_stop()
                print(
                    f"{stage} {status['completed']}/{status['total']} failures={status['failed']} {result['id']}",
                    flush=True,
                )
            status.update(
                utc=now(),
                active=[paths(j)[0] for j in pending.values()],
                elapsed_seconds=time.perf_counter() - started,
            )
            count = status["completed"] - initial + status["failed"]
            status["eta_seconds"] = (
                status["elapsed_seconds"]
                * (len(jobs) - status["completed"] - status["failed"])
                / count
                if count
                else None
            )
            json_write(ROOT / "results/status.json", status)
            fill()
    status.update(state="stopped" if stopped() else "completed", utc=now())
    json_write(ROOT / "results/status.json", status)
    return not stopped()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--stage", choices=["all", "N2A", "N3pilot", "N3"], default="all"
    )
    args = parser.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    lock = (ROOT / "results/campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if stopped():
        raise RuntimeError("STOP exists; rename it before resuming")
    validation = json.loads((ROOT / "results/validation/latest.json").read_text())
    if validation["returncode"] != 0:
        raise RuntimeError("Validation did not pass")
    origin = snapshot_origin()
    if validation["provenance"]["source_sha256"] != origin["source_sha256"]:
        raise RuntimeError("Validate current sources before launch")
    verify_history()
    json_write(
        ROOT / "results/process.json",
        {
            "pid": os.getpid(),
            "utc": now(),
            "stage": args.stage,
            "workers": args.workers,
            "provenance_sha256": origin["source_sha256"],
        },
    )
    event(
        ROOT / "results/campaign_events.jsonl",
        event="start_or_resume",
        pid=os.getpid(),
        stage=args.stage,
    )
    if args.stage in ("all", "N2A"):
        if not run_stage("N2A", n2a_jobs(), origin, args.workers):
            return
        analyze_n2a()
        audit("N2A")
        manifest()
    if args.stage in ("all", "N3pilot"):
        if not run_stage("N3pilot", online_jobs("N3pilot"), origin, args.workers):
            return
        frozen = freeze_n3(origin)
        audit("N3pilot")
        print(
            "FROZEN",
            frozen["selected_rates"],
            "time_budget",
            frozen["time_budget_seconds"],
            flush=True,
        )
        manifest()
    if args.stage in ("all", "N3"):
        frozen = config("n3_frozen")
        if frozen["provenance"]["source_sha256"] != origin["source_sha256"]:
            raise RuntimeError("Frozen source mismatch")
        if not run_stage("N3", online_jobs("N3", frozen), origin, args.workers):
            return
        analyze_n3()
        audit("N3")
    report()
    json_write(
        ROOT / "results/history_preserved_at_completion.json", verify_history(full=True)
    )
    json_write(
        ROOT / "results/status.json",
        {"state": "completed", "stage": args.stage, "utc": now(), "pid": os.getpid()},
    )
    event(ROOT / "results/campaign_events.jsonl", event="completed", stage=args.stage)
    count = manifest()
    print(f"Finished; manifest contains {count} artifacts", flush=True)


if __name__ == "__main__":
    main()
