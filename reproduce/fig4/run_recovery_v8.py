from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(
    os.environ.get("AUTOLOGIC_RECOVERY_OUTPUT", REPO_ROOT / "outputs" / "fig4_recovery_v8")
).resolve()
LOGS = RUN_ROOT / "logs"
TABLES = RUN_ROOT / "tables"
OVERLAY = REPO_ROOT / "compat"

PYTHON_EXE = os.environ.get("AUTOLOGIC_PYTHON_EXE", sys.executable)
LLM = os.environ.get("AUTOLOGIC_LLM", "gpt-3.5-turbo")
TIMEOUT_SECONDS = int(os.environ.get("AUTOLOGIC_JOB_TIMEOUT_SECONDS", "10800"))
STAGE_TIMEOUT_SECONDS = os.environ.get("AUTOLOGIC_STAGE_TIMEOUT_SECONDS", "300")

CLASS_SCRIPT = REPO_ROOT / "reproduce" / "fig4" / "snapshots" / "v6" / "classification_auto_ensemble.py"
MULTI_SCRIPT = REPO_ROOT / "reproduce" / "fig4" / "snapshots" / "v6" / "multiclassification_auto_ensemble.py"
REG_SCRIPT = REPO_ROOT / "reproduce" / "fig4" / "snapshots" / "v6" / "regression_auto_ensemble.py"
CLUSTER_SCRIPT = REPO_ROOT / "legacy_sage" / "ensemble" / "cluster_ensemble" / "cluster_auto_ensemble_cached_labels.py"


def common_args(dataset: str, seed: int) -> list[str]:
    return [
        "-d",
        dataset,
        "-s",
        str(seed),
        "-e",
        "1",
        "-l",
        LLM,
        "-f",
        "10",
        "-m",
        "7",
        "-p",
        "5",
        "--enable_optimization",
        "--enable_feedback",
        "--stage_timeout_s",
        STAGE_TIMEOUT_SECONDS,
    ]


def make_job(job_no: str, task: str, dataset: str, seed: int, script: Path, route: str, extra: list[str] | None = None) -> dict:
    argv = common_args(dataset, seed)
    if extra:
        argv.extend(extra)
    return {
        "job_no": job_no,
        "task": task,
        "dataset": dataset,
        "dataset_arg": dataset,
        "seed": seed,
        "framework": "SAGE-Loop" if route.startswith("sageloop") else "autoLOGIC_compare",
        "route": route,
        "setting": "f10_m7_p5",
        "script": str(script),
        "workdir": str(script.parent),
        "argv": argv,
    }


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    jobs.append(make_job("0006", "classification", "cd2", 42, CLASS_SCRIPT, "compare_binary_f10_m7_p5_recovery"))

    multiclass = {
        "balance-scale": ["0035", "0055", "0075", "0095"],
        "cmc": ["0016", "0036", "0056", "0076", "0096"],
        "eucalyptus": ["0017", "0037", "0057", "0077", "0097"],
        "jungle_chess": ["0018", "0038", "0058", "0078", "0098"],
        "car": ["0019", "0039", "0059", "0079", "0099"],
        "vehicle": ["0020", "0040", "0060", "0080", "0100"],
    }
    seeds_by_index = {
        0: 42,
        1: 43,
        2: 44,
        3: 45,
        4: 46,
    }
    for dataset, job_numbers in multiclass.items():
        if dataset == "balance-scale":
            seeds = [43, 44, 45, 46]
        else:
            seeds = [42, 43, 44, 45, 46]
        for job_no, seed in zip(job_numbers, seeds):
            jobs.append(make_job(job_no, "classification", dataset, seed, MULTI_SCRIPT, "compare_multiclass_f10_m7_p5_recovery"))

    for dataset, job_numbers in {
        "crab": ["0104", "0114", "0124", "0134", "0144"],
        "forest": ["0108", "0118", "0128", "0138", "0148"],
    }.items():
        for idx, job_no in enumerate(job_numbers):
            jobs.append(make_job(job_no, "regression", dataset, seeds_by_index[idx], REG_SCRIPT, "compare_regression_f10_m7_p5_recovery"))

    cluster_extra = ["--min_ensemble_ari", "0.30", "--top_k_values", "1", "2", "3", "5"]
    cluster_specs = [
        ("0156", "cd2", 42),
        ("0166", "cd2", 43),
        ("0176", "cd2", 44),
        ("0186", "cd2", 45),
        ("0196", "cd2", 46),
        ("0159", "ld2", 42),
        ("0169", "ld2", 43),
        ("0179", "ld2", 44),
        ("0189", "ld2", 45),
        ("0199", "ld2", 46),
        ("0191", "breast", 46),
        ("0182", "glass", 45),
        ("0192", "glass", 46),
        ("0195", "seeds", 46),
        ("0197", "ld1", 46),
        ("0198", "cd1", 46),
        ("0190", "cc3", 45),
        ("0200", "cc3", 46),
    ]
    for job_no, dataset, seed in cluster_specs:
        jobs.append(
            make_job(
                job_no,
                "clustering",
                dataset,
                seed,
                CLUSTER_SCRIPT,
                "sageloop_clustering_cached_labels_f10_m7_p5_recovery",
                cluster_extra,
            )
        )
    jobs.sort(key=lambda x: int(x["job_no"]))
    if len(jobs) != 58:
        raise RuntimeError(f"Expected exactly 58 jobs, got {len(jobs)}")
    return jobs


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict, fields: list[str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def load_done(status_path: Path) -> set[str]:
    if not status_path.exists():
        return set()
    with status_path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["job_no"] for row in csv.DictReader(f) if row.get("job_no")}


def kill_tree(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def run_job(job: dict, status_path: Path, status_fields: list[str]) -> None:
    name = f"{job['job_no']}_{job['task']}_{job['dataset']}_seed{job['seed']}_{job['setting']}"
    stdout_path = LOGS / f"{name}.stdout.log"
    stderr_path = LOGS / f"{name}.stderr.log"

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["FIG5_OPENAI_TIMEOUT_S"] = env.get("FIG5_OPENAI_TIMEOUT_S", "120")
    env["PYTHONPATH"] = str(OVERLAY) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [PYTHON_EXE, job["script"], *job["argv"]]
    start = time.time()
    current = {
        "job_no": job["job_no"],
        "task": job["task"],
        "dataset": job["dataset"],
        "seed": job["seed"],
        "route": job["route"],
        "setting": job["setting"],
        "pid": "",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "timeout_seconds": TIMEOUT_SECONDS,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    (TABLES / "current_job.json").write_text(json.dumps(current, indent=2), encoding="utf-8")

    with stdout_path.open("w", encoding="utf-8", errors="replace") as out, stderr_path.open("w", encoding="utf-8", errors="replace") as err:
        proc = subprocess.Popen(cmd, cwd=job["workdir"], env=env, stdout=out, stderr=err)
        current["pid"] = proc.pid
        (TABLES / "current_job.json").write_text(json.dumps(current, indent=2), encoding="utf-8")
        timed_out = False
        last_heartbeat = 0.0
        while True:
            rc = proc.poll()
            elapsed = time.time() - start
            now = time.time()
            if now - last_heartbeat >= 60:
                heartbeat = dict(current)
                heartbeat["elapsed_sec"] = round(elapsed, 2)
                heartbeat["heartbeat_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                (TABLES / "heartbeat.json").write_text(json.dumps(heartbeat, indent=2), encoding="utf-8")
                last_heartbeat = now
            if rc is not None:
                break
            if elapsed > TIMEOUT_SECONDS:
                timed_out = True
                kill_tree(proc.pid)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                break
            time.sleep(5)

    runtime_sec = round(time.time() - start, 2)
    if timed_out:
        status = "timeout"
        exit_code = -999
    else:
        exit_code = proc.returncode
        status = "completed" if exit_code == 0 else "failed"

    row = {
        "job_no": job["job_no"],
        "framework": job["framework"],
        "task": job["task"],
        "dataset": job["dataset"],
        "dataset_arg": job["dataset_arg"],
        "seed": job["seed"],
        "route": job["route"],
        "setting": job["setting"],
        "status": status,
        "exit_code": exit_code,
        "runtime_sec": runtime_sec,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "notes": "fixed 58-job f10_m7_p5 recovery; no retries; outer timeout 10800s",
    }
    append_csv(status_path, row, status_fields)


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_BASE_URL") or not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY must be set in the runner environment.")

    jobs = build_jobs()
    status_path = TABLES / "run_status.csv"
    status_fields = [
        "job_no",
        "framework",
        "task",
        "dataset",
        "dataset_arg",
        "seed",
        "route",
        "setting",
        "status",
        "exit_code",
        "runtime_sec",
        "stdout",
        "stderr",
        "notes",
    ]
    manifest_fields = ["job_no", "framework", "task", "dataset", "dataset_arg", "seed", "route", "setting", "script", "workdir", "argv"]
    manifest_rows = [{**j, "argv": " ".join(j["argv"])} for j in jobs]
    write_csv(TABLES / "jobs_manifest.csv", manifest_rows, manifest_fields)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "Run only the 58 missing seed-level experiments from trusted v1-v7 completion table",
        "job_count_fixed": len(jobs),
        "runner_will_not_add_jobs": True,
        "no_retries": True,
        "timeout_seconds_per_job": TIMEOUT_SECONDS,
        "llm": LLM,
        "api_base": os.environ.get("OPENAI_BASE_URL"),
        "api_key_written_to_files": False,
        "setting": "f10_m7_p5 for all 58 jobs",
        "crab_note": "crab is run with autoLOGIC_compare regression to satisfy f10_m7_p5, differing from the original SAGE-Loop route in v6 manifest",
        "cluster_note": "clustering uses cached-label SAGE cluster script with f10_m7_p5, min_ensemble_ari=0.30, top_k_values=1,2,3,5",
    }
    (TABLES / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (TABLES / "runner_pid.txt").write_text(str(os.getpid()), encoding="utf-8")

    done = load_done(status_path)
    for job in jobs:
        if job["job_no"] in done:
            continue
        run_job(job, status_path, status_fields)
        done.add(job["job_no"])

    (TABLES / "current_job.json").write_text(json.dumps({"status": "all_jobs_finished", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
