import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(
    os.environ.get("AUTOLOGIC_RECOVERY_OUTPUT", REPO_ROOT / "outputs" / "fig4_recovery_v9")
).resolve()
LOGS = RUN_ROOT / "logs"
TABLES = RUN_ROOT / "tables"
OVERLAY = REPO_ROOT / "compat"

PYTHON_EXE = os.environ.get("AUTOLOGIC_PYTHON_EXE", sys.executable)
LLM = os.environ.get("AUTOLOGIC_LLM", "gpt-3.5-turbo")
TIMEOUT_SECONDS = int(os.environ.get("AUTOLOGIC_JOB_TIMEOUT_SECONDS", "18000"))
STAGE_TIMEOUT_SECONDS = os.environ.get("AUTOLOGIC_STAGE_TIMEOUT_SECONDS", "300")

MULTI_SCRIPT = REPO_ROOT / "reproduce" / "fig4" / "snapshots" / "v9" / "multiclassification_auto_ensemble.py"
REG_SCRIPT = REPO_ROOT / "reproduce" / "fig4" / "snapshots" / "v9" / "regression_auto_ensemble.py"
CLUSTER_SCRIPT = REPO_ROOT / "legacy_sage" / "ensemble" / "cluster_ensemble" / "cluster_auto_ensemble_cached_labels.py"


def common_args(dataset: str, seed: int) -> list[str]:
    return [
        "-d", dataset,
        "-s", str(seed),
        "-e", "1",
        "-l", LLM,
        "-f", "10",
        "-m", "7",
        "-p", "5",
        "--enable_optimization",
        "--enable_feedback",
        "--stage_timeout_s", STAGE_TIMEOUT_SECONDS,
    ]


def make_job(job_no: str, task: str, dataset: str, seed: int, script: Path, route: str, extra=None) -> dict:
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
    cluster_extra = ["--min_ensemble_ari", "0.30", "--top_k_values", "1", "2", "3", "5"]
    jobs = [
        make_job("0098", "classification", "jungle_chess", 46, MULTI_SCRIPT, "compare_multiclass_f10_m7_p5_recovery_fixed"),
        make_job("0104", "regression", "crab", 42, REG_SCRIPT, "compare_regression_f10_m7_p5_recovery_fixed"),
        make_job("0159", "clustering", "ld2", 42, CLUSTER_SCRIPT, "sageloop_clustering_cached_labels_f10_m7_p5_recovery_fixed", cluster_extra),
        make_job("0169", "clustering", "ld2", 43, CLUSTER_SCRIPT, "sageloop_clustering_cached_labels_f10_m7_p5_recovery_fixed", cluster_extra),
        make_job("0196", "clustering", "cd2", 46, CLUSTER_SCRIPT, "sageloop_clustering_cached_labels_f10_m7_p5_recovery_fixed", cluster_extra),
        make_job("0199", "clustering", "ld2", 46, CLUSTER_SCRIPT, "sageloop_clustering_cached_labels_f10_m7_p5_recovery_fixed", cluster_extra),
        make_job("0200", "clustering", "cc3", 46, CLUSTER_SCRIPT, "sageloop_clustering_cached_labels_f10_m7_p5_recovery_fixed", cluster_extra),
    ]
    jobs.sort(key=lambda x: int(x["job_no"]))
    if len(jobs) != 7:
        raise RuntimeError(f"Expected exactly 7 jobs, got {len(jobs)}")
    return jobs


def kill_tree(pid: int):
    subprocess.run(
        ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_job(job: dict, status_writer):
    name = f"{job['job_no']}_{job['task']}_{job['dataset']}_seed{job['seed']}_f10_m7_p5"
    stdout_path = LOGS / f"{name}.stdout.log"
    stderr_path = LOGS / f"{name}.stderr.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["FIG5_OPENAI_TIMEOUT_S"] = env.get("FIG5_OPENAI_TIMEOUT_S", "120")
    env["PYTHONPATH"] = str(OVERLAY) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [PYTHON_EXE, "-u", job["script"], *job["argv"]]
    started = time.time()
    proc = None
    status = "failed"
    exit_code = None
    with stdout_path.open("w", encoding="utf-8", errors="replace") as out, stderr_path.open("w", encoding="utf-8", errors="replace") as err:
        proc = subprocess.Popen(cmd, cwd=job["workdir"], env=env, stdout=out, stderr=err)
        current = {
            **{k: job[k] for k in ("job_no", "task", "dataset", "seed", "route", "setting")},
            "pid": proc.pid,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "timeout_seconds": TIMEOUT_SECONDS,
            "stage_timeout_seconds": STAGE_TIMEOUT_SECONDS,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        write_json(TABLES / "current_job.json", current)
        while True:
            ret = proc.poll()
            elapsed = time.time() - started
            heartbeat = dict(current)
            heartbeat["elapsed_sec"] = round(elapsed, 2)
            heartbeat["heartbeat_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            write_json(TABLES / "heartbeat.json", heartbeat)
            if ret is not None:
                exit_code = ret
                status = "completed" if ret == 0 else "failed"
                break
            if elapsed >= TIMEOUT_SECONDS:
                kill_tree(proc.pid)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                exit_code = -999
                status = "timeout"
                break
            time.sleep(30)

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
        "runtime_sec": round(time.time() - started, 2),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "notes": "fixed 7-job f10_m7_p5 recovery; no retries; unbuffered logs; patched stage timeout, regression feature checks, clustering deepcopy fallback",
    }
    status_writer.writerow(row)
    status_writer.file.flush()
    return row


class StatusWriter:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "job_no", "framework", "task", "dataset", "dataset_arg", "seed",
                "route", "setting", "status", "exit_code", "runtime_sec",
                "stdout", "stderr", "notes",
            ],
        )
        self.writer.writeheader()

    def writerow(self, row: dict):
        self.writer.writerow(row)

    def close(self):
        self.file.close()


def main():
    LOGS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    jobs = build_jobs()
    write_json(TABLES / "run_manifest.json", {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "Rerun only the 6 failed and 1 timeout tasks from v8 with targeted non-result-changing safeguards",
        "job_count_fixed": len(jobs),
        "runner_will_not_add_jobs": True,
        "no_retries": True,
        "timeout_seconds_per_job": TIMEOUT_SECONDS,
        "stage_timeout_seconds": STAGE_TIMEOUT_SECONDS,
        "llm": LLM,
        "api_base": os.environ.get("OPENAI_BASE_URL") or os.environ.get("AUTOLOGIC_API_BASE"),
        "api_key_written_to_files": False,
        "setting": "f10_m7_p5 for all 7 jobs",
        "patches": [
            "python -u and PYTHONUNBUFFERED=1 for real-time logs",
            "code_overlay.utils.time_limit now raises TimeoutExceeded instead of being a no-op",
            "regression skips generated feature columns absent from train/test outputs",
            "clustering cached-label script falls back when model deepcopy fails",
        ],
    })
    with (TABLES / "jobs_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()))
        w.writeheader()
        w.writerows(jobs)
    (TABLES / "runner_pid.txt").write_text(str(os.getpid()), encoding="utf-8")

    status_writer = StatusWriter(TABLES / "run_status.csv")
    try:
        for job in jobs:
            run_job(job, status_writer)
    finally:
        status_writer.close()
    write_json(TABLES / "current_job.json", {"status": "all_jobs_finished", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})


if __name__ == "__main__":
    main()
