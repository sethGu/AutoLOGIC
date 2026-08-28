from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


PYTHON = os.environ.get("AUTOLOGIC_PYTHON_EXE", sys.executable)
LLM = "gpt-4o-mini"
SEEDS = [42, 44, 46]
CLASSIFICATION_DATASETS = ["cc1", "credit-g"]
REGRESSION_DATASETS = ["boston", "concrete"]
CLUSTERING_DATASETS = ["breast", "glass", "students"]


CONFIG_MAP = {
    "配置1": ("teacher_no_distill_no_calibration", "teacher", "none"),
    "配置2": ("teacher_no_distill_sigmoid_calibration", "teacher", "sigmoid_or_conformal"),
    "配置3": ("student_distilled_sigmoid_calibration", "student", "sigmoid_or_conformal"),
    "配置4": ("student_distilled_no_calibration", "student", "none"),
}


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def mean_std(values: Iterable[Any]) -> Tuple[float, float, int]:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def map_config(raw: str) -> Tuple[str, str, str]:
    raw = raw or ""
    for prefix, mapped in CONFIG_MAP.items():
        if raw.startswith(prefix):
            return mapped
    return raw, "unknown", "unknown"


def latest_jsonl(log_dir: Path, dataset: str, before: set[str]) -> Optional[Path]:
    candidates = [p for p in log_dir.glob(f"{dataset}_*.jsonl") if p.name not in before]
    if not candidates:
        candidates = list(log_dir.glob(f"{dataset}_*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_process(cmd: List[str], cwd: Path, stdout: Path, stderr: Path, env: dict, timeout_s: int) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with stdout.open("w", encoding="utf-8", errors="replace") as out, stderr.open("w", encoding="utf-8", errors="replace") as err:
        out.write("START " + datetime.now().isoformat(timespec="seconds") + "\n")
        out.write("COMMAND " + " ".join(cmd) + "\n")
        out.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=out, stderr=err, env=env)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rc = 124
            out.write(f"\nTIMEOUT killed_pid={proc.pid} timeout_s={timeout_s}\n")
        out.write(f"\nEND returncode={rc} seconds={time.time() - start:.1f}\n")
    return rc


def env_for_run() -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("PYTHONWARNINGS", "ignore")
    cores = str(max(1, min(8, os.cpu_count() or 1)))
    env["OMP_NUM_THREADS"] = cores
    env["MKL_NUM_THREADS"] = cores
    env["OPENBLAS_NUM_THREADS"] = cores
    env["NUMEXPR_NUM_THREADS"] = cores
    return env


def parse_classification(path: Path, dataset: str, seed: int) -> Tuple[List[dict], List[dict]]:
    per_seed, bins = [], []
    for rec in read_jsonl(path):
        if rec.get("stage") != "ensemble_eval":
            continue
        meta = rec.get("meta_config") or {}
        config, family, cal = map_config(meta.get("config_name", ""))
        row = {
            "task": "classification",
            "dataset": dataset,
            "seed": seed,
            "config": config,
            "model_family": family,
            "calibration_method": "sigmoid" if "sigmoid" in config else "none",
            "auc": safe_float(rec.get("test_score")),
            "acc": safe_float(rec.get("accuracy")),
            "ece": safe_float(rec.get("test_trust")),
            "brier": safe_float(rec.get("brier")),
            "nll": safe_float(rec.get("nll")),
            "failed_flag": 0 if rec.get("status") != "error" else 1,
            "failure_reason": rec.get("error", ""),
            "jsonl_path": str(path),
        }
        per_seed.append(row)
        for b in rec.get("reliability_bins") or []:
            bins.append({
                "task": "classification",
                "dataset": dataset,
                "seed": seed,
                "config": config,
                "bin_id": b.get("bin_id"),
                "bin_count": b.get("bin_count"),
                "mean_confidence": b.get("mean_confidence"),
                "empirical_accuracy": b.get("empirical_accuracy"),
            })
    return per_seed, bins


def parse_regression(path: Path, dataset: str, seed: int) -> Tuple[List[dict], List[dict]]:
    per_seed, curve = [], []
    for rec in read_jsonl(path):
        if rec.get("stage") != "ensemble_eval":
            continue
        meta = rec.get("meta_config") or {}
        raw_config, family, cal = map_config(meta.get("config_name", ""))
        is_interval = bool(meta.get("calibrate"))
        if is_interval:
            config = raw_config.replace("sigmoid_calibration", "split_conformal_alpha_0_10").replace("calibration", "split_conformal_alpha_0_10")
            alpha = safe_float(rec.get("alpha"))
            picp = safe_float(rec.get("test_trust"))
            mpiw = safe_float(rec.get("width"))
            coverage_error = abs(picp - (1.0 - alpha)) if math.isfinite(picp) and math.isfinite(alpha) else float("nan")
        else:
            config = raw_config.replace("no_calibration", "point_prediction")
            alpha = float("nan")
            picp = mpiw = coverage_error = float("nan")
        row = {
            "task": "regression",
            "dataset": dataset,
            "seed": seed,
            "config": config,
            "model_family": family,
            "interval_method": "split_conformal" if is_interval else "point",
            "alpha": alpha,
            "mae": safe_float(rec.get("mae")),
            "rmse": safe_float(rec.get("test_score")),
            "r2": safe_float(rec.get("r2")),
            "picp": picp,
            "mpiw": mpiw,
            "pinaw": float("nan"),
            "coverage_error": coverage_error,
            "failed_flag": 0 if rec.get("status") != "skipped" else 1,
            "failure_reason": rec.get("error", rec.get("skip_reason", "")),
            "jsonl_path": str(path),
        }
        per_seed.append(row)
        if is_interval:
            curve.append({k: row[k] for k in ["task", "dataset", "seed", "config", "model_family", "alpha", "picp", "mpiw", "pinaw", "coverage_error"]})
    return per_seed, curve


def parse_clustering(path: Path, dataset: str, seed: int) -> List[dict]:
    rows = []
    for rec in read_jsonl(path):
        if rec.get("stage") != "ensemble_eval":
            continue
        meta = rec.get("meta_config") or {}
        raw_config, family, _ = map_config(meta.get("config_name", ""))
        if family == "unknown":
            family = "student" if "蒸" in str(meta.get("config_name", "")) else "teacher"
        config = raw_config.replace("sigmoid_calibration", "best_single").replace("no_calibration", "best_single")
        rows.append({
            "task": "clustering",
            "dataset": dataset,
            "seed": seed,
            "config": config,
            "model_family": family,
            "postprocessing_method": "cached_label_best_single",
            "selected_k": float("nan"),
            "ari": safe_float(rec.get("test_score")),
            "nmi": safe_float(rec.get("nmi")),
            "silhouette": float("nan"),
            "db": float("nan"),
            "ch": float("nan"),
            "selected_model_ari": json.dumps(meta.get("selected_model_ari"), ensure_ascii=False),
            "eligible_model_count": meta.get("eligible_model_count"),
            "fallback_used": meta.get("fallback_used"),
            "failed_flag": 0 if rec.get("status") != "error" else 1,
            "failure_reason": rec.get("error", ""),
            "quadratic_sample_operations": "disabled",
            "jsonl_path": str(path),
        })
    return rows


def aggregate(df: pd.DataFrame, keys: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for key_vals, g in df.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = {k: v for k, v in zip(keys, key_vals)}
        row["n_rows"] = int(len(g))
        row["n_seeds"] = int(g["seed"].nunique()) if "seed" in g else 0
        row["failed_count"] = int(pd.to_numeric(g.get("failed_flag", 0), errors="coerce").fillna(0).sum())
        for metric in metrics:
            if metric in g:
                m, s, n = mean_std(g[metric])
                row[f"{metric}_mean"] = m
                row[f"{metric}_std"] = s
                row[f"{metric}_n"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(out_dir: Path, tables: Dict[str, pd.DataFrame], run_records: pd.DataFrame, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    run_records.to_csv(out_dir / "live_run_records.csv", index=False, encoding="utf-8-sig")
    failed = run_records[run_records["returncode"] != 0].copy() if not run_records.empty else pd.DataFrame()
    failed.to_csv(out_dir / "failed_runs.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_dir / "fig6_panel_b_source_data.xlsx", engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        run_records.to_excel(writer, sheet_name="live_run_records", index=False)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "llm": args.llm,
        "python": args.python,
        "source_runs": 21,
        "datasets": {
            "classification": CLASSIFICATION_DATASETS,
            "regression": REGRESSION_DATASETS,
            "clustering": CLUSTERING_DATASETS,
        },
        "seeds": SEEDS,
        "iterations": {
            "classification": {"exam": 1, "feat": 1, "model": args.class_model_iterations, "param": 1},
            "regression": {"exam": 1, "feat": 1, "model": args.reg_model_iterations, "param": 1},
            "clustering": {"exam": 1, "feat": 1, "model": args.cluster_model_iterations, "param": 1},
        },
        "clustering_policy": {
            "cached_labels": True,
            "top_k": 1,
            "linear_voting_configs": "not executed by this runner; best-single source records only",
            "quadratic_sample_operations": "disabled",
        },
        "budget_note": "Used gpt-4o-mini and 21 reduced source runs; API key is not stored.",
    }
    (out_dir / "configs_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = f"""# Fig. 6b compare live reduced run

Generated: {manifest['generated_at']}

This directory contains a reduced live run using `autoLOGIC_compare` panel-b source executions.
It uses 21 source runs: 2 classification datasets, 2 regression datasets, 3 clustering datasets, and seeds 42/44/46.

The clustering source path uses cached labels with `top_k=1`; sample-pair matrix consensus is disabled.

Files:
- `classification_per_seed.csv`, `classification_summary.csv`, `classification_reliability_bins.csv`
- `regression_per_seed.csv`, `regression_summary.csv`, `regression_interval_curve.csv`
- `clustering_per_seed.csv`, `clustering_summary.csv`
- `live_run_records.csv`, `failed_runs.csv`
- `fig6_panel_b_source_data.xlsx`
- `configs_manifest.json`
"""
    (out_dir / "run_readme.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_fig6_panel_b_compare_live_reduced_21_20260514")
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--llm", default=LLM)
    parser.add_argument("--class-model-iterations", type=int, default=3)
    parser.add_argument("--reg-model-iterations", type=int, default=2)
    parser.add_argument("--cluster-model-iterations", type=int, default=5)
    parser.add_argument("--timeout-s", type=int, default=1800)
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[2]
    out_dir = Path(args.output).expanduser()
    if not out_dir.is_absolute():
        out_dir = workspace / out_dir
    raw_dir = out_dir / "raw"
    env = env_for_run()
    if not env.get("OPENAI_BASE_URL") or not env.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY must be set in the environment.")

    class_rows: List[dict] = []
    class_bins: List[dict] = []
    reg_rows: List[dict] = []
    reg_curve: List[dict] = []
    cluster_rows: List[dict] = []
    run_records: List[dict] = []

    class_script = workspace / "autologic" / "ensemble" / "classification_ensemble" / "classification_auto_ensemble_panelb_live.py"
    reg_script = workspace / "autologic" / "ensemble" / "regression_ensemble" / "regression_auto_ensemble_panelb_live.py"
    cluster_script = workspace / "autologic" / "ensemble" / "cluster_ensemble" / "cluster_auto_ensemble_panelb_linear.py"

    jobs = []
    for dataset in CLASSIFICATION_DATASETS:
        for seed in SEEDS:
            jobs.append(("classification", dataset, seed, class_script, workspace / "autologic/result/logs/classification", [
                args.python, "-u", str(class_script), "--llm", args.llm, "--dataset", dataset,
                "--default_seed", str(seed), "--exam_iterations", "1", "--feat_iterations", "1",
                "--model_iterations", str(args.class_model_iterations), "--param_iterations", "1",
            ]))
    for dataset in REGRESSION_DATASETS:
        for seed in SEEDS:
            jobs.append(("regression", dataset, seed, reg_script, workspace / "autologic/result/logs/regression", [
                args.python, "-u", str(reg_script), "--llm", args.llm, "--dataset", dataset,
                "--default_seed", str(seed), "--exam_iterations", "1", "--feat_iterations", "1",
                "--model_iterations", str(args.reg_model_iterations), "--param_iterations", "1",
            ]))
    for dataset in CLUSTERING_DATASETS:
        for seed in SEEDS:
            jobs.append(("clustering", dataset, seed, cluster_script, workspace / "autologic/result/logs/clustering", [
                args.python, "-u", str(cluster_script), "--llm", args.llm, "--dataset", dataset,
                "--default_seed", str(seed), "--exam_iterations", "1", "--feat_iterations", "1",
                "--model_iterations", str(args.cluster_model_iterations), "--param_iterations", "1",
                "--top_k", "1", "--min_ensemble_ari", "0.30",
            ]))

    for idx, (task, dataset, seed, script, log_dir, cmd) in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {task} dataset={dataset} seed={seed}", flush=True)
        before = {p.name for p in log_dir.glob(f"{dataset}_*.jsonl")} if log_dir.exists() else set()
        stdout = raw_dir / task / f"{dataset}_seed{seed}.stdout.log"
        stderr = raw_dir / task / f"{dataset}_seed{seed}.stderr.log"
        start = time.time()
        rc = run_process(cmd, workspace, stdout, stderr, env, args.timeout_s)
        seconds = time.time() - start
        jsonl = latest_jsonl(log_dir, dataset, before)
        run_records.append({
            "task": task,
            "dataset": dataset,
            "seed": seed,
            "returncode": rc,
            "seconds": seconds,
            "jsonl_path": str(jsonl) if jsonl else "",
            "stdout_path": str(stdout),
            "stderr_path": str(stderr),
        })
        if jsonl:
            if task == "classification":
                rows, bins = parse_classification(jsonl, dataset, seed)
                class_rows.extend(rows)
                class_bins.extend(bins)
            elif task == "regression":
                rows, curve = parse_regression(jsonl, dataset, seed)
                reg_rows.extend(rows)
                reg_curve.extend(curve)
            elif task == "clustering":
                cluster_rows.extend(parse_clustering(jsonl, dataset, seed))
        write_outputs(
            out_dir,
            {
                "classification_per_seed": pd.DataFrame(class_rows),
                "classification_summary": aggregate(pd.DataFrame(class_rows), ["dataset", "config", "model_family", "calibration_method"], ["auc", "acc", "ece", "brier", "nll"]),
                "classification_reliability_bins": pd.DataFrame(class_bins),
                "regression_per_seed": pd.DataFrame(reg_rows),
                "regression_summary": aggregate(pd.DataFrame(reg_rows), ["dataset", "config", "model_family", "interval_method", "alpha"], ["mae", "rmse", "r2", "picp", "mpiw", "pinaw", "coverage_error"]),
                "regression_interval_curve": pd.DataFrame(reg_curve),
                "clustering_per_seed": pd.DataFrame(cluster_rows),
                "clustering_summary": aggregate(pd.DataFrame(cluster_rows), ["dataset", "config", "model_family", "postprocessing_method"], ["ari", "nmi", "selected_k", "silhouette", "db", "ch"]),
            },
            pd.DataFrame(run_records),
            args,
        )
    print(f"[done] output={out_dir}", flush=True)


if __name__ == "__main__":
    main()
