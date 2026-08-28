from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
LIVE_CLUSTER_RUNNER = HERE / "run_fig6b_live_clustering.py"


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(x) for x in parse_csv(value)]


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_csv_append(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="", encoding="utf-8") as fh:
            fields = next(csv.reader(fh))
        fields = list(dict.fromkeys(fields + list(row.keys())))
    else:
        fields = list(row.keys())
    if exists:
        old = pd.read_csv(path)
        for col in fields:
            if col not in old.columns:
                old[col] = ""
        new = pd.DataFrame([row])
        for col in fields:
            if col not in new.columns:
                new[col] = ""
        pd.concat([old[fields], new[fields]], ignore_index=True).to_csv(path, index=False)
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def combine_outputs(output_dir: Path) -> Dict[str, pd.DataFrame]:
    raw_dir = output_dir / "raw"
    names = [
        "clustering_per_seed",
        "clustering_summary",
        "clustering_vs_comparison",
        "clustering_instability",
        "llm_generation_records",
        "clustering_assignment_records",
        "clustering_assignment_fidelity",
    ]
    tables: Dict[str, pd.DataFrame] = {}
    for name in names:
        parts = []
        for path in raw_dir.glob(f"*/{name}.csv"):
            df = read_csv(path)
            if df.empty:
                continue
            df["source_run_dir"] = str(path.parent)
            parts.append(df)
        tables[name] = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        tables[name].to_csv(output_dir / f"{name}.csv", index=False)
    return tables


def numeric_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.mean()) if vals.notna().any() else float("nan")


def write_summary(output_dir: Path, args: argparse.Namespace, tables: Dict[str, pd.DataFrame]) -> None:
    per_seed = tables.get("clustering_per_seed", pd.DataFrame())
    assign_fid = tables.get("clustering_assignment_fidelity", pd.DataFrame())
    llm = tables.get("llm_generation_records", pd.DataFrame())
    cmd = read_csv(output_dir / "batch_command_records.csv")

    dataset_rows = []
    if not per_seed.empty:
        success = per_seed[pd.to_numeric(per_seed.get("failed_flag"), errors="coerce").fillna(1).eq(0)]
        for dataset, group in success.groupby("dataset", sort=True):
            students = group[group.get("model_family").eq("distilled_student")] if "model_family" in group else pd.DataFrame()
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "n_rows": int(len(group)),
                    "n_success_student_rows": int(len(students)),
                    "ari_mean": numeric_mean(group.get("ari")),
                    "nmi_mean": numeric_mean(group.get("nmi")),
                    "teacher_student_ari_mean": numeric_mean(students.get("teacher_student_ari")) if not students.empty else float("nan"),
                    "teacher_student_nmi_mean": numeric_mean(students.get("teacher_student_nmi")) if not students.empty else float("nan"),
                    "knn_overlap_mean": numeric_mean(students.get("knn_overlap")) if not students.empty else float("nan"),
                    "retention_ratio_mean": numeric_mean(students.get("retention_ratio")) if not students.empty else float("nan"),
                }
            )
    dataset_summary = pd.DataFrame(dataset_rows)
    dataset_summary.to_csv(output_dir / "clustering_dataset_summary.csv", index=False)

    split_rows = []
    if not assign_fid.empty:
        for key, group in assign_fid.groupby(["dataset", "split"], dropna=False, sort=True):
            dataset, split = key
            split_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "n_rows": int(len(group)),
                    "assignment_ari_mean": numeric_mean(group.get("assignment_ari")),
                    "assignment_nmi_mean": numeric_mean(group.get("assignment_nmi")),
                    "rand_index_mean": numeric_mean(group.get("rand_index")),
                    "knn_overlap_mean": numeric_mean(group.get("knn_overlap")),
                }
            )
    split_summary = pd.DataFrame(split_rows)
    split_summary.to_csv(output_dir / "clustering_assignment_split_summary.csv", index=False)
    selected, selected_summary = selected_by_train_fidelity(
        tables.get("clustering_assignment_records", pd.DataFrame()),
        tables.get("clustering_assignment_fidelity", pd.DataFrame()),
    )
    selected.to_csv(output_dir / "clustering_selected_by_train_fidelity.csv", index=False)
    selected_summary.to_csv(output_dir / "clustering_selected_by_train_fidelity_summary.csv", index=False)

    total_tokens = float(pd.to_numeric(llm.get("total_tokens"), errors="coerce").fillna(0).sum()) if not llm.empty else 0.0
    max_call_tokens = float(pd.to_numeric(llm.get("total_tokens"), errors="coerce").max()) if not llm.empty else float("nan")
    token_abnormal = bool(total_tokens > args.max_total_tokens or (np.isfinite(max_call_tokens) and max_call_tokens > args.max_tokens_per_call))
    token_by_run = pd.DataFrame()
    if not llm.empty and {"dataset", "seed", "total_tokens"}.issubset(llm.columns):
        tmp = llm.copy()
        tmp["total_tokens_num"] = pd.to_numeric(tmp["total_tokens"], errors="coerce").fillna(0)
        token_by_run = (
            tmp.groupby(["dataset", "seed"], as_index=False)
            .agg(
                llm_calls=("total_tokens_num", "size"),
                llm_total_tokens=("total_tokens_num", "sum"),
                llm_max_call_tokens=("total_tokens_num", "max"),
            )
            .sort_values(["dataset", "seed"], kind="stable")
        )
        token_by_run["token_abnormal"] = (
            (token_by_run["llm_total_tokens"] > args.max_tokens_per_dataset_seed)
            | (token_by_run["llm_max_call_tokens"] > args.max_tokens_per_call)
        )
    token_by_run.to_csv(output_dir / "llm_token_by_run.csv", index=False)
    status_counts = cmd["status"].value_counts().to_dict() if not cmd.empty and "status" in cmd else {}
    lines = [
        "# Panel d clustering batch summary",
        "",
        f"Datasets: {args.datasets}",
        f"Seeds: {args.seeds}",
        f"LLM: {args.llm_model}",
        f"Iterations: feature={args.feature_iterations}, model={args.model_iterations}, param={args.param_iterations}",
        f"Per-run timeout seconds: {args.timeout_seconds}",
        f"Total LLM tokens recorded: {total_tokens:.0f}",
        f"Max tokens in one call: {max_call_tokens if np.isfinite(max_call_tokens) else ''}",
        f"Token abnormal: {token_abnormal}",
        f"Command status counts: {status_counts}",
        "",
        "Token by dataset-seed:",
        "",
        token_by_run.to_csv(index=False) if not token_by_run.empty else "(empty)",
        "",
        "Dataset summary:",
        "",
        dataset_summary.to_csv(index=False) if not dataset_summary.empty else "(empty)",
        "",
        "Assignment split summary:",
        "",
        split_summary.to_csv(index=False) if not split_summary.empty else "(empty)",
        "",
        "Selected delivery route by train-side fidelity:",
        "",
        selected_summary.to_csv(index=False) if not selected_summary.empty else "(empty)",
        "",
        "Files:",
        "",
        "- clustering_per_seed.csv",
        "- clustering_assignment_records.csv",
        "- clustering_assignment_fidelity.csv",
        "- clustering_dataset_summary.csv",
        "- clustering_assignment_split_summary.csv",
        "- clustering_selected_by_train_fidelity.csv",
        "- clustering_selected_by_train_fidelity_summary.csv",
        "- llm_generation_records.csv",
        "- llm_token_by_run.csv",
        "- batch_command_records.csv",
        "- batch_status.jsonl",
    ]
    (output_dir / "RUN_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def build_cmd(args: argparse.Namespace, run_output: Path, dataset: str, seed: int) -> List[str]:
    return [
        args.python,
        str(LIVE_CLUSTER_RUNNER),
        "--output",
        str(run_output),
        "--datasets",
        dataset,
        "--seeds",
        str(seed),
        "--llm-model",
        args.llm_model,
        "--feature-iterations",
        str(args.feature_iterations),
        "--model-iterations",
        str(args.model_iterations),
        "--param-iterations",
        str(args.param_iterations),
        "--distill-epochs",
        str(args.distill_epochs),
    ]


def selected_by_train_fidelity(assignments: pd.DataFrame, assignment_fidelity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if assignments.empty or assignment_fidelity.empty:
        return pd.DataFrame(), pd.DataFrame()
    student = assignments[assignments.get("model_family").eq("distilled_student")].copy()
    rows = []
    for key, group in student.groupby(["dataset", "seed", "config", "feature_setting"], sort=True):
        dataset, seed, config, feature_setting = key
        for split in ["train_or_fit", "panel_d_test", "all"]:
            sub = group if split == "all" else group[group.get("split").eq(split)]
            if sub.empty:
                continue
            teacher = pd.to_numeric(sub.get("teacher_cluster_label"), errors="coerce")
            student_labels = pd.to_numeric(sub.get("student_cluster_label"), errors="coerce")
            valid = teacher.notna() & student_labels.notna()
            if int(valid.sum()) < 2:
                continue
            t = teacher[valid].astype(int).to_numpy()
            s = student_labels[valid].astype(int).to_numpy()
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "config": config,
                    "feature_setting": feature_setting,
                    "split": split,
                    "n_samples": int(valid.sum()),
                    "assignment_ari": float(adjusted_rand_score(t, s)),
                    "assignment_nmi": float(normalized_mutual_info_score(t, s)),
                    "rand_index": float(np.mean((t[:, None] == t[None, :]) == (s[:, None] == s[None, :]))),
                }
            )
    train_metrics = pd.DataFrame(rows)
    if train_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    train = train_metrics[train_metrics["split"].eq("train_or_fit")].sort_values(
        ["dataset", "seed", "assignment_ari", "assignment_nmi", "rand_index"],
        ascending=[True, True, False, False, False],
    )
    best = train.groupby(["dataset", "seed"], as_index=False).head(1)[
        ["dataset", "seed", "config", "feature_setting", "assignment_ari", "assignment_nmi", "rand_index"]
    ].rename(
        columns={
            "assignment_ari": "selection_train_assignment_ari",
            "assignment_nmi": "selection_train_assignment_nmi",
            "rand_index": "selection_train_rand_index",
        }
    )
    test = assignment_fidelity[assignment_fidelity.get("split").eq("panel_d_test")][
        ["dataset", "seed", "config", "feature_setting", "assignment_ari", "assignment_nmi", "rand_index", "knn_overlap"]
    ].rename(
        columns={
            "assignment_ari": "test_assignment_ari",
            "assignment_nmi": "test_assignment_nmi",
            "rand_index": "test_rand_index",
            "knn_overlap": "test_knn_overlap",
        }
    )
    selected = best.merge(test, on=["dataset", "seed", "config", "feature_setting"], how="left")
    summary = (
        selected.groupby("dataset", as_index=False)
        .agg(
            n=("seed", "count"),
            test_assignment_ari_mean=("test_assignment_ari", "mean"),
            test_assignment_nmi_mean=("test_assignment_nmi", "mean"),
            test_rand_index_mean=("test_rand_index", "mean"),
            test_knn_overlap_mean=("test_knn_overlap", "mean"),
            test_knn_overlap_min=("test_knn_overlap", "min"),
            selection_train_assignment_ari_mean=("selection_train_assignment_ari", "mean"),
        )
        .sort_values("dataset", kind="stable")
    )
    return selected, summary


def run_one(args: argparse.Namespace, output_dir: Path, dataset: str, seed: int) -> None:
    tag = f"{dataset}_seed{seed}"
    run_output = output_dir / "raw" / tag
    logs = output_dir / "command_logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{tag}.stdout.log"
    stderr_path = logs / f"{tag}.stderr.log"
    cmd = build_cmd(args, run_output, dataset, seed)
    append_jsonl(output_dir / "batch_status.jsonl", {"event": "start", "dataset": dataset, "seed": seed, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    start = time.time()
    try:
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_f, stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_f:
            proc = subprocess.run(
                cmd,
                cwd=str(WORKSPACE),
                stdout=stdout_f,
                stderr=stderr_f,
                timeout=args.timeout_seconds,
                env=os.environ.copy(),
                check=False,
            )
        status = "success" if proc.returncode == 0 else "failed"
        return_code: Any = proc.returncode
        error = ""
    except subprocess.TimeoutExpired:
        status = "timeout"
        return_code = ""
        error = f"timeout after {args.timeout_seconds}s"
    elapsed = time.time() - start
    row = {
        "dataset": dataset,
        "seed": seed,
        "status": status,
        "return_code": return_code,
        "seconds": elapsed,
        "timeout_seconds": args.timeout_seconds,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "run_output": str(run_output),
        "error": error,
        "command_without_secrets": " ".join(cmd),
    }
    write_csv_append(output_dir / "batch_command_records.csv", row)
    append_jsonl(output_dir / "batch_status.jsonl", {"event": status, "dataset": dataset, "seed": seed, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": elapsed, "error": error})
    tables = combine_outputs(output_dir)
    write_summary(output_dir, args, tables)


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.getenv("OPENAI_BASE_URL") or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY must be set in the runtime environment.")
    manifest = {
        "python": args.python,
        "live_cluster_runner": str(LIVE_CLUSTER_RUNNER),
        "code_family": "autoLOGIC_new",
        "datasets": args.datasets,
        "seeds": args.seeds,
        "llm_model": args.llm_model,
        "feature_iterations": args.feature_iterations,
        "model_iterations": args.model_iterations,
        "param_iterations": args.param_iterations,
        "enable_optimization": args.param_iterations > 0,
        "enable_feedback": True,
        "timeout_seconds": args.timeout_seconds,
        "max_total_tokens": args.max_total_tokens,
        "max_tokens_per_dataset_seed": args.max_tokens_per_dataset_seed,
        "max_tokens_per_call": args.max_tokens_per_call,
        "api_base_url_from_env": bool(os.getenv("OPENAI_BASE_URL")),
        "api_key_from_env": bool(os.getenv("OPENAI_API_KEY")),
        "note": "API keys are not written by this batch runner.",
    }
    (output_dir / "configs_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for dataset in args.datasets:
        for seed in args.seeds:
            run_one(args, output_dir, dataset, int(seed))
    tables = combine_outputs(output_dir)
    write_summary(output_dir, args, tables)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process-level batch wrapper for panel d clustering live runs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", type=parse_csv, default=["breast", "glass", "students"])
    parser.add_argument("--seeds", type=parse_int_csv, default=[42, 44, 46])
    parser.add_argument("--llm-model", default="gpt-3.5-turbo")
    parser.add_argument("--feature-iterations", type=int, default=10)
    parser.add_argument("--model-iterations", type=int, default=10)
    parser.add_argument("--param-iterations", type=int, default=5)
    parser.add_argument("--distill-epochs", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--max-total-tokens", type=int, default=2000000)
    parser.add_argument("--max-tokens-per-dataset-seed", type=int, default=500000)
    parser.add_argument("--max-tokens-per-call", type=int, default=30000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = run(args)
    print(f"[done] panel d clustering batch written to {out}", flush=True)


if __name__ == "__main__":
    main()
