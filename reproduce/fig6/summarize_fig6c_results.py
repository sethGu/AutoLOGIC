from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_COLS = [
    "teacher_auc",
    "student_auc",
    "auc_delta_student_minus_teacher",
    "fidelity_prob_pearson",
    "fidelity_prob_mae",
    "fidelity_prob_js",
    "fidelity_top10_overlap",
    "teacher_latency_ms_per_sample",
    "student_latency_ms_per_sample",
    "latency_ratio_student_over_teacher",
    "teacher_artifact_bytes",
    "student_artifact_bytes",
    "artifact_size_ratio_student_over_teacher",
    "llm_calls",
    "llm_total_tokens",
]

REG_COLS = [
    "teacher_rmse",
    "student_rmse",
    "rmse_delta_student_minus_teacher",
    "teacher_r2",
    "student_r2",
    "r2_delta_student_minus_teacher",
    "fidelity_pred_pearson",
    "fidelity_pred_nrmse_y_std",
    "teacher_student_top10_pred_overlap",
    "teacher_latency_ms_per_sample",
    "student_latency_ms_per_sample",
    "latency_ratio_student_over_teacher",
    "teacher_artifact_bytes",
    "student_artifact_bytes",
    "artifact_size_ratio_student_over_teacher",
    "llm_calls",
    "llm_total_tokens",
]


def mean_sd(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return ""
    if len(values) == 1:
        return f"{values.iloc[0]:.4g}"
    return f"{values.mean():.4g} +/- {values.std(ddof=1):.4g}"


def summarize(metrics: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in metrics.columns]
    rows = []
    for (task, dataset), group in metrics.groupby(["task", "dataset"], dropna=False):
        row = {
            "task": task,
            "dataset": dataset,
            "n_successful_seeds": int(group["seed"].nunique()),
        }
        for col in keep:
            row[col] = mean_sd(group[col])
        rows.append(row)
    return pd.DataFrame(rows)


def build_panel_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        if row["task"] == "classification":
            rows.append(
                {
                    "task": row["task"],
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "teacher_metric": row.get("teacher_auc", np.nan),
                    "student_metric": row.get("student_auc", np.nan),
                    "student_minus_teacher": row.get("auc_delta_student_minus_teacher", np.nan),
                    "fidelity_primary": row.get("fidelity_prob_pearson", np.nan),
                    "fidelity_error": row.get("fidelity_prob_mae", np.nan),
                    "latency_ratio_student_over_teacher": row.get("latency_ratio_student_over_teacher", np.nan),
                    "artifact_size_ratio_student_over_teacher": row.get("artifact_size_ratio_student_over_teacher", np.nan),
                    "llm_total_tokens": row.get("llm_total_tokens", np.nan),
                }
            )
        elif row["task"] == "regression":
            rows.append(
                {
                    "task": row["task"],
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "teacher_metric": row.get("teacher_rmse", np.nan),
                    "student_metric": row.get("student_rmse", np.nan),
                    "student_minus_teacher": row.get("rmse_delta_student_minus_teacher", np.nan),
                    "fidelity_primary": row.get("fidelity_pred_pearson", np.nan),
                    "fidelity_error": row.get("fidelity_pred_nrmse_y_std", np.nan),
                    "latency_ratio_student_over_teacher": row.get("latency_ratio_student_over_teacher", np.nan),
                    "artifact_size_ratio_student_over_teacher": row.get("artifact_size_ratio_student_over_teacher", np.nan),
                    "llm_total_tokens": row.get("llm_total_tokens", np.nan),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize panel c delivery-fidelity outputs.")
    parser.add_argument("--input", required=True, help="Directory containing per_run_metrics.csv.")
    args = parser.parse_args()
    out_dir = Path(args.input).resolve()
    metrics_path = out_dir / "per_run_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["status"].astype(str).str.lower().eq("success")].copy()
    if metrics.empty:
        raise RuntimeError("No successful runs in per_run_metrics.csv.")

    class_summary = summarize(metrics[metrics["task"].eq("classification")], CLASS_COLS)
    reg_summary = summarize(metrics[metrics["task"].eq("regression")], REG_COLS)
    panel_table = build_panel_table(metrics)

    if not class_summary.empty:
        class_summary.to_csv(out_dir / "summary_classification.csv", index=False)
    if not reg_summary.empty:
        reg_summary.to_csv(out_dir / "summary_regression.csv", index=False)
    panel_table.to_csv(out_dir / "panel_c_candidate_table.csv", index=False)


if __name__ == "__main__":
    main()
