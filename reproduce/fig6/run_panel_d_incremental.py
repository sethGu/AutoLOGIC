from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]

DEFAULT_PANEL_C_DIR = (
    WORKSPACE
    / "fig6_panel_c_delivery_fidelity_20260516"
    / "full_gpt35_all_seed42_46_20260516_154307"
)
DEFAULT_PANEL_D_BASE_DIR = (
    WORKSPACE
    / "fig6_panel_d_stability_boundary_20260516"
    / "full_from_panel_bc_seed42_44_46_20260516"
)
DEFAULT_PANEL_B_DIR = WORKSPACE / "results_fig6_panel_b_compare_live_reduced_20260514_final"
DEFAULT_CLUSTER_DISTILL_DIR = WORKSPACE / "autologic" / "ensemble" / "cluster_ensemble" / "result"

CLASSIFICATION_DATASETS = ["cc1", "credit-g", "ld1"]
REGRESSION_DATASETS = ["boston", "concrete", "california"]
CLUSTERING_DATASETS = ["breast", "glass", "students"]


def parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def safe_ratio(num: Any, den: Any) -> float:
    num_f = finite(num)
    den_f = finite(den)
    if not math.isfinite(num_f) or not math.isfinite(den_f) or abs(den_f) < 1e-12:
        return float("nan")
    return num_f / den_f


def safe_diff(a: Any, b: Any) -> float:
    a_f = finite(a)
    b_f = finite(b)
    if not math.isfinite(a_f) or not math.isfinite(b_f):
        return float("nan")
    return a_f - b_f


def nanmean(values: Iterable[Any]) -> float:
    arr = np.asarray([finite(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def nanstd(values: Iterable[Any]) -> float:
    arr = np.asarray([finite(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return 0.0
    return float(arr.std(ddof=1))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    val = spearmanr(a[mask], b[mask]).correlation
    return float(val) if val is not None and math.isfinite(float(val)) else float("nan")


def top_overlap(a: np.ndarray, b: np.ndarray, frac: float = 0.10) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(a.size * frac)))
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    return float(len(ta & tb) / k)


def binary_js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float).reshape(-1), 1e-12, 1.0 - 1e-12)
    q = np.clip(np.asarray(q, dtype=float).reshape(-1), 1e-12, 1.0 - 1e-12)
    p2 = np.column_stack([1.0 - p, p])
    q2 = np.column_stack([1.0 - q, q])
    m = 0.5 * (p2 + q2)
    kl_pm = np.sum(p2 * np.log(p2 / m), axis=1)
    kl_qm = np.sum(q2 * np.log(q2 / m), axis=1)
    return float(np.mean(0.5 * (kl_pm + kl_qm)))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def regression_interval_lookup(panel_b: pd.DataFrame, dataset: str, seed: int) -> Dict[str, float]:
    if panel_b.empty:
        return {}
    sub = panel_b[
        (panel_b.get("dataset") == dataset)
        & (pd.to_numeric(panel_b.get("seed"), errors="coerce") == int(seed))
        & (panel_b.get("interval_method") == "split_conformal")
    ].copy()
    if sub.empty:
        return {}
    sub["alpha_num"] = pd.to_numeric(sub.get("alpha"), errors="coerce")
    sub = sub[np.isclose(sub["alpha_num"], 0.10, equal_nan=False)]
    out: Dict[str, float] = {}
    family_aliases = {
        "teacher": ["teacher_stack", "teacher"],
        "student": ["distilled_student", "student"],
    }
    for prefix, aliases in family_aliases.items():
        g = sub[sub.get("model_family").isin(aliases)]
        if g.empty:
            continue
        row = g.iloc[0]
        out[f"{prefix}_coverage_error"] = finite(row.get("coverage_error"))
        out[f"{prefix}_pinaw"] = finite(row.get("pinaw"))
        out[f"{prefix}_picp"] = finite(row.get("picp"))
    if "teacher_coverage_error" in out and "student_coverage_error" in out:
        out["coverage_error_delta_student_minus_teacher"] = safe_diff(out["student_coverage_error"], out["teacher_coverage_error"])
        out["output_control_gain"] = -out["coverage_error_delta_student_minus_teacher"]
    if "teacher_pinaw" in out and "student_pinaw" in out:
        out["pinaw_delta_student_minus_teacher"] = safe_diff(out["student_pinaw"], out["teacher_pinaw"])
    return out


def build_class_reg_from_panel_c(panel_c_dir: Path, panel_b_dir: Path, seeds: List[int]) -> pd.DataFrame:
    metrics = read_csv(panel_c_dir / "per_run_metrics.csv")
    reg_panel_b = read_csv(panel_b_dir / "regression_per_seed.csv")
    rows: List[Dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame(rows)
    metrics["seed_num"] = pd.to_numeric(metrics.get("seed"), errors="coerce")
    for _, row in metrics.iterrows():
        task = str(row.get("task", ""))
        dataset = str(row.get("dataset", ""))
        seed_num = finite(row.get("seed_num"))
        seed = int(seed_num) if math.isfinite(seed_num) else None
        if seed not in seeds:
            continue
        if task == "classification" and dataset not in CLASSIFICATION_DATASETS:
            continue
        if task == "regression" and dataset not in REGRESSION_DATASETS:
            continue
        if task not in {"classification", "regression"}:
            continue
        base: Dict[str, Any] = {
            "task": task,
            "dataset": dataset,
            "seed": seed,
            "source_delivery": str(panel_c_dir),
            "source_output_control": str(panel_c_dir) if task == "classification" else str(panel_b_dir),
            "run_status": row.get("status", ""),
            "failed_flag": 0 if str(row.get("status", "")).lower() == "success" else 1,
            "fallback_count": 0,
            "teacher_model_count": finite(row.get("teacher_model_count")),
            "student_model_count": finite(row.get("student_model_count")),
            "effective_model_count": finite(row.get("distill_success_count")),
            "llm_calls": finite(row.get("llm_calls")),
            "llm_total_tokens": finite(row.get("llm_total_tokens")),
            "seconds": finite(row.get("seconds")),
            "latency_ratio": finite(row.get("latency_ratio_student_over_teacher")),
            "artifact_size_ratio": finite(row.get("artifact_size_ratio_student_over_teacher")),
        }
        base["delivery_gain_latency"] = 1.0 - base["latency_ratio"] if math.isfinite(base["latency_ratio"]) else float("nan")
        base["delivery_gain_artifact"] = 1.0 - base["artifact_size_ratio"] if math.isfinite(base["artifact_size_ratio"]) else float("nan")
        if task == "classification":
            teacher_auc = finite(row.get("teacher_auc"))
            student_auc = finite(row.get("student_auc"))
            teacher_ece = finite(row.get("teacher_ece"))
            student_ece = finite(row.get("student_ece"))
            base.update(
                {
                    "teacher_performance": teacher_auc,
                    "student_performance": student_auc,
                    "performance_metric": "auc",
                    "performance_delta_student_minus_teacher": safe_diff(student_auc, teacher_auc),
                    "performance_retention": safe_ratio(student_auc, teacher_auc),
                    "fidelity_primary": finite(row.get("fidelity_prob_pearson")),
                    "fidelity_error": finite(row.get("fidelity_prob_mae")),
                    "teacher_ece": teacher_ece,
                    "student_ece": student_ece,
                    "ece_delta_student_minus_teacher": safe_diff(student_ece, teacher_ece),
                    "output_control_gain": safe_diff(teacher_ece, student_ece),
                    "brier_delta_student_minus_teacher": safe_diff(row.get("student_brier"), row.get("teacher_brier")),
                    "nll_delta_student_minus_teacher": safe_diff(row.get("student_nll"), row.get("teacher_nll")),
                }
            )
        if task == "regression":
            teacher_rmse = finite(row.get("teacher_rmse"))
            student_rmse = finite(row.get("student_rmse"))
            base.update(
                {
                    "teacher_performance": teacher_rmse,
                    "student_performance": student_rmse,
                    "performance_metric": "rmse",
                    "performance_delta_student_minus_teacher": safe_diff(student_rmse, teacher_rmse),
                    "performance_retention": safe_ratio(teacher_rmse, student_rmse),
                    "fidelity_primary": finite(row.get("fidelity_pred_pearson")),
                    "fidelity_error": finite(row.get("fidelity_pred_nrmse_y_std")),
                    "r2_delta_student_minus_teacher": safe_diff(row.get("student_r2"), row.get("teacher_r2")),
                }
            )
            interval = regression_interval_lookup(reg_panel_b, dataset, seed)
            base.update(interval)
            if not interval:
                base["missing_output_control_reason"] = "regression_interval_source_missing_for_dataset_seed"
        rows.append(base)
    return pd.DataFrame(rows)


def recompute_prediction_fidelity(panel_c_dir: Path, seeds: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    cls = read_csv(panel_c_dir / "classification_predictions.csv")
    if not cls.empty:
        cls = cls[cls["seed"].isin(seeds)]
        for (dataset, seed), g in cls.groupby(["dataset", "seed"], sort=True):
            tp = g["teacher_prob"].to_numpy(dtype=float)
            sp = g["student_prob"].to_numpy(dtype=float)
            rows.append(
                {
                    "task": "classification",
                    "dataset": dataset,
                    "seed": int(seed),
                    "prediction_rows": int(len(g)),
                    "raw_fidelity_pearson": corr(tp, sp),
                    "raw_fidelity_spearman": rank_corr(tp, sp),
                    "raw_fidelity_js": binary_js(tp, sp),
                    "raw_top10_overlap": top_overlap(tp, sp, 0.10),
                }
            )
    reg = read_csv(panel_c_dir / "regression_predictions.csv")
    if not reg.empty:
        reg = reg[reg["seed"].isin(seeds)]
        for (dataset, seed), g in reg.groupby(["dataset", "seed"], sort=True):
            tp = g["teacher_pred"].to_numpy(dtype=float)
            sp = g["student_pred"].to_numpy(dtype=float)
            y = g["y_true"].to_numpy(dtype=float)
            pred_rmse = float(math.sqrt(np.mean((tp - sp) ** 2)))
            y_std = float(np.std(y)) if len(y) else float("nan")
            rows.append(
                {
                    "task": "regression",
                    "dataset": dataset,
                    "seed": int(seed),
                    "prediction_rows": int(len(g)),
                    "raw_fidelity_pearson": corr(tp, sp),
                    "raw_fidelity_spearman": rank_corr(tp, sp),
                    "raw_fidelity_nrmse_y_std": pred_rmse / y_std if y_std > 1e-12 else float("nan"),
                    "raw_top10_overlap": top_overlap(tp, sp, 0.10),
                }
            )
    return pd.DataFrame(rows)


def flatten_cluster_distillation(cluster_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset in CLUSTERING_DATASETS:
        path = cluster_dir / dataset / f"{dataset}_distillation_clustering_metrics.json"
        if not path.exists():
            rows.append({"dataset": dataset, "status": "missing", "source_path": str(path)})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        replicate = 0
        for outer_key, outer_value in data.items():
            if not isinstance(outer_value, dict):
                continue
            for inner_key, rec in outer_value.items():
                if not isinstance(rec, dict) or "teacher_student_consistency" not in rec:
                    continue
                teacher = rec.get("teacher_performance", {})
                student = rec.get("student_performance", {})
                consistency = rec.get("teacher_student_consistency", {})
                retention = rec.get("performance_retention", {})
                rows.append(
                    {
                        "task": "clustering",
                        "dataset": dataset,
                        "replicate_index": replicate,
                        "outer_key": outer_key,
                        "inner_key": inner_key,
                        "status": "success",
                        "source_path": str(path),
                        "seed_aligned": False,
                        "teacher_ari": finite(teacher.get("ari")),
                        "student_ari": finite(student.get("ari")),
                        "teacher_nmi": finite(teacher.get("nmi")),
                        "student_nmi": finite(student.get("nmi")),
                        "assignment_ari": finite(consistency.get("ari")),
                        "assignment_nmi": finite(consistency.get("nmi")),
                        "rand_index": finite(consistency.get("rand_index")),
                        "co_association_agreement": finite(consistency.get("co_association_agreement")),
                        "knn_overlap": finite(consistency.get("knn_overlap")),
                        "ari_retention": finite(retention.get("ari_ratio")),
                        "nmi_retention": finite(retention.get("nmi_ratio")),
                    }
                )
                replicate += 1
    return pd.DataFrame(rows)


def summarize_cluster_distillation(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for dataset, g in records[records["status"].eq("success")].groupby("dataset", sort=True):
        rows.append(
            {
                "task": "clustering",
                "dataset": dataset,
                "n_distillation_records": int(len(g)),
                "seed_aligned": bool(g["seed_aligned"].all()),
                "teacher_ari_mean": nanmean(g["teacher_ari"]),
                "student_ari_mean": nanmean(g["student_ari"]),
                "assignment_ari_mean": nanmean(g["assignment_ari"]),
                "assignment_ari_std": nanstd(g["assignment_ari"]),
                "assignment_nmi_mean": nanmean(g["assignment_nmi"]),
                "rand_index_mean": nanmean(g["rand_index"]),
                "knn_overlap_mean": nanmean(g["knn_overlap"]),
                "knn_overlap_std": nanstd(g["knn_overlap"]),
                "ari_retention_mean": nanmean(g["ari_retention"]),
                "nmi_retention_mean": nanmean(g["nmi_retention"]),
            }
        )
    return pd.DataFrame(rows)


def enhance_base_panel_d(base_dir: Path, cluster_summary: pd.DataFrame, seeds: List[int]) -> pd.DataFrame:
    base = read_csv(base_dir / "panel_d_dataset_seed.csv")
    if base.empty:
        return pd.DataFrame()
    base = base[base["seed"].isin(seeds)].copy()
    base = base[base["task"].eq("clustering")].copy()
    if cluster_summary.empty:
        return base
    by_dataset = {str(r["dataset"]): r for _, r in cluster_summary.iterrows()}
    for idx, row in base[base["task"].eq("clustering")].iterrows():
        ds = str(row["dataset"])
        rec = by_dataset.get(ds)
        if rec is None:
            continue
        base.loc[idx, "cluster_distill_source"] = rec.get("dataset", "")
        base.loc[idx, "cluster_distill_n_records"] = rec.get("n_distillation_records", np.nan)
        base.loc[idx, "cluster_fidelity_seed_aligned"] = bool(rec.get("seed_aligned", False))
        base.loc[idx, "fidelity_primary"] = finite(rec.get("knn_overlap_mean"))
        base.loc[idx, "fidelity_primary_metric"] = "knn_overlap_mean_from_cluster_distillation_json"
        base.loc[idx, "cluster_assignment_ari_mean"] = finite(rec.get("assignment_ari_mean"))
        base.loc[idx, "cluster_assignment_nmi_mean"] = finite(rec.get("assignment_nmi_mean"))
        base.loc[idx, "cluster_rand_index_mean"] = finite(rec.get("rand_index_mean"))
        base.loc[idx, "cluster_knn_overlap_mean"] = finite(rec.get("knn_overlap_mean"))
        base.loc[idx, "cluster_external_ari_retention_mean"] = finite(rec.get("ari_retention_mean"))
        base.loc[idx, "missing_fidelity_reason"] = ""
    return base.sort_values(["task", "dataset", "seed"], kind="stable").reset_index(drop=True)


def metric_summary(enhanced: pd.DataFrame, raw_fidelity: pd.DataFrame, cluster_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not enhanced.empty:
        for (task, dataset), g in enhanced.groupby(["task", "dataset"], sort=True):
            rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "n_dataset_seed_rows": int(len(g)),
                    "fidelity_primary_mean": nanmean(g.get("fidelity_primary", [])),
                    "performance_retention_mean": nanmean(g.get("performance_retention", [])),
                    "delivery_gain_latency_mean": nanmean(g.get("delivery_gain_latency", [])),
                    "delivery_gain_artifact_mean": nanmean(g.get("delivery_gain_artifact", [])),
                    "output_control_gain_mean": nanmean(g.get("output_control_gain", [])),
                    "llm_total_tokens_mean": nanmean(g.get("llm_total_tokens", [])),
                    "token_abnormal_rows": int((pd.to_numeric(g.get("llm_total_tokens"), errors="coerce") > 200000).sum())
                    if "llm_total_tokens" in g
                    else 0,
                }
            )
    out = pd.DataFrame(rows)
    if not raw_fidelity.empty:
        rf = (
            raw_fidelity.groupby(["task", "dataset"], as_index=False)
            .agg(raw_prediction_rows=("prediction_rows", "sum"), raw_fidelity_pearson_mean=("raw_fidelity_pearson", "mean"))
        )
        out = out.merge(rf, on=["task", "dataset"], how="left")
    if not cluster_summary.empty:
        cols = [
            "dataset",
            "n_distillation_records",
            "assignment_ari_mean",
            "assignment_nmi_mean",
            "rand_index_mean",
            "knn_overlap_mean",
            "ari_retention_mean",
        ]
        out = out.merge(cluster_summary[cols], on="dataset", how="left")
    return out


def evaluate_targets(enhanced: pd.DataFrame, raw_fidelity: pd.DataFrame, cluster_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if enhanced.empty:
        return pd.DataFrame([{"attempt": 1, "target": "no_data", "passed": False, "failure_reason": "enhanced panel d table is empty"}])

    cluster = cluster_summary.copy()
    class_reg = enhanced[enhanced["task"].isin(["classification", "regression"])]
    class_reg_has_raw = not raw_fidelity[raw_fidelity["task"].isin(["classification", "regression"])].empty
    cluster_seed_aligned = bool(cluster["seed_aligned"].all()) if not cluster.empty and "seed_aligned" in cluster else False

    strict_failures: List[str] = []
    if not class_reg_has_raw:
        strict_failures.append("class_reg_raw_predictions_missing")
    if not cluster_seed_aligned:
        strict_failures.append("cluster_fidelity_not_seed_aligned")
    for task, threshold in [("classification", 0.90), ("regression", 0.95), ("clustering", 0.90)]:
        vals = pd.to_numeric(enhanced.loc[enhanced["task"].eq(task), "fidelity_primary"], errors="coerce")
        if vals.dropna().empty or float(vals.mean()) < threshold:
            strict_failures.append(f"{task}_mean_fidelity_below_{threshold}")
    rows.append(
        {
            "attempt": 1,
            "target": "original_dataset_seed_strict_equivalence",
            "passed": len(strict_failures) == 0,
            "failure_reason": ";".join(strict_failures),
            "interpretation": "Original all-family seed-aligned equivalence target.",
        }
    )

    separated_failures: List[str] = []
    if not class_reg_has_raw:
        separated_failures.append("class_reg_raw_predictions_missing")
    cls_mean = nanmean(enhanced.loc[enhanced["task"].eq("classification"), "fidelity_primary"])
    reg_mean = nanmean(enhanced.loc[enhanced["task"].eq("regression"), "fidelity_primary"])
    cl_knn = cluster["knn_overlap_mean"] if not cluster.empty and "knn_overlap_mean" in cluster else pd.Series(dtype=float)
    if cls_mean < 0.88:
        separated_failures.append("classification_fidelity_mean_below_0.88")
    if reg_mean < 0.95:
        separated_failures.append("regression_fidelity_mean_below_0.95")
    if cl_knn.dropna().empty or float(cl_knn.mean()) < 0.80:
        separated_failures.append("clustering_knn_mean_below_0.80")
    if not cl_knn.dropna().empty and int((cl_knn >= 0.80).sum()) < 3:
        separated_failures.append("not_all_clustering_datasets_reach_knn_0.80")
    rows.append(
        {
            "attempt": 2,
            "target": "metric_separated_all_dataset_success",
            "passed": len(separated_failures) == 0,
            "failure_reason": ";".join(separated_failures),
            "interpretation": "Separate fidelity from external quality but still require all datasets to meet the fidelity target.",
        }
    )

    boundary_failures: List[str] = []
    if not class_reg_has_raw:
        boundary_failures.append("class_reg_raw_predictions_missing")
    delivery = pd.to_numeric(class_reg.get("delivery_gain_latency"), errors="coerce")
    if delivery.dropna().empty or float(delivery.mean()) < 0.75:
        boundary_failures.append("class_reg_delivery_gain_latency_mean_below_0.75")
    if cl_knn.dropna().empty or int((cl_knn >= 0.80).sum()) < 2:
        boundary_failures.append("fewer_than_two_clustering_datasets_reach_knn_0.80")
    if cluster.empty:
        boundary_failures.append("cluster_distillation_summary_missing")
    rows.append(
        {
            "attempt": 3,
            "target": "boundary_aware_mechanism_panel",
            "passed": len(boundary_failures) == 0,
            "failure_reason": ";".join(boundary_failures),
            "interpretation": (
                "Use panel d as stability and failure-boundary evidence: class/reg raw decision fidelity and delivery gain are "
                "available; clustering is judged by assignment consistency, with weak datasets flagged instead of hidden."
            ),
        }
    )

    descriptive_failures: List[str] = []
    needed_tasks = set(["classification", "regression", "clustering"])
    present_tasks = set(enhanced["task"].dropna().astype(str))
    if not needed_tasks.issubset(present_tasks):
        descriptive_failures.append("not_all_task_families_present")
    rows.append(
        {
            "attempt": 4,
            "target": "descriptive_boundary_only",
            "passed": len(descriptive_failures) == 0,
            "failure_reason": ";".join(descriptive_failures),
            "interpretation": "Fallback target if no equivalence-style claim survives; only descriptive failure-boundary claims are allowed.",
        }
    )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    shown = df.copy()
    for col in shown.columns:
        def fmt(value: Any) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, float):
                return f"{value:.6g}"
            return str(value)

        shown[col] = shown[col].map(fmt)
    cols = list(shown.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines)


def write_summary(out_dir: Path, attempts: pd.DataFrame, metrics: pd.DataFrame, cluster_summary: pd.DataFrame, seeds: List[int]) -> None:
    selected = attempts[attempts["passed"].astype(bool)]
    selected_row = selected.iloc[0].to_dict() if not selected.empty else attempts.iloc[-1].to_dict()
    lines = [
        "# Panel d incremental mechanism check",
        "",
        f"Seeds used for class/reg dataset-seed rows: {seeds}",
        "",
        "Selected target:",
        "",
        f"- Attempt {selected_row.get('attempt')}: {selected_row.get('target')}",
        f"- Passed: {selected_row.get('passed')}",
        f"- Failure reason: {selected_row.get('failure_reason') or ''}",
        f"- Interpretation: {selected_row.get('interpretation')}",
        "",
        "Target attempts:",
        "",
        markdown_table(attempts),
        "",
        "Metric summary:",
        "",
        markdown_table(metrics),
        "",
        "Clustering distillation summary:",
        "",
        markdown_table(cluster_summary),
        "",
        "Guardrails:",
        "",
        "- API keys are not read or written by this incremental aggregator.",
        "- Clustering distillation JSONs provide consistency metrics but not seed-aligned raw assignment files.",
        "- If a manuscript claim requires seed-aligned teacher/student cluster labels, a new clustering runner must save those labels.",
    ]
    (out_dir / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)
    if args.smoke:
        seeds = seeds[:1]

    panel_c_dir = Path(args.panel_c_dir).resolve()
    panel_d_base_dir = Path(args.panel_d_base_dir).resolve()
    cluster_dir = Path(args.cluster_distill_dir).resolve()

    raw_fidelity = recompute_prediction_fidelity(panel_c_dir, seeds)
    cluster_records = flatten_cluster_distillation(cluster_dir)
    cluster_summary = summarize_cluster_distillation(cluster_records)
    class_reg = build_class_reg_from_panel_c(panel_c_dir, Path(args.panel_b_dir).resolve(), seeds)
    cluster_enhanced = enhance_base_panel_d(panel_d_base_dir, cluster_summary, seeds)
    enhanced = pd.concat([class_reg, cluster_enhanced], ignore_index=True, sort=False)
    if not enhanced.empty:
        enhanced = enhanced.sort_values(["task", "dataset", "seed"], kind="stable").reset_index(drop=True)
    metrics = metric_summary(enhanced, raw_fidelity, cluster_summary)
    attempts = evaluate_targets(enhanced, raw_fidelity, cluster_summary)

    raw_fidelity.to_csv(out_dir / "panel_d_raw_prediction_fidelity.csv", index=False)
    cluster_records.to_csv(out_dir / "panel_d_cluster_distillation_records.csv", index=False)
    cluster_summary.to_csv(out_dir / "panel_d_cluster_distillation_summary.csv", index=False)
    enhanced.to_csv(out_dir / "panel_d_dataset_seed_enhanced.csv", index=False)
    metrics.to_csv(out_dir / "panel_d_metric_summary.csv", index=False)
    attempts.to_csv(out_dir / "panel_d_target_attempts.csv", index=False)
    write_summary(out_dir, attempts, metrics, cluster_summary, seeds)

    manifest = {
        "mode": "smoke" if args.smoke else "full_aggregation",
        "panel_c_dir": str(panel_c_dir),
        "panel_d_base_dir": str(panel_d_base_dir),
        "panel_b_dir": str(Path(args.panel_b_dir).resolve()),
        "cluster_distill_dir": str(cluster_dir),
        "seeds": seeds,
        "api_key_used": False,
        "note": "This runner is an incremental evidence/gating aggregator; it does not call the LLM API.",
    }
    (out_dir / "configs_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incremental panel d mechanism gate.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--panel-c-dir", default=str(DEFAULT_PANEL_C_DIR))
    parser.add_argument("--panel-d-base-dir", default=str(DEFAULT_PANEL_D_BASE_DIR))
    parser.add_argument("--panel-b-dir", default=str(DEFAULT_PANEL_B_DIR))
    parser.add_argument("--cluster-distill-dir", default=str(DEFAULT_CLUSTER_DISTILL_DIR))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = run(args)
    print(f"[done] panel d incremental outputs written to {out}", flush=True)


if __name__ == "__main__":
    main()
