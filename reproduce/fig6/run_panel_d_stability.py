from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


CLASSIFICATION_DATASETS = ["cc1", "credit-g", "ld1"]
REGRESSION_DATASETS = ["boston", "concrete", "california"]
CLUSTERING_DATASETS = ["breast", "glass", "students"]

DEFAULT_SEEDS = [42, 44, 46]

DEFAULT_PANEL_C_DIR = (
    Path(__file__).resolve().parents[2]
    / "fig6_panel_c_delivery_fidelity_20260516"
    / "full_gpt35_all_seed42_46_20260516_154307"
)
DEFAULT_PANEL_B_DIR = (
    Path(__file__).resolve().parents[2]
    / "results_fig6_panel_b_compare_live_reduced_20260514_final"
)

THRESHOLDS = {
    "classification": {
        "fidelity_primary_min": 0.90,
        "performance_retention_min": 0.98,
        "ece_worsening_max": 0.02,
        "latency_ratio_max": 0.25,
        "artifact_ratio_max": 0.25,
    },
    "regression": {
        "fidelity_primary_min": 0.95,
        "performance_retention_min": 0.90,
        "coverage_error_worsening_max": 0.05,
        "latency_ratio_max": 0.25,
        "artifact_ratio_max": 0.25,
    },
    "clustering": {
        "fidelity_primary_min": 0.90,
        "performance_retention_min": 0.90,
        "postprocess_ari_gain_min": 0.00,
    },
}


def parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def safe_ratio(num: Any, den: Any) -> float:
    num_f = finite_float(num)
    den_f = finite_float(den)
    if not math.isfinite(num_f) or not math.isfinite(den_f) or abs(den_f) < 1e-12:
        return float("nan")
    return num_f / den_f


def safe_diff(a: Any, b: Any) -> float:
    a_f = finite_float(a)
    b_f = finite_float(b)
    if not math.isfinite(a_f) or not math.isfinite(b_f):
        return float("nan")
    return a_f - b_f


def value_or_nan(row: pd.Series, key: str) -> float:
    if key not in row:
        return float("nan")
    return finite_float(row.get(key))


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def nanmean(values: Iterable[float]) -> float:
    arr = np.asarray([finite_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def nanstd(values: Iterable[float]) -> float:
    arr = np.asarray([finite_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(arr.std(ddof=1))


def bootstrap_mean_ci(values: Iterable[float], n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    arr = np.asarray([finite_float(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1 or n_boot <= 1:
        return float(arr[0]), float(arr[0])
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def standardized_mean(values: Iterable[float]) -> float:
    mu = nanmean(values)
    sd = nanstd(values)
    if not math.isfinite(mu) or not math.isfinite(sd) or sd <= 1e-12:
        return float("nan")
    return float(mu / sd)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def make_mock_sources(out_dir: Path) -> Tuple[Path, Path]:
    panel_c = out_dir / "mock_panel_c"
    panel_b = out_dir / "mock_panel_b"
    panel_c.mkdir(parents=True, exist_ok=True)
    panel_b.mkdir(parents=True, exist_ok=True)

    rows = []
    for task, datasets in [("classification", CLASSIFICATION_DATASETS), ("regression", REGRESSION_DATASETS)]:
        for dataset in datasets:
            for seed in [42, 44, 46]:
                if task == "classification":
                    rows.append(
                        {
                            "task": task,
                            "dataset": dataset,
                            "seed": seed,
                            "status": "success",
                            "seconds": 10 + seed % 3,
                            "teacher_model_count": 3,
                            "student_model_count": 3,
                            "distill_success_count": 3,
                            "teacher_auc": 0.80,
                            "student_auc": 0.79,
                            "teacher_ece": 0.10,
                            "student_ece": 0.08,
                            "teacher_brier": 0.18,
                            "student_brier": 0.17,
                            "teacher_nll": 0.50,
                            "student_nll": 0.51,
                            "fidelity_prob_pearson": 0.94,
                            "latency_ratio_student_over_teacher": 0.10,
                            "artifact_size_ratio_student_over_teacher": 0.05,
                            "llm_calls": 5,
                            "llm_total_tokens": 5000,
                        }
                    )
                else:
                    rows.append(
                        {
                            "task": task,
                            "dataset": dataset,
                            "seed": seed,
                            "status": "success",
                            "seconds": 11 + seed % 3,
                            "teacher_model_count": 3,
                            "student_model_count": 3,
                            "distill_success_count": 3,
                            "teacher_rmse": 2.0,
                            "student_rmse": 2.2,
                            "teacher_r2": 0.88,
                            "student_r2": 0.85,
                            "fidelity_pred_pearson": 0.96,
                            "fidelity_pred_nrmse_y_std": 0.20,
                            "latency_ratio_student_over_teacher": 0.12,
                            "artifact_size_ratio_student_over_teacher": 0.07,
                            "llm_calls": 5,
                            "llm_total_tokens": 5000,
                        }
                    )
    pd.DataFrame(rows).to_csv(panel_c / "per_run_metrics.csv", index=False)

    reg_rows = []
    for dataset in REGRESSION_DATASETS:
        for seed in [42, 44, 46]:
            for family, prefix, ce in [
                ("teacher_stack", "teacher", 0.03),
                ("distilled_student", "student", 0.04),
            ]:
                reg_rows.append(
                    {
                        "task": "regression",
                        "dataset": dataset,
                        "seed": seed,
                        "config": f"{prefix}_split_conformal_alpha_0_10",
                        "model_family": family,
                        "interval_method": "split_conformal",
                        "alpha": 0.10,
                        "picp": 0.90 - ce,
                        "pinaw": 0.25,
                        "coverage_error": ce,
                        "failed_flag": 0,
                    }
                )
    pd.DataFrame(reg_rows).to_csv(panel_b / "regression_per_seed.csv", index=False)

    cluster_rows = []
    for dataset in CLUSTERING_DATASETS:
        for seed in [42, 44, 46]:
            for config, family, ari, fallback in [
                ("teacher_no_consensus", "teacher_stack", 0.55, False),
                ("teacher_majority_voting", "teacher_stack", 0.58, False),
                ("teacher_weighted_voting", "teacher_stack", 0.59, False),
                ("student_no_consensus", "distilled_student", 0.52, False),
                ("student_majority_voting", "distilled_student", 0.54, False),
                ("student_weighted_voting", "distilled_student", 0.55, False),
            ]:
                cluster_rows.append(
                    {
                        "task": "clustering",
                        "dataset": dataset,
                        "seed": seed,
                        "config": config,
                        "model_family": family,
                        "postprocessing_method": config.split("_", 1)[1],
                        "ari": ari,
                        "nmi": ari - 0.08,
                        "failed_flag": 0,
                        "selected_model_count": 2,
                        "eligible_model_count": 3,
                        "fallback_used": fallback,
                    }
                )
    pd.DataFrame(cluster_rows).to_csv(panel_b / "clustering_per_seed.csv", index=False)
    pd.DataFrame(
        [
            {"task": "clustering", "dataset": d, "seed": s, "status": "success", "seconds": 5, "llm_calls": 3, "llm_tokens_logged": 0}
            for d in CLUSTERING_DATASETS
            for s in [42, 44, 46]
        ]
    ).to_csv(panel_b / "live_run_records.csv", index=False)
    return panel_c, panel_b


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
        out[f"{prefix}_coverage_error"] = value_or_nan(row, "coverage_error")
        out[f"{prefix}_pinaw"] = value_or_nan(row, "pinaw")
        out[f"{prefix}_picp"] = value_or_nan(row, "picp")
    if "teacher_coverage_error" in out and "student_coverage_error" in out:
        out["coverage_error_delta_student_minus_teacher"] = safe_diff(out["student_coverage_error"], out["teacher_coverage_error"])
        out["output_control_gain"] = -out["coverage_error_delta_student_minus_teacher"]
    if "teacher_pinaw" in out and "student_pinaw" in out:
        out["pinaw_delta_student_minus_teacher"] = safe_diff(out["student_pinaw"], out["teacher_pinaw"])
    return out


def build_from_panel_c(panel_c_dir: Path, panel_b_dir: Path, seeds: List[int]) -> pd.DataFrame:
    metrics = read_csv(panel_c_dir / "per_run_metrics.csv")
    reg_panel_b = read_csv(panel_b_dir / "regression_per_seed.csv")
    rows: List[Dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame(rows)
    metrics["seed_num"] = pd.to_numeric(metrics.get("seed"), errors="coerce")
    for _, row in metrics.iterrows():
        task = str(row.get("task", ""))
        dataset = str(row.get("dataset", ""))
        seed = int(row["seed_num"]) if math.isfinite(row["seed_num"]) else None
        if seed not in seeds:
            continue
        if task == "classification" and dataset not in CLASSIFICATION_DATASETS:
            continue
        if task == "regression" and dataset not in REGRESSION_DATASETS:
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
            "teacher_model_count": value_or_nan(row, "teacher_model_count"),
            "student_model_count": value_or_nan(row, "student_model_count"),
            "effective_model_count": value_or_nan(row, "distill_success_count"),
            "llm_calls": value_or_nan(row, "llm_calls"),
            "llm_total_tokens": value_or_nan(row, "llm_total_tokens"),
            "seconds": value_or_nan(row, "seconds"),
            "latency_ratio": value_or_nan(row, "latency_ratio_student_over_teacher"),
            "artifact_size_ratio": value_or_nan(row, "artifact_size_ratio_student_over_teacher"),
        }
        base["delivery_gain_latency"] = 1.0 - base["latency_ratio"] if math.isfinite(base["latency_ratio"]) else float("nan")
        base["delivery_gain_artifact"] = 1.0 - base["artifact_size_ratio"] if math.isfinite(base["artifact_size_ratio"]) else float("nan")
        if task == "classification":
            teacher_auc = value_or_nan(row, "teacher_auc")
            student_auc = value_or_nan(row, "student_auc")
            teacher_ece = value_or_nan(row, "teacher_ece")
            student_ece = value_or_nan(row, "student_ece")
            base.update(
                {
                    "teacher_performance": teacher_auc,
                    "student_performance": student_auc,
                    "performance_metric": "auc",
                    "performance_delta_student_minus_teacher": safe_diff(student_auc, teacher_auc),
                    "performance_retention": safe_ratio(student_auc, teacher_auc),
                    "fidelity_primary": value_or_nan(row, "fidelity_prob_pearson"),
                    "fidelity_error": value_or_nan(row, "fidelity_prob_mae"),
                    "teacher_ece": teacher_ece,
                    "student_ece": student_ece,
                    "ece_delta_student_minus_teacher": safe_diff(student_ece, teacher_ece),
                    "output_control_gain": safe_diff(teacher_ece, student_ece),
                    "brier_delta_student_minus_teacher": safe_diff(value_or_nan(row, "student_brier"), value_or_nan(row, "teacher_brier")),
                    "nll_delta_student_minus_teacher": safe_diff(value_or_nan(row, "student_nll"), value_or_nan(row, "teacher_nll")),
                }
            )
        elif task == "regression":
            teacher_rmse = value_or_nan(row, "teacher_rmse")
            student_rmse = value_or_nan(row, "student_rmse")
            base.update(
                {
                    "teacher_performance": teacher_rmse,
                    "student_performance": student_rmse,
                    "performance_metric": "rmse",
                    "performance_delta_student_minus_teacher": safe_diff(student_rmse, teacher_rmse),
                    "performance_retention": safe_ratio(teacher_rmse, student_rmse),
                    "fidelity_primary": value_or_nan(row, "fidelity_pred_pearson"),
                    "fidelity_error": value_or_nan(row, "fidelity_pred_nrmse_y_std"),
                    "r2_delta_student_minus_teacher": safe_diff(value_or_nan(row, "student_r2"), value_or_nan(row, "teacher_r2")),
                }
            )
            interval = regression_interval_lookup(reg_panel_b, dataset, seed)
            base.update(interval)
            if not interval:
                base["missing_output_control_reason"] = "regression_interval_source_missing_for_dataset_seed"
        rows.append(base)
    return pd.DataFrame(rows)


def first_config(df: pd.DataFrame, dataset: str, seed: int, config: str) -> Optional[pd.Series]:
    sub = df[
        (df.get("dataset") == dataset)
        & (pd.to_numeric(df.get("seed"), errors="coerce") == int(seed))
        & (df.get("config") == config)
    ]
    if sub.empty:
        return None
    return sub.iloc[0]


def build_clustering(panel_b_dir: Path, seeds: List[int]) -> pd.DataFrame:
    cluster = read_csv(panel_b_dir / "clustering_per_seed.csv")
    live = read_csv(panel_b_dir / "live_run_records.csv")
    rows: List[Dict[str, Any]] = []
    if cluster.empty:
        return pd.DataFrame(rows)
    for dataset in CLUSTERING_DATASETS:
        for seed in seeds:
            teacher = first_config(cluster, dataset, seed, "teacher_no_consensus")
            student = first_config(cluster, dataset, seed, "student_no_consensus")
            teacher_majority = first_config(cluster, dataset, seed, "teacher_majority_voting")
            teacher_weighted = first_config(cluster, dataset, seed, "teacher_weighted_voting")
            student_majority = first_config(cluster, dataset, seed, "student_majority_voting")
            student_weighted = first_config(cluster, dataset, seed, "student_weighted_voting")
            refs = [x for x in [teacher, student, teacher_majority, teacher_weighted, student_majority, student_weighted] if x is not None]
            fallback_count = sum(1 for x in refs if boolish(x.get("fallback_used", False)))
            failed_count = sum(int(finite_float(x.get("failed_flag", 0)) > 0) for x in refs)
            teacher_ari = value_or_nan(teacher, "ari") if teacher is not None else float("nan")
            student_ari = value_or_nan(student, "ari") if student is not None else float("nan")
            teacher_nmi = value_or_nan(teacher, "nmi") if teacher is not None else float("nan")
            student_nmi = value_or_nan(student, "nmi") if student is not None else float("nan")
            teacher_post_ari = max([value_or_nan(x, "ari") for x in [teacher_majority, teacher_weighted] if x is not None] or [float("nan")])
            student_post_ari = max([value_or_nan(x, "ari") for x in [student_majority, student_weighted] if x is not None] or [float("nan")])
            live_sub = live[
                (live.get("task") == "clustering")
                & (live.get("dataset") == dataset)
                & (pd.to_numeric(live.get("seed"), errors="coerce") == int(seed))
            ]
            live_row = live_sub.iloc[0] if not live_sub.empty else pd.Series(dtype=object)
            rows.append(
                {
                    "task": "clustering",
                    "dataset": dataset,
                    "seed": seed,
                    "source_delivery": str(panel_b_dir),
                    "source_output_control": str(panel_b_dir),
                    "run_status": live_row.get("status", "unknown"),
                    "failed_flag": failed_count,
                    "fallback_count": fallback_count,
                    "teacher_model_count": value_or_nan(teacher, "selected_model_count") if teacher is not None else float("nan"),
                    "student_model_count": value_or_nan(student, "selected_model_count") if student is not None else float("nan"),
                    "effective_model_count": value_or_nan(teacher, "eligible_model_count") if teacher is not None else float("nan"),
                    "llm_calls": value_or_nan(live_row, "llm_calls"),
                    "llm_total_tokens": value_or_nan(live_row, "llm_tokens_logged"),
                    "seconds": value_or_nan(live_row, "seconds"),
                    "teacher_performance": teacher_ari,
                    "student_performance": student_ari,
                    "performance_metric": "ari",
                    "performance_delta_student_minus_teacher": safe_diff(student_ari, teacher_ari),
                    "performance_retention": safe_ratio(student_ari, teacher_ari),
                    "teacher_nmi": teacher_nmi,
                    "student_nmi": student_nmi,
                    "nmi_delta_student_minus_teacher": safe_diff(student_nmi, teacher_nmi),
                    "postprocess_teacher_ari_gain": safe_diff(teacher_post_ari, teacher_ari),
                    "postprocess_student_ari_gain": safe_diff(student_post_ari, student_ari),
                    "output_control_gain": safe_diff(teacher_post_ari, teacher_ari),
                    "fidelity_primary": float("nan"),
                    "fidelity_error": float("nan"),
                    "missing_fidelity_reason": "assignment_level_teacher_student_labels_not_recorded",
                    "latency_ratio": float("nan"),
                    "artifact_size_ratio": float("nan"),
                    "delivery_gain_latency": float("nan"),
                    "delivery_gain_artifact": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def add_threshold_flags(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in source.iterrows():
        task = str(row.get("task", ""))
        thresholds = THRESHOLDS.get(task, {})
        reasons = []
        fidelity = value_or_nan(row, "fidelity_primary")
        if "fidelity_primary_min" in thresholds:
            if not math.isfinite(fidelity):
                reasons.append("fidelity_missing")
            elif fidelity < thresholds["fidelity_primary_min"]:
                reasons.append(f"fidelity<{thresholds['fidelity_primary_min']}")
        retention = value_or_nan(row, "performance_retention")
        if "performance_retention_min" in thresholds:
            if not math.isfinite(retention):
                reasons.append("performance_retention_missing")
            elif retention < thresholds["performance_retention_min"]:
                reasons.append(f"performance_retention<{thresholds['performance_retention_min']}")
        latency = value_or_nan(row, "latency_ratio")
        if "latency_ratio_max" in thresholds and math.isfinite(latency) and latency > thresholds["latency_ratio_max"]:
            reasons.append(f"latency_ratio>{thresholds['latency_ratio_max']}")
        artifact = value_or_nan(row, "artifact_size_ratio")
        if "artifact_ratio_max" in thresholds and math.isfinite(artifact) and artifact > thresholds["artifact_ratio_max"]:
            reasons.append(f"artifact_ratio>{thresholds['artifact_ratio_max']}")
        if task == "classification":
            ece_delta = value_or_nan(row, "ece_delta_student_minus_teacher")
            if math.isfinite(ece_delta) and ece_delta > thresholds["ece_worsening_max"]:
                reasons.append(f"ece_worsening>{thresholds['ece_worsening_max']}")
        if task == "regression":
            coverage_delta = value_or_nan(row, "coverage_error_delta_student_minus_teacher")
            if math.isfinite(coverage_delta) and coverage_delta > thresholds["coverage_error_worsening_max"]:
                reasons.append(f"coverage_error_worsening>{thresholds['coverage_error_worsening_max']}")
        if task == "clustering":
            gain = value_or_nan(row, "postprocess_teacher_ari_gain")
            if math.isfinite(gain) and gain < thresholds["postprocess_ari_gain_min"]:
                reasons.append(f"postprocess_ari_gain<{thresholds['postprocess_ari_gain_min']}")
        rows.append(
            {
                "task": task,
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "passes_all_available_thresholds": len(reasons) == 0,
                "failure_reasons": ";".join(reasons),
                "missing_fidelity_reason": row.get("missing_fidelity_reason", ""),
                "missing_output_control_reason": row.get("missing_output_control_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def source_coverage(source: pd.DataFrame, seeds: List[int]) -> pd.DataFrame:
    rows = []
    expected = {
        "classification": CLASSIFICATION_DATASETS,
        "regression": REGRESSION_DATASETS,
        "clustering": CLUSTERING_DATASETS,
    }
    for task, datasets in expected.items():
        for dataset in datasets:
            for seed in seeds:
                g = source[(source.get("task") == task) & (source.get("dataset") == dataset) & (source.get("seed") == seed)]
                row = g.iloc[0] if not g.empty else pd.Series(dtype=object)
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "seed": seed,
                        "row_present": not g.empty,
                        "has_fidelity": math.isfinite(value_or_nan(row, "fidelity_primary")) if not g.empty else False,
                        "has_performance_retention": math.isfinite(value_or_nan(row, "performance_retention")) if not g.empty else False,
                        "has_output_control_gain": math.isfinite(value_or_nan(row, "output_control_gain")) if not g.empty else False,
                        "has_delivery_latency": math.isfinite(value_or_nan(row, "delivery_gain_latency")) if not g.empty else False,
                        "has_delivery_artifact": math.isfinite(value_or_nan(row, "delivery_gain_artifact")) if not g.empty else False,
                    }
                )
    return pd.DataFrame(rows)


def effects(source: pd.DataFrame, n_boot: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    metrics = [
        ("fidelity_primary", "higher"),
        ("performance_retention", "higher"),
        ("output_control_gain", "higher"),
        ("delivery_gain_latency", "higher"),
        ("delivery_gain_artifact", "higher"),
        ("ece_delta_student_minus_teacher", "lower"),
        ("coverage_error_delta_student_minus_teacher", "lower"),
        ("postprocess_teacher_ari_gain", "higher"),
    ]
    rows = []
    for task, g in source.groupby("task", sort=True):
        for metric, direction in metrics:
            if metric not in g:
                continue
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "direction": direction,
                        "n": 0,
                        "mean": float("nan"),
                        "sd": float("nan"),
                        "bootstrap_ci_low": float("nan"),
                        "bootstrap_ci_high": float("nan"),
                        "standardized_mean": float("nan"),
                    }
                )
                continue
            ci_low, ci_high = bootstrap_mean_ci(vals, n_boot=n_boot, rng=rng)
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "direction": direction,
                    "n": int(vals.size),
                    "mean": float(vals.mean()),
                    "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "standardized_mean": standardized_mean(vals),
                }
            )
    return pd.DataFrame(rows)


def failure_table(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, dataset), g in source.groupby(["task", "dataset"], sort=True):
        rows.append(
            {
                "task": task,
                "dataset": dataset,
                "n_dataset_seed": int(len(g)),
                "failed_count": int(pd.to_numeric(g.get("failed_flag"), errors="coerce").fillna(0).sum()),
                "fallback_count": int(pd.to_numeric(g.get("fallback_count"), errors="coerce").fillna(0).sum()),
                "mean_effective_model_count": nanmean(pd.to_numeric(g.get("effective_model_count"), errors="coerce")),
                "mean_llm_calls": nanmean(pd.to_numeric(g.get("llm_calls"), errors="coerce")),
                "mean_llm_total_tokens": nanmean(pd.to_numeric(g.get("llm_total_tokens"), errors="coerce")),
                "mean_seconds": nanmean(pd.to_numeric(g.get("seconds"), errors="coerce")),
            }
        )
    return pd.DataFrame(rows)


def token_check(source: pd.DataFrame, max_tokens_per_run: int) -> pd.DataFrame:
    rows = []
    for _, row in source.iterrows():
        tokens = value_or_nan(row, "llm_total_tokens")
        rows.append(
            {
                "task": row.get("task"),
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "llm_total_tokens": tokens,
                "max_tokens_per_run": max_tokens_per_run,
                "token_abnormal": math.isfinite(tokens) and tokens > max_tokens_per_run,
            }
        )
    return pd.DataFrame(rows)


def write_plot_spec(out_dir: Path) -> None:
    spec = {
        "recommended_panel_d_layout": {
            "rows": ["classification", "regression", "clustering"],
            "columns": ["fidelity_retention", "performance_retention", "output_control_gain", "delivery_gain"],
            "mark": "compact forest + jittered strip",
            "unit": "dataset_seed",
            "color": "task_family",
            "shape_or_outline": "threshold_failure",
            "side_table": ["failure_rate", "fallback_rate", "mean_effective_model_count", "mean_llm_total_tokens"],
        },
        "columns": {
            "fidelity_retention": "fidelity_primary",
            "performance_retention": "performance_retention",
            "output_control_gain": "output_control_gain",
            "delivery_gain_latency": "delivery_gain_latency",
            "delivery_gain_artifact": "delivery_gain_artifact",
        },
        "caveats": [
            "clustering assignment-level teacher-student fidelity is missing unless raw labels are recorded",
            "regression coverage/PINAW are missing for dataset-seeds absent from the panel-b interval source",
        ],
    }
    (out_dir / "panel_d_plot_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["(empty)"]
    shown = df.copy()
    for col in shown.columns:
        shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}" if isinstance(x, float) else str(x))
    cols = list(shown.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return lines


def write_summary(
    out_dir: Path,
    source: pd.DataFrame,
    eff: pd.DataFrame,
    flags: pd.DataFrame,
    coverage: pd.DataFrame,
    panel_c_dir: Path,
    panel_b_dir: Path,
    smoke: bool,
) -> None:
    lines = [
        "# Panel d stability-boundary run summary",
        "",
        f"Mode: {'smoke' if smoke else 'full'}",
        f"Panel c source: `{panel_c_dir}`",
        f"Panel b source: `{panel_b_dir}`",
        "",
        "Files:",
        "",
        "- `panel_d_dataset_seed.csv`: unified dataset-seed table.",
        "- `panel_d_effects.csv`: task-level means, standardized means, and bootstrap CIs.",
        "- `panel_d_threshold_flags.csv`: equivalence-threshold and missing-evidence flags.",
        "- `panel_d_source_coverage.csv`: metric availability by dataset-seed.",
        "- `panel_d_failure_table.csv`: failure/fallback/effective-model/token/runtime summary.",
        "- `panel_d_token_check.csv`: per-run token anomaly checks.",
        "- `panel_d_plot_spec.json`: compact forest/violin/strip plotting specification.",
        "",
        "Coverage:",
        "",
    ]
    cov_summary = coverage.groupby("task").agg(
        rows=("row_present", "sum"),
        has_fidelity=("has_fidelity", "sum"),
        has_output_control_gain=("has_output_control_gain", "sum"),
        has_delivery_latency=("has_delivery_latency", "sum"),
        has_delivery_artifact=("has_delivery_artifact", "sum"),
    )
    lines.extend(markdown_table(cov_summary.reset_index()))
    lines.extend(["", "Threshold failures:", ""])
    fail = flags[~flags["passes_all_available_thresholds"]]
    if fail.empty:
        lines.append("No threshold failures in available metrics.")
    else:
        fail_summary = fail.groupby("task").size().reset_index(name="n_failed_dataset_seed")
        lines.extend(markdown_table(fail_summary))
    lines.extend(["", "Effect summary:", ""])
    keep_metrics = {"fidelity_primary", "performance_retention", "output_control_gain", "delivery_gain_latency", "delivery_gain_artifact"}
    eff_show = eff[eff["metric"].isin(keep_metrics)].copy()
    if not eff_show.empty:
        lines.extend(markdown_table(eff_show))
    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "",
            "- Classification and regression delivery-fidelity are available from the latest gpt-3.5-turbo panel-c run.",
            "- Clustering assignment-level fidelity is not available from the current source records because raw teacher/student cluster labels were not saved.",
            "- Regression interval coverage/PINAW are available only where the panel-b interval source contains matching dataset-seed records.",
        ]
    )
    (out_dir / "RUN_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mock:
        panel_c_dir, panel_b_dir = make_mock_sources(out_dir)
    else:
        panel_c_dir = Path(args.panel_c_dir).resolve()
        panel_b_dir = Path(args.panel_b_dir).resolve()
    seeds = parse_ints(args.seeds)

    class_reg = build_from_panel_c(panel_c_dir, panel_b_dir, seeds)
    cluster = build_clustering(panel_b_dir, seeds)
    source = pd.concat([class_reg, cluster], ignore_index=True, sort=False)
    source = source.sort_values(["task", "dataset", "seed"], kind="stable").reset_index(drop=True)

    flags = add_threshold_flags(source)
    coverage = source_coverage(source, seeds)
    eff = effects(source, n_boot=args.bootstrap, random_state=args.random_state)
    failures = failure_table(source)
    tokens = token_check(source, max_tokens_per_run=args.max_tokens_per_run)

    source.to_csv(out_dir / "panel_d_dataset_seed.csv", index=False)
    eff.to_csv(out_dir / "panel_d_effects.csv", index=False)
    flags.to_csv(out_dir / "panel_d_threshold_flags.csv", index=False)
    coverage.to_csv(out_dir / "panel_d_source_coverage.csv", index=False)
    failures.to_csv(out_dir / "panel_d_failure_table.csv", index=False)
    tokens.to_csv(out_dir / "panel_d_token_check.csv", index=False)
    write_plot_spec(out_dir)
    write_summary(out_dir, source, eff, flags, coverage, panel_c_dir, panel_b_dir, smoke=bool(args.mock))

    manifest = {
        "mode": "smoke" if args.mock else "full",
        "panel_c_dir": str(panel_c_dir),
        "panel_b_dir": str(panel_b_dir),
        "seeds": seeds,
        "classification_datasets": CLASSIFICATION_DATASETS,
        "regression_datasets": REGRESSION_DATASETS,
        "clustering_datasets": CLUSTERING_DATASETS,
        "thresholds": THRESHOLDS,
        "max_tokens_per_run": args.max_tokens_per_run,
        "note": "API keys are not used or written by this aggregation runner.",
    }
    (out_dir / "configs_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Panel d stability and failure-boundary aggregator.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--panel-c-dir", default=str(DEFAULT_PANEL_C_DIR))
    parser.add_argument("--panel-b-dir", default=str(DEFAULT_PANEL_B_DIR))
    parser.add_argument("--seeds", default="42,44,46")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-tokens-per-run", type=int, default=200000)
    parser.add_argument("--mock", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = run(args)
    print(f"[done] panel d outputs written to {out}", flush=True)


if __name__ == "__main__":
    main()
