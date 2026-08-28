import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def as_float(value):
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if math.isfinite(value) else np.nan


def parse_artifact_path(path: Path, artifacts_root: Path):
    rel = path.relative_to(artifacts_root)
    parts = rel.parts
    if len(parts) < 5:
        raise ValueError(f"Unexpected artifact path: {path}")
    task = parts[0]
    dataset = parts[1]
    seed = int(parts[2].replace("seed_", ""))
    condition = parts[3]
    return task, dataset, seed, condition


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_classification_distill(strong_dir: Path):
    artifacts = strong_dir / "artifacts"
    rows = []
    for path in artifacts.rglob("distillation_metrics.json"):
        try:
            task, dataset, seed, condition = parse_artifact_path(path, artifacts)
        except Exception:
            continue
        if task != "classification":
            continue
        data = read_json(path)
        for round_id, models in data.items():
            for model_id, rec in models.items():
                teacher = rec.get("teacher", {})
                student = rec.get("student", {})
                t_auc = as_float(teacher.get("auc"))
                s_auc = as_float(student.get("auc"))
                t_ece = as_float(teacher.get("ece"))
                s_ece = as_float(student.get("ece"))
                t_brier = as_float(teacher.get("brier"))
                s_brier = as_float(student.get("brier"))
                t_nll = as_float(teacher.get("nll"))
                s_nll = as_float(student.get("nll"))
                rows.append(
                    {
                        "mechanism": "distill",
                        "task": task,
                        "dataset": dataset,
                        "seed": seed,
                        "condition": condition,
                        "round": int(round_id),
                        "model_id": int(model_id),
                        "teacher_auc": t_auc,
                        "student_auc": s_auc,
                        "auc_delta_student_minus_teacher": s_auc - t_auc,
                        "auc_retention_ratio": s_auc / t_auc if t_auc and np.isfinite(t_auc) else np.nan,
                        "student_auc_ge_teacher": bool(s_auc >= t_auc) if np.isfinite(s_auc) and np.isfinite(t_auc) else np.nan,
                        "teacher_ece": t_ece,
                        "student_ece": s_ece,
                        "ece_delta_student_minus_teacher": s_ece - t_ece,
                        "student_ece_le_teacher": bool(s_ece <= t_ece) if np.isfinite(s_ece) and np.isfinite(t_ece) else np.nan,
                        "teacher_brier": t_brier,
                        "student_brier": s_brier,
                        "brier_delta_student_minus_teacher": s_brier - t_brier,
                        "student_brier_le_teacher": bool(s_brier <= t_brier) if np.isfinite(s_brier) and np.isfinite(t_brier) else np.nan,
                        "teacher_nll": t_nll,
                        "student_nll": s_nll,
                        "nll_delta_student_minus_teacher": s_nll - t_nll,
                        "student_nll_le_teacher": bool(s_nll <= t_nll) if np.isfinite(s_nll) and np.isfinite(t_nll) else np.nan,
                        "source_file": str(path),
                    }
                )
    return pd.DataFrame(rows)


def collect_regression_distill(strong_dir: Path):
    artifacts = strong_dir / "artifacts"
    interval_by_key = {}
    for path in artifacts.rglob("*_distillation_interval_metrics.json"):
        try:
            task, dataset, seed, condition = parse_artifact_path(path, artifacts)
        except Exception:
            continue
        if task != "regression":
            continue
        data = read_json(path)
        for round_id, models in data.items():
            for model_id, rec in models.items():
                teacher = rec.get("teacher", {})
                student = rec.get("student", {})
                key = (dataset, seed, condition, int(round_id), int(model_id))
                interval_by_key[key] = {
                    "teacher_picp": as_float(teacher.get("picp")),
                    "student_picp": as_float(student.get("picp")),
                    "picp_delta_student_minus_teacher": as_float(student.get("picp")) - as_float(teacher.get("picp")),
                    "teacher_mpiw": as_float(teacher.get("mpiw")),
                    "student_mpiw": as_float(student.get("mpiw")),
                    "mpiw_delta_student_minus_teacher": as_float(student.get("mpiw")) - as_float(teacher.get("mpiw")),
                    "student_mpiw_le_teacher": bool(as_float(student.get("mpiw")) <= as_float(teacher.get("mpiw"))),
                    "interval_source_file": str(path),
                }

    rows = []
    for path in artifacts.rglob("*_distillation_regression_metrics.json"):
        try:
            task, dataset, seed, condition = parse_artifact_path(path, artifacts)
        except Exception:
            continue
        if task != "regression":
            continue
        data = read_json(path)
        for round_id, models in data.items():
            for model_id, rec in models.items():
                teacher = rec.get("teacher_performance", {})
                student = rec.get("student_performance", {})
                consistency = rec.get("teacher_student_consistency", {})
                t_rmse = as_float(teacher.get("rmse"))
                s_rmse = as_float(student.get("rmse"))
                t_mae = as_float(teacher.get("mae"))
                s_mae = as_float(student.get("mae"))
                t_r2 = as_float(teacher.get("r2"))
                s_r2 = as_float(student.get("r2"))
                key = (dataset, seed, condition, int(round_id), int(model_id))
                row = {
                    "mechanism": "distill",
                    "task": task,
                    "dataset": dataset,
                    "seed": seed,
                    "condition": condition,
                    "round": int(round_id),
                    "model_id": int(model_id),
                    "teacher_rmse": t_rmse,
                    "student_rmse": s_rmse,
                    "rmse_delta_student_minus_teacher": s_rmse - t_rmse,
                    "rmse_ratio_student_over_teacher": s_rmse / t_rmse if t_rmse and np.isfinite(t_rmse) else np.nan,
                    "student_rmse_le_teacher": bool(s_rmse <= t_rmse) if np.isfinite(s_rmse) and np.isfinite(t_rmse) else np.nan,
                    "student_rmse_within_10pct_teacher": bool(s_rmse <= 1.1 * t_rmse) if np.isfinite(s_rmse) and np.isfinite(t_rmse) else np.nan,
                    "teacher_mae": t_mae,
                    "student_mae": s_mae,
                    "mae_delta_student_minus_teacher": s_mae - t_mae,
                    "teacher_r2": t_r2,
                    "student_r2": s_r2,
                    "r2_delta_student_minus_teacher": s_r2 - t_r2,
                    "prediction_corr_teacher_student": as_float(consistency.get("pred_corr")),
                    "prediction_mae_teacher_student": as_float(consistency.get("pred_mae")),
                    "source_file": str(path),
                }
                row.update(interval_by_key.get(key, {}))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_distill(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()
    out = []
    group_cols = ["task", "condition"]
    for keys, g in df.groupby(group_cols, dropna=False):
        task, condition = keys
        row = {"mechanism": "distill", "task": task, "dataset": "ALL", "condition": condition, "n": len(g)}
        if task == "classification":
            row.update(
                {
                    "mean_auc_delta_student_minus_teacher": g["auc_delta_student_minus_teacher"].mean(),
                    "mean_auc_retention_ratio": g["auc_retention_ratio"].mean(),
                    "pct_student_auc_ge_teacher": g["student_auc_ge_teacher"].mean(),
                    "mean_ece_delta_student_minus_teacher": g["ece_delta_student_minus_teacher"].mean(),
                    "pct_student_ece_le_teacher": g["student_ece_le_teacher"].mean(),
                    "mean_brier_delta_student_minus_teacher": g["brier_delta_student_minus_teacher"].mean(),
                    "pct_student_brier_le_teacher": g["student_brier_le_teacher"].mean(),
                    "mean_nll_delta_student_minus_teacher": g["nll_delta_student_minus_teacher"].mean(),
                    "pct_student_nll_le_teacher": g["student_nll_le_teacher"].mean(),
                }
            )
        elif task == "regression":
            row.update(
                {
                    "mean_rmse_delta_student_minus_teacher": g["rmse_delta_student_minus_teacher"].mean(),
                    "mean_rmse_ratio_student_over_teacher": g["rmse_ratio_student_over_teacher"].mean(),
                    "pct_student_rmse_le_teacher": g["student_rmse_le_teacher"].mean(),
                    "pct_student_rmse_within_10pct_teacher": g["student_rmse_within_10pct_teacher"].mean(),
                    "mean_prediction_corr_teacher_student": g["prediction_corr_teacher_student"].mean(),
                    "mean_picp_delta_student_minus_teacher": g.get("picp_delta_student_minus_teacher", pd.Series(dtype=float)).mean(),
                    "mean_mpiw_delta_student_minus_teacher": g.get("mpiw_delta_student_minus_teacher", pd.Series(dtype=float)).mean(),
                    "pct_student_mpiw_le_teacher": g.get("student_mpiw_le_teacher", pd.Series(dtype=float)).mean(),
                }
            )
        out.append(row)
    for keys, g in df.groupby(["task", "dataset", "condition"], dropna=False):
        task, dataset, condition = keys
        row = {"mechanism": "distill", "task": task, "dataset": dataset, "condition": condition, "n": len(g)}
        if task == "classification":
            row.update(
                {
                    "mean_auc_delta_student_minus_teacher": g["auc_delta_student_minus_teacher"].mean(),
                    "mean_auc_retention_ratio": g["auc_retention_ratio"].mean(),
                    "pct_student_auc_ge_teacher": g["student_auc_ge_teacher"].mean(),
                    "mean_ece_delta_student_minus_teacher": g["ece_delta_student_minus_teacher"].mean(),
                    "pct_student_ece_le_teacher": g["student_ece_le_teacher"].mean(),
                    "mean_brier_delta_student_minus_teacher": g["brier_delta_student_minus_teacher"].mean(),
                    "pct_student_brier_le_teacher": g["student_brier_le_teacher"].mean(),
                    "mean_nll_delta_student_minus_teacher": g["nll_delta_student_minus_teacher"].mean(),
                    "pct_student_nll_le_teacher": g["student_nll_le_teacher"].mean(),
                }
            )
        elif task == "regression":
            row.update(
                {
                    "mean_rmse_delta_student_minus_teacher": g["rmse_delta_student_minus_teacher"].mean(),
                    "mean_rmse_ratio_student_over_teacher": g["rmse_ratio_student_over_teacher"].mean(),
                    "pct_student_rmse_le_teacher": g["student_rmse_le_teacher"].mean(),
                    "pct_student_rmse_within_10pct_teacher": g["student_rmse_within_10pct_teacher"].mean(),
                    "mean_prediction_corr_teacher_student": g["prediction_corr_teacher_student"].mean(),
                    "mean_picp_delta_student_minus_teacher": g.get("picp_delta_student_minus_teacher", pd.Series(dtype=float)).mean(),
                    "mean_mpiw_delta_student_minus_teacher": g.get("mpiw_delta_student_minus_teacher", pd.Series(dtype=float)).mean(),
                    "pct_student_mpiw_le_teacher": g.get("student_mpiw_le_teacher", pd.Series(dtype=float)).mean(),
                }
            )
        out.append(row)
    return pd.DataFrame(out)


def parse_value_list(text):
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [as_float(x) for x in re.split(r"\s*,\s*", inner)]
    return [as_float(text.split("±")[0].strip())]


def collect_classification_calibration(strong_dir: Path):
    artifacts = strong_dir / "artifacts"
    rows = []
    pattern = re.compile(r"^\s*(ECE|Brier|NLL)\((raw|cal)\):\s*(\[[^\]]*\])")
    for path in artifacts.rglob("*_results.txt"):
        try:
            task, dataset, seed, condition = parse_artifact_path(path, artifacts)
        except Exception:
            continue
        if task != "classification":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        config = None
        values = {}
        for line in text.splitlines():
            if "配置2" in line:
                config = "cfg2_no_distill_cal"
            elif "配置3" in line:
                config = "cfg3_distill_cal"
            match = pattern.search(line)
            if not match or config is None:
                continue
            metric, kind, raw_values = match.groups()
            values.setdefault(config, {}).setdefault(metric.lower(), {})[kind] = parse_value_list(raw_values)

        for config_name, metric_map in values.items():
            max_len = 0
            for metric_values in metric_map.values():
                max_len = max(max_len, len(metric_values.get("raw", [])), len(metric_values.get("cal", [])))
            for idx in range(max_len):
                row = {
                    "mechanism": "calibration",
                    "task": task,
                    "dataset": dataset,
                    "seed": seed,
                    "condition": condition,
                    "config": config_name,
                    "round_index": idx,
                    "source_file": str(path),
                }
                for metric in ["ece", "brier", "nll"]:
                    raw_list = metric_map.get(metric, {}).get("raw", [])
                    cal_list = metric_map.get(metric, {}).get("cal", [])
                    raw = raw_list[idx] if idx < len(raw_list) else np.nan
                    cal = cal_list[idx] if idx < len(cal_list) else np.nan
                    row[f"{metric}_raw"] = raw
                    row[f"{metric}_cal"] = cal
                    row[f"{metric}_delta_cal_minus_raw"] = cal - raw
                    row[f"{metric}_cal_le_raw"] = bool(cal <= raw) if np.isfinite(cal) and np.isfinite(raw) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def collect_regression_intervals(strong_dir: Path):
    artifacts = strong_dir / "artifacts"
    rows = []
    for path in artifacts.rglob("*_meta_model_interval_metrics.json"):
        try:
            task, dataset, seed, condition = parse_artifact_path(path, artifacts)
        except Exception:
            continue
        if task != "regression":
            continue
        data = read_json(path)
        for round_id, configs in data.items():
            for config_name, alphas in configs.items():
                for alpha_str, rec in alphas.items():
                    alpha = as_float(alpha_str)
                    picp = as_float(rec.get("picp"))
                    mpiw = as_float(rec.get("mpiw"))
                    rows.append(
                        {
                            "mechanism": "interval_calibration",
                            "task": task,
                            "dataset": dataset,
                            "seed": seed,
                            "condition": condition,
                            "round": int(round_id),
                            "config": config_name,
                            "alpha": alpha,
                            "target_coverage": 1.0 - alpha if np.isfinite(alpha) else np.nan,
                            "picp": picp,
                            "mpiw": mpiw,
                            "pinaw": as_float(rec.get("pinaw")),
                            "coverage_error_abs": abs(picp - (1.0 - alpha)) if np.isfinite(picp) and np.isfinite(alpha) else np.nan,
                            "source_file": str(path),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_calibration(class_cal: pd.DataFrame, reg_intervals: pd.DataFrame):
    rows = []
    if not class_cal.empty:
        for keys, g in class_cal.groupby(["task", "condition", "config"], dropna=False):
            task, condition, config = keys
            row = {"mechanism": "calibration", "task": task, "dataset": "ALL", "condition": condition, "config": config, "n": len(g)}
            for metric in ["ece", "brier", "nll"]:
                row[f"mean_{metric}_delta_cal_minus_raw"] = g[f"{metric}_delta_cal_minus_raw"].mean()
                row[f"pct_{metric}_cal_le_raw"] = g[f"{metric}_cal_le_raw"].mean()
            rows.append(row)
        for keys, g in class_cal.groupby(["task", "dataset", "condition", "config"], dropna=False):
            task, dataset, condition, config = keys
            row = {"mechanism": "calibration", "task": task, "dataset": dataset, "condition": condition, "config": config, "n": len(g)}
            for metric in ["ece", "brier", "nll"]:
                row[f"mean_{metric}_delta_cal_minus_raw"] = g[f"{metric}_delta_cal_minus_raw"].mean()
                row[f"pct_{metric}_cal_le_raw"] = g[f"{metric}_cal_le_raw"].mean()
            rows.append(row)
    if not reg_intervals.empty:
        focus = reg_intervals[np.isclose(reg_intervals["alpha"], 0.1, equal_nan=False)].copy()
        if focus.empty:
            focus = reg_intervals.copy()
        for keys, g in focus.groupby(["task", "condition", "config"], dropna=False):
            task, condition, config = keys
            rows.append(
                {
                    "mechanism": "interval_calibration",
                    "task": task,
                    "dataset": "ALL",
                    "condition": condition,
                    "config": config,
                    "n": len(g),
                    "alpha": g["alpha"].median(),
                    "mean_picp": g["picp"].mean(),
                    "mean_mpiw": g["mpiw"].mean(),
                    "mean_pinaw": g["pinaw"].mean(),
                    "mean_coverage_error_abs": g["coverage_error_abs"].mean(),
                }
            )
        for keys, g in focus.groupby(["task", "dataset", "condition", "config"], dropna=False):
            task, dataset, condition, config = keys
            rows.append(
                {
                    "mechanism": "interval_calibration",
                    "task": task,
                    "dataset": dataset,
                    "condition": condition,
                    "config": config,
                    "n": len(g),
                    "alpha": g["alpha"].median(),
                    "mean_picp": g["picp"].mean(),
                    "mean_mpiw": g["mpiw"].mean(),
                    "mean_pinaw": g["pinaw"].mean(),
                    "mean_coverage_error_abs": g["coverage_error_abs"].mean(),
                }
            )
    return pd.DataFrame(rows)


def collect_meta(strict_dir: Path):
    by_seed_path = strict_dir / "analysis" / "strict_final_wtl_by_seed.csv"
    summary_path = strict_dir / "analysis" / "strict_final_wtl_summary.csv"
    by_seed = pd.read_csv(by_seed_path) if by_seed_path.exists() else pd.DataFrame()
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if not by_seed.empty:
        by_seed = by_seed[by_seed["comparison"] == "wo_meta"].copy()
        by_seed.insert(0, "mechanism", "meta")
        by_seed["comparison_definition"] = "All validation-selected option vs validation-selected best single from the same candidate pool"
    if not summary.empty:
        summary = summary[summary["comparison"] == "wo_meta"].copy()
        summary.insert(0, "mechanism", "meta")
        summary["comparison_definition"] = "All validation-selected option vs validation-selected best single from the same candidate pool"
    return by_seed, summary


def write_inventory(out_dir: Path, strong_dir: Path, strict_dir: Path, distill_raw: pd.DataFrame, class_cal: pd.DataFrame, reg_intervals: pd.DataFrame, meta_summary: pd.DataFrame):
    rows = [
        {
            "mechanism": "Distill retention",
            "existing_status": "usable",
            "source": str(strong_dir / "artifacts"),
            "usable_for_main_fig": "yes, but as retention/consistency rather than raw performance",
            "limitation": "does not contain saved teacher/student artifacts or direct predict latency",
            "needs_new_run": "artifact complexity and inference latency",
            "n_existing_rows": len(distill_raw),
        },
        {
            "mechanism": "Calibration classification",
            "existing_status": "partly usable",
            "source": str(strong_dir / "artifacts"),
            "usable_for_main_fig": "usable if reported as raw-vs-cal metric effect",
            "limitation": "existing calibration is not always favorable; validation-selected calibration probe is cleaner",
            "needs_new_run": "recommended for cleaner ECE/Brier/NLL panel",
            "n_existing_rows": len(class_cal),
        },
        {
            "mechanism": "Regression interval quality",
            "existing_status": "usable with caveat",
            "source": str(strong_dir / "artifacts"),
            "usable_for_main_fig": "usable as PICP/MPIW interval-quality evidence",
            "limitation": "not a strict raw-vs-calibrated pair unless supplemented",
            "needs_new_run": "recommended if the claim is calibration gain, not interval availability",
            "n_existing_rows": len(reg_intervals),
        },
        {
            "mechanism": "Meta",
            "existing_status": "usable",
            "source": str(strict_dir / "analysis"),
            "usable_for_main_fig": "yes, this is the legal validation-selected best single vs ensemble comparison",
            "limitation": "overall gain is modest; do not frame as universal performance dominance",
            "needs_new_run": "no",
            "n_existing_rows": len(meta_summary),
        },
        {
            "mechanism": "Artifact complexity",
            "existing_status": "missing",
            "source": str(strong_dir / "artifacts"),
            "usable_for_main_fig": "no",
            "limitation": "no serialized model artifacts in existing Fig.5 logs",
            "needs_new_run": "yes, local non-API latency/size probe",
            "n_existing_rows": 0,
        },
        {
            "mechanism": "Inference latency",
            "existing_status": "missing",
            "source": str(strong_dir / "artifacts"),
            "usable_for_main_fig": "no",
            "limitation": "existing logs contain total training/runtime, not isolated predict latency",
            "needs_new_run": "yes, local non-API latency/size probe",
            "n_existing_rows": 0,
        },
    ]
    pd.DataFrame(rows).to_csv(out_dir / "evidence_inventory.csv", index=False, encoding="utf-8-sig")


def write_readme(out_dir: Path, strong_dir: Path, strict_dir: Path):
    text = f"""# Fig.5 mechanism evidence

Generated at: {datetime.now().isoformat(timespec="seconds")}

Existing sources:
- Strong AutoLOGIC artifact run: `{strong_dir}`
- Strict fixed-pool performance run: `{strict_dir}`

Files:
- `distill_existing_raw.csv`: teacher/student retention and consistency records parsed from existing strong-run JSON files.
- `distill_existing_summary.csv`: task-, dataset-, and condition-level distillation summaries.
- `classification_calibration_existing_raw.csv`: ECE/Brier/NLL raw-vs-cal records parsed from existing classification result logs.
- `regression_interval_existing_raw.csv`: PICP/MPIW/PINAW interval-quality records parsed from existing regression interval JSON files.
- `calibration_existing_summary.csv`: summary of classification calibration and regression interval quality.
- `meta_validation_selected_by_seed.csv`: legal Meta comparison at seed level.
- `meta_validation_selected_summary.csv`: legal Meta win/tie/loss summary.
- `evidence_inventory.csv`: which mechanisms can already be supported and which require a new local probe.

Interpretation guardrails:
- Distillation should be presented as retention plus deployability evidence, not as a raw-performance ablation.
- Calibration should use ECE/Brier/NLL for classification and PICP/MPIW for regression intervals; AUC/RMSE are not calibration metrics.
- Meta must be compared against validation-selected best single model from the same candidate pool, not a test-set oracle.
- Existing logs do not provide clean artifact complexity or isolated inference latency, so those need a new non-API probe.
"""
    (out_dir / "README_fig5_mechanism_evidence.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strong-dir", default=str(ROOT / "detailed_results" / "fig5_strong_full_20260518_121643"))
    parser.add_argument("--strict-dir", default=str(ROOT / "detailed_results" / "fig5_strict_fixed_pool_20260521_011927"))
    parser.add_argument("--out-dir", default=str(ROOT / "detailed_results" / "fig5_mechanism_evidence_20260522"))
    args = parser.parse_args()

    strong_dir = Path(args.strong_dir).resolve()
    strict_dir = Path(args.strict_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    class_distill = collect_classification_distill(strong_dir)
    reg_distill = collect_regression_distill(strong_dir)
    distill_raw = pd.concat([class_distill, reg_distill], ignore_index=True, sort=False)
    distill_summary = summarize_distill(distill_raw)

    class_cal = collect_classification_calibration(strong_dir)
    reg_intervals = collect_regression_intervals(strong_dir)
    cal_summary = summarize_calibration(class_cal, reg_intervals)

    meta_by_seed, meta_summary = collect_meta(strict_dir)

    distill_raw.to_csv(out_dir / "distill_existing_raw.csv", index=False, encoding="utf-8-sig")
    distill_summary.to_csv(out_dir / "distill_existing_summary.csv", index=False, encoding="utf-8-sig")
    class_cal.to_csv(out_dir / "classification_calibration_existing_raw.csv", index=False, encoding="utf-8-sig")
    reg_intervals.to_csv(out_dir / "regression_interval_existing_raw.csv", index=False, encoding="utf-8-sig")
    cal_summary.to_csv(out_dir / "calibration_existing_summary.csv", index=False, encoding="utf-8-sig")
    meta_by_seed.to_csv(out_dir / "meta_validation_selected_by_seed.csv", index=False, encoding="utf-8-sig")
    meta_summary.to_csv(out_dir / "meta_validation_selected_summary.csv", index=False, encoding="utf-8-sig")
    write_inventory(out_dir, strong_dir, strict_dir, distill_raw, class_cal, reg_intervals, meta_summary)
    write_readme(out_dir, strong_dir, strict_dir)

    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "distill_rows": int(len(distill_raw)),
            "classification_calibration_rows": int(len(class_cal)),
            "regression_interval_rows": int(len(reg_intervals)),
            "meta_rows": int(len(meta_by_seed)),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
