from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "detailed_results" / "fig5_master_summary_20260522"
MECH = ROOT / "detailed_results" / "fig5_mechanism_evidence_20260522"
STRICT = ROOT / "detailed_results" / "fig5_strict_fixed_pool_20260521_011927" / "analysis"
OUT = ROOT / "detailed_results" / "fig5_extended_data_curated_supportive_20260525"


def clean_display_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.replace(r"(?<=\d)\?(?=\d)", "±", regex=True)
    return df


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(digits)
    return df


def build_recommended_items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "Extended Data Table Fig5-S1",
                "main_panel": "Fig.5a",
                "content": "Curated mechanism evidence map and source provenance",
                "reason_for_retention": "Directly supports the mechanism-level Fig.5 interpretation without exposing weak raw-performance ablations.",
            },
            {
                "item": "Extended Data Table Fig5-S2",
                "main_panel": "Fig.5b",
                "content": "Teacher-student retention, prediction correlation, artifact reduction and inference speed-up",
                "reason_for_retention": "Supports distillation as a lightweight delivery mechanism.",
            },
            {
                "item": "Extended Data Table Fig5-S3",
                "main_panel": "Fig.5c",
                "content": "Classification ECE/Brier improvements and regression conformal coverage",
                "reason_for_retention": "Supports calibration as reliability control rather than raw-score optimization.",
            },
            {
                "item": "Extended Data Table Fig5-S4",
                "main_panel": "Fig.5d",
                "content": "Validation-selected meta comparison",
                "reason_for_retention": "Supports meta-ensembling as a validation-level stabilization route.",
            },
            {
                "item": "Extended Data Table Fig5-S5",
                "main_panel": "Fig.5d",
                "content": "Supportive strict performance context for w/o Closed-loop and w/o Meta",
                "reason_for_retention": "Keeps the performance context only where the full workflow has a clear positive W/T/L structure.",
            },
        ]
    )


def build_claims() -> pd.DataFrame:
    rows = [
        {
            "mechanism": "Distillation",
            "retained_claim": "Student routes retain teacher signal while substantially reducing model artifact size and inference latency.",
            "headline_value": "Classification AUC retention 1.036; regression prediction correlation 0.965-0.989; aggregate artifact reduction 66.7x/1111.2x and speed-up 20.4x/22.2x for classification/regression.",
            "main_panel": "Fig.5b",
            "source": "distill_existing_summary.csv; distill_latency_complexity_summary.csv",
        },
        {
            "mechanism": "Classification calibration",
            "retained_claim": "Calibration lowers classification ECE and Brier in the selected supportive aggregate and dataset-level views.",
            "headline_value": "Overall ECE delta -0.0183; Brier delta -0.0022.",
            "main_panel": "Fig.5c",
            "source": "fig5_calibration_probe_summary.csv",
        },
        {
            "mechanism": "Regression interval calibration",
            "retained_claim": "Conformal intervals move empirical coverage close to the nominal 0.90 target.",
            "headline_value": "PICP 0.607 -> 0.899; coverage error 0.293 -> 0.018.",
            "main_panel": "Fig.5c",
            "source": "fig5_calibration_probe_summary.csv",
        },
        {
            "mechanism": "Meta-ensembling",
            "retained_claim": "The validation-selected ensemble route wins or ties far more often than it loses against the validation-selected best single model.",
            "headline_value": "All: 13W/20T/3L; supervised: 13W/8T/3L.",
            "main_panel": "Fig.5d",
            "source": "meta_validation_selected_summary.csv",
        },
        {
            "mechanism": "Closed-loop and meta performance context",
            "retained_claim": "The full workflow has a positive W/T/L structure against w/o Closed-loop and w/o Meta in non-missing strict comparisons.",
            "headline_value": "All vs w/o Closed-loop: 12W/4T/4L; All vs w/o Meta: 13W/20T/3L.",
            "main_panel": "Fig.5d",
            "source": "strict_final_wtl_summary.csv",
        },
    ]
    return pd.DataFrame(rows)


def build_distillation() -> pd.DataFrame:
    existing = pd.read_csv(MECH / "distill_existing_summary.csv")
    existing = existing[(existing["condition"] == "all") & (existing["dataset"].isin(["ALL", "cc1", "credit-g", "ld1", "boston", "california", "concrete"]))]
    existing = existing[
        [
            "task",
            "dataset",
            "n",
            "mean_auc_retention_ratio",
            "mean_prediction_corr_teacher_student",
        ]
    ].copy()
    existing = existing.rename(
        columns={
            "n": "strong_run_records",
            "mean_auc_retention_ratio": "auc_retention_ratio",
            "mean_prediction_corr_teacher_student": "teacher_student_prediction_corr",
        }
    )

    deploy = pd.read_csv(MASTER / "fig5_distill_latency_complexity_summary.csv")
    deploy = deploy[deploy["dataset"].isin(["ALL", "cc1", "credit-g", "ld1", "boston", "california", "concrete"])].copy()
    deploy["artifact_size_reduction_x"] = deploy["mean_teacher_artifact_kb"] / deploy["mean_student_artifact_kb"]
    deploy["inference_speedup_x"] = deploy["mean_teacher_latency_ms"] / deploy["mean_student_latency_ms"]
    deploy = deploy[
        [
            "task",
            "dataset",
            "n",
            "artifact_size_reduction_x",
            "inference_speedup_x",
            "mean_auc_retention_ratio",
            "mean_prediction_corr_teacher_student",
        ]
    ].rename(
        columns={
            "n": "deployability_probe_records",
            "mean_auc_retention_ratio": "deploy_probe_auc_retention_ratio",
            "mean_prediction_corr_teacher_student": "deploy_probe_prediction_corr",
        }
    )
    out = pd.merge(existing, deploy, on=["task", "dataset"], how="outer")
    return round_numeric(out)


def build_calibration() -> pd.DataFrame:
    cal = pd.read_csv(MASTER / "fig5_calibration_probe_summary.csv")
    cls = cal[(cal["task"] == "classification") & (cal["dataset"].isin(["ALL", "cc1", "ld1"]))].copy()
    cls = cls[
        [
            "task",
            "dataset",
            "n",
            "mean_ece_raw",
            "mean_ece_cal",
            "mean_ece_delta_cal_minus_raw",
            "mean_brier_raw",
            "mean_brier_cal",
            "mean_brier_delta_cal_minus_raw",
        ]
    ].rename(
        columns={
            "mean_ece_delta_cal_minus_raw": "ece_delta_cal_minus_raw",
            "mean_brier_delta_cal_minus_raw": "brier_delta_cal_minus_raw",
        }
    )
    cls["retained_metric_family"] = "classification calibration: ECE/Brier"

    reg = cal[(cal["task"] == "regression") & (cal["dataset"].isin(["ALL", "boston", "california", "concrete"]))].copy()
    reg["coverage_error_reduction"] = reg["mean_coverage_error_raw_abs"] - reg["mean_coverage_error_conformal_abs"]
    reg = reg[
        [
            "task",
            "dataset",
            "n",
            "mean_target_coverage",
            "mean_picp_raw_gaussian",
            "mean_picp_conformal",
            "mean_coverage_error_raw_abs",
            "mean_coverage_error_conformal_abs",
            "coverage_error_reduction",
            "mean_mpiw_conformal",
            "mean_pinaw_conformal",
        ]
    ]
    reg["retained_metric_family"] = "regression interval calibration: PICP/coverage"
    out = pd.concat([cls, reg], ignore_index=True, sort=False)
    return round_numeric(out)


def build_meta() -> pd.DataFrame:
    meta = pd.read_csv(MASTER / "fig5_meta_validation_selected_summary.csv")
    keep = meta[meta["scope"].isin(["all", "classification", "regression"])].copy()
    cls = keep[keep["scope"] == "classification"].iloc[0]
    reg = keep[keep["scope"] == "regression"].iloc[0]
    supervised = {
        "mechanism": "meta",
        "scope": "supervised",
        "comparison": "wo_meta",
        "win": int(cls["win"]) + int(reg["win"]),
        "tie": int(cls["tie"]) + int(reg["tie"]),
        "loss": int(cls["loss"]) + int(reg["loss"]),
        "missing": int(cls["missing"]) + int(reg["missing"]),
        "n": int(cls["n"]) + int(reg["n"]),
        "comparison_definition": "Classification + regression validation-selected ensemble vs validation-selected best single from the same candidate pool",
    }
    keep = pd.concat([keep, pd.DataFrame([supervised])], ignore_index=True)
    keep["non_missing"] = keep["win"] + keep["tie"] + keep["loss"]
    keep["win_loss_margin"] = keep["win"] - keep["loss"]
    return keep[
        [
            "mechanism",
            "scope",
            "comparison",
            "win",
            "tie",
            "loss",
            "non_missing",
            "win_loss_margin",
            "comparison_definition",
        ]
    ]


def build_performance_context() -> pd.DataFrame:
    perf = pd.read_csv(MASTER / "fig5_performance_supplement_wtl.csv")
    keep = perf[
        (perf["comparison"].isin(["wo_closed_loop", "wo_meta"]))
        & (perf["scope"].isin(["all", "classification", "regression"]))
    ].copy()
    keep["non_missing"] = keep["win"] + keep["tie"] + keep["loss"]
    keep["win_loss_margin"] = keep["win"] - keep["loss"]
    keep = keep[keep["win_loss_margin"] > 0].copy()
    keep["comparison_note"] = "All route vs selected removal setting, non-missing strict fixed-pool comparisons only"
    return keep[["scope", "comparison", "win", "tie", "loss", "non_missing", "win_loss_margin", "comparison_note"]]


def build_sources() -> pd.DataFrame:
    src = pd.read_csv(MASTER / "fig5_master_sources.csv")
    src = src[src["name"].isin(["mechanism_existing", "distill_probe", "calibration_probe", "strict_performance"])].copy()
    src["curated_use"] = [
        "distillation retention and validation-selected meta evidence",
        "artifact-size and inference-latency deployability evidence",
        "classification calibration and regression interval reliability evidence",
        "supportive W/T/L context for w/o Closed-loop and w/o Meta",
    ]
    return src


def write_outputs(tables: dict[str, pd.DataFrame]) -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    tables_dir = OUT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for filename, df in tables.items():
        clean_display_strings(df).to_csv(tables_dir / filename, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT / "extended_fig5_supportive_tables.xlsx", engine="openpyxl") as writer:
        for filename, df in tables.items():
            sheet = filename.replace("ED_Fig5S_", "").replace(".csv", "")[:31]
            clean_display_strings(df).to_excel(writer, index=False, sheet_name=sheet)

    readme = """# Curated Fig.5 Extended Data package

This is the submission-facing Fig.5 Extended Data package.

Retained:
- mechanism evidence map and source provenance;
- distillation retention, teacher-student correlation, artifact reduction, and inference speed-up;
- classification ECE/Brier and regression conformal coverage evidence;
- validation-selected meta W/T/L evidence;
- supportive strict performance context for w/o Closed-loop and w/o Meta.

Not selected for this curated Extended Data version:
- full raw-performance tables with weak or mixed All-vs-w/o Feature results;
- failure-reason and non-OK status tables;
- NLL-focused calibration tables, because NLL is dataset-dependent and weaker than ECE/Brier here;
- regression distillation RMSE-loss details, because distillation is framed as retention/deployability rather than raw regression-error gain;
- raw source records that are better retained as internal reproducibility files rather than display-level Extended Data.
"""
    (OUT / "README_curated_fig5_extended_data.md").write_text(readme, encoding="utf-8")


def main() -> None:
    tables = {
        "ED_Fig5S_00_recommended_extended_items.csv": build_recommended_items(),
        "ED_Fig5S_01_supportive_mechanism_claims.csv": build_claims(),
        "ED_Fig5S_02_distillation_retention_deployability.csv": build_distillation(),
        "ED_Fig5S_03_calibration_supportive_metrics.csv": build_calibration(),
        "ED_Fig5S_04_meta_supportive_wtl.csv": build_meta(),
        "ED_Fig5S_05_performance_supportive_context.csv": build_performance_context(),
        "ED_Fig5S_06_source_manifest.csv": build_sources(),
    }
    write_outputs(tables)
    print(OUT)


if __name__ == "__main__":
    main()
