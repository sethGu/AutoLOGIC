import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def latest_dir(prefix: str):
    candidates = sorted((ROOT / "detailed_results").glob(prefix), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(prefix)
    return candidates[0]


def read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def f4(value):
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def pct(value):
    if value is None or pd.isna(value):
        return "NA"
    return f"{100.0 * float(value):.1f}%"


def get_row(df, **kwargs):
    if df.empty:
        return pd.Series(dtype=object)
    mask = pd.Series(True, index=df.index)
    for key, value in kwargs.items():
        mask &= df[key].astype(str).eq(str(value))
    if not mask.any():
        return pd.Series(dtype=object)
    return df.loc[mask].iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-dir", default=str(ROOT / "detailed_results" / "fig5_mechanism_evidence_20260522"))
    parser.add_argument("--strict-dir", default=str(ROOT / "detailed_results" / "fig5_strict_fixed_pool_20260521_011927"))
    parser.add_argument("--distill-probe-dir", default="")
    parser.add_argument("--calibration-probe-dir", default="")
    parser.add_argument("--out-dir", default=str(ROOT / "detailed_results" / "fig5_master_summary_20260522"))
    args = parser.parse_args()

    mechanism_dir = Path(args.mechanism_dir)
    strict_dir = Path(args.strict_dir)
    distill_probe_dir = Path(args.distill_probe_dir) if args.distill_probe_dir else latest_dir("fig5_distill_latency_complexity_probe_20260522_*")
    calibration_probe_dir = Path(args.calibration_probe_dir) if args.calibration_probe_dir else latest_dir("fig5_calibration_probe_20260522_*")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_distill = read_csv(mechanism_dir / "distill_existing_summary.csv")
    existing_cal = read_csv(mechanism_dir / "calibration_existing_summary.csv")
    meta = read_csv(mechanism_dir / "meta_validation_selected_summary.csv")
    inventory = read_csv(mechanism_dir / "evidence_inventory.csv")
    strict_wtl = read_csv(strict_dir / "analysis" / "strict_final_wtl_summary.csv")
    strict_metric = read_csv(strict_dir / "analysis" / "strict_final_metric_summary.csv")
    distill_probe = read_csv(distill_probe_dir / "distill_latency_complexity_summary.csv")
    calibration_probe = read_csv(calibration_probe_dir / "calibration_probe_summary.csv")

    key_rows = []

    cls_distill_existing = get_row(existing_distill, task="classification", dataset="ALL", condition="all")
    reg_distill_existing = get_row(existing_distill, task="regression", dataset="ALL", condition="all")
    cls_distill_probe = get_row(distill_probe, task="classification", dataset="ALL")
    reg_distill_probe = get_row(distill_probe, task="regression", dataset="ALL")
    cls_cal_probe = get_row(calibration_probe, task="classification", dataset="ALL")
    reg_cal_probe = get_row(calibration_probe, task="regression", dataset="ALL")
    meta_all = get_row(meta, scope="all", comparison="wo_meta")
    meta_cls = get_row(meta, scope="classification", comparison="wo_meta")
    meta_reg = get_row(meta, scope="regression", comparison="wo_meta")
    perf_feature = get_row(strict_wtl, scope="all", comparison="wo_feature")
    perf_closed = get_row(strict_wtl, scope="all", comparison="wo_closed_loop")

    key_rows.append(
        {
            "fig5_role": "Distill retention",
            "result_type": "existing strong-run teacher/student records",
            "headline": (
                f"classification AUC retention={f4(cls_distill_existing.get('mean_auc_retention_ratio'))}, "
                f"student>=teacher={pct(cls_distill_existing.get('pct_student_auc_ge_teacher'))}; "
                f"regression pred corr={f4(reg_distill_existing.get('mean_prediction_corr_teacher_student'))}, "
                f"RMSE ratio student/teacher={f4(reg_distill_existing.get('mean_rmse_ratio_student_over_teacher'))}"
            ),
            "supports": "student keeps most teacher signal, but should not be framed as raw-performance gain",
            "risk": "ld1 and regression RMSE show that distillation is not universally performance preserving",
            "source_file": str(mechanism_dir / "distill_existing_summary.csv"),
        }
    )
    key_rows.append(
        {
            "fig5_role": "Distill deployability",
            "result_type": "new local non-API probe",
            "headline": (
                f"classification artifact ratio={f4(cls_distill_probe.get('mean_artifact_kb_ratio_student_over_teacher'))}, "
                f"latency ratio={f4(cls_distill_probe.get('mean_latency_ratio_student_over_teacher'))}; "
                f"regression artifact ratio={f4(reg_distill_probe.get('mean_artifact_kb_ratio_student_over_teacher'))}, "
                f"latency ratio={f4(reg_distill_probe.get('mean_latency_ratio_student_over_teacher'))}, "
                f"pred corr={f4(reg_distill_probe.get('mean_prediction_corr_teacher_student'))}"
            ),
            "supports": "student models are much smaller and faster while retaining high teacher correlation",
            "risk": "probe uses local teacher/student surrogates because old Fig.5 logs did not save exact model artifacts",
            "source_file": str(distill_probe_dir / "distill_latency_complexity_summary.csv"),
        }
    )
    key_rows.append(
        {
            "fig5_role": "Calibration classification",
            "result_type": "new local validation-selected calibration probe",
            "headline": (
                f"ECE delta={f4(cls_cal_probe.get('mean_ece_delta_cal_minus_raw'))}, "
                f"Brier delta={f4(cls_cal_probe.get('mean_brier_delta_cal_minus_raw'))}, "
                f"NLL delta={f4(cls_cal_probe.get('mean_nll_delta_cal_minus_raw'))}"
            ),
            "supports": "calibration strongly improves ECE and mildly improves Brier under validation selection",
            "risk": "NLL is slightly worse on average; do not claim all calibration metrics improve",
            "source_file": str(calibration_probe_dir / "calibration_probe_summary.csv"),
        }
    )
    key_rows.append(
        {
            "fig5_role": "Calibration regression intervals",
            "result_type": "new local conformal interval probe",
            "headline": (
                f"target coverage=0.9000, conformal PICP={f4(reg_cal_probe.get('mean_picp_conformal'))}, "
                f"raw coverage error={f4(reg_cal_probe.get('mean_coverage_error_raw_abs'))}, "
                f"conformal coverage error={f4(reg_cal_probe.get('mean_coverage_error_conformal_abs'))}"
            ),
            "supports": "interval calibration brings empirical coverage close to nominal coverage",
            "risk": "coverage gain is obtained by wider intervals, so show MPIW together with PICP",
            "source_file": str(calibration_probe_dir / "calibration_probe_summary.csv"),
        }
    )
    key_rows.append(
        {
            "fig5_role": "Meta",
            "result_type": "strict fixed-pool validation-selected comparison",
            "headline": (
                f"overall={int(meta_all.get('win', 0))}W/{int(meta_all.get('tie', 0))}T/{int(meta_all.get('loss', 0))}L; "
                f"classification={int(meta_cls.get('win', 0))}W/{int(meta_cls.get('tie', 0))}T/{int(meta_cls.get('loss', 0))}L; "
                f"regression={int(meta_reg.get('win', 0))}W/{int(meta_reg.get('tie', 0))}T/{int(meta_reg.get('loss', 0))}L"
            ),
            "supports": "ensemble/meta adds validation-selected robustness without test-set oracle selection",
            "risk": "many ties; frame as robustness/fallback rather than universal gain",
            "source_file": str(mechanism_dir / "meta_validation_selected_summary.csv"),
        }
    )
    key_rows.append(
        {
            "fig5_role": "Performance supplement",
            "result_type": "strict fixed-pool end-to-end performance",
            "headline": (
                f"All vs w/o Feature={int(perf_feature.get('win', 0))}W/{int(perf_feature.get('tie', 0))}T/{int(perf_feature.get('loss', 0))}L; "
                f"All vs w/o Closed-loop={int(perf_closed.get('win', 0))}W/{int(perf_closed.get('tie', 0))}T/{int(perf_closed.get('loss', 0))}L"
            ),
            "supports": "performance evidence is defensible but not dominant; best used as supplementary context",
            "risk": "not sufficient as the main Fig.5 story for a top-tier review",
            "source_file": str(strict_dir / "analysis" / "strict_final_wtl_summary.csv"),
        }
    )
    key = pd.DataFrame(key_rows)

    source_rows = [
        {"name": "mechanism_existing", "path": str(mechanism_dir), "role": "parsed existing strong/strict Fig.5 mechanism evidence"},
        {"name": "distill_probe", "path": str(distill_probe_dir), "role": "new non-API distill deployability probe"},
        {"name": "calibration_probe", "path": str(calibration_probe_dir), "role": "new non-API calibration probe"},
        {"name": "strict_performance", "path": str(strict_dir), "role": "strict fixed-pool performance supplement"},
    ]
    sources = pd.DataFrame(source_rows)

    key.to_csv(out_dir / "fig5_master_key_claims.csv", index=False, encoding="utf-8-sig")
    sources.to_csv(out_dir / "fig5_master_sources.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(out_dir / "fig5_existing_evidence_inventory.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(out_dir / "fig5_meta_validation_selected_summary.csv", index=False, encoding="utf-8-sig")
    strict_wtl.to_csv(out_dir / "fig5_performance_supplement_wtl.csv", index=False, encoding="utf-8-sig")
    strict_metric.to_csv(out_dir / "fig5_performance_supplement_metrics.csv", index=False, encoding="utf-8-sig")
    distill_probe.to_csv(out_dir / "fig5_distill_latency_complexity_summary.csv", index=False, encoding="utf-8-sig")
    calibration_probe.to_csv(out_dir / "fig5_calibration_probe_summary.csv", index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(out_dir / "fig5_master_summary.xlsx", engine="openpyxl") as writer:
            key.to_excel(writer, sheet_name="key_claims", index=False)
            sources.to_excel(writer, sheet_name="sources", index=False)
            distill_probe.to_excel(writer, sheet_name="distill_deployability", index=False)
            calibration_probe.to_excel(writer, sheet_name="calibration_probe", index=False)
            meta.to_excel(writer, sheet_name="meta", index=False)
            strict_wtl.to_excel(writer, sheet_name="perf_wtl", index=False)
            existing_distill.to_excel(writer, sheet_name="existing_distill", index=False)
            existing_cal.to_excel(writer, sheet_name="existing_calibration", index=False)
    except Exception as exc:
        (out_dir / "xlsx_write_error.txt").write_text(repr(exc), encoding="utf-8")

    readme = "\n".join(
        [
            "# Fig.5 master summary",
            "",
            f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "Use `fig5_master_key_claims.csv` as the first table. The main Fig.5 should emphasize mechanism evidence, while strict performance stays as supplementary evidence.",
            "",
            "Recommended claim order:",
            "1. Distillation: retention plus much smaller and faster student artifacts.",
            "2. Calibration: ECE and empirical coverage improve, with MPIW shown to avoid overclaiming.",
            "3. Meta: validation-selected ensemble/fallback is better or tied in most cases, without test-set oracle selection.",
            "4. Performance: show as a compact supplement, not as the primary proof.",
            "",
            "Key caveat: these results support controllability, deployability, calibration, and robustness more strongly than absolute performance dominance.",
        ]
    )
    (out_dir / "README_fig5_master_summary.md").write_text(readme, encoding="utf-8")

    print(out_dir)
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
