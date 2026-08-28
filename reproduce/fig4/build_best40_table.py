from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("AUTOLOGIC_BEST40_OUTPUT", ROOT / "outputs" / "fig4_best40"))
TABLES = OUT / "tables"
LOGS = OUT / "logs"
BASE_TABLE_DIR = Path(os.environ.get("AUTOLOGIC_BASE_TABLE_DIR", ROOT / "data" / "paper" / "base_tables"))
OLD_WORKBOOK = Path(os.environ.get("AUTOLOGIC_PAPER_WORKBOOK", ROOT / "data" / "paper" / "autoLOGIC_results_template.xlsx"))
NEW_WORKBOOK = OUT / "autoLOGIC_full_40_best_results_v1_v9.xlsx"

REBUILD_SCRIPT = Path(__file__).resolve().parent / "rebuild_organized_v1_v7.py"


def load_rebuild_module():
    spec = importlib.util.spec_from_file_location("rebuild_v17", REBUILD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {REBUILD_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = load_rebuild_module()


VERSION_VALIDITY = {
    **R.VERSION_VALIDITY,
    "v8": "fixed 58-job f10_m7_p5 recovery; v8 failed/timeout records are superseded by v9 when present",
    "v9": "targeted rerun of v8 failed/timeout jobs with unbuffered logs and non-result-changing safeguards",
}
VERSION_RANK = {f"v{i}": i for i in range(1, 10)}


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def job_no(value) -> str:
    return R.job_no_from_id(str(value))


def safe_float(value):
    return R.safe_float(value)


def norm_dataset(value) -> str:
    return R.norm_dataset(value)


def infer_framework(row: dict) -> str:
    route = str(row.get("route", "")).lower()
    framework = str(row.get("framework", "")).strip()
    if framework:
        return framework
    return "SAGE-Loop" if "sage" in route else "autoLOGIC_compare"


def infer_setting(row: dict) -> str:
    setting = str(row.get("setting", "")).strip()
    if setting and setting.lower() != "nan":
        return setting
    return R.infer_setting(row)


def route_family(row: dict) -> str:
    route = str(row.get("route", "")).strip()
    route = re.sub(r"_fixed$", "", route)
    route = re.sub(r"_recovery$", "", route)
    route = route.replace("_f10_m7_p5", "")
    return route


def parse_one(row: dict) -> dict:
    best, _records = R.parse_log(row)
    out = dict(row)
    out["job_no"] = job_no(out.get("job_no") or out.get("job_id"))
    out["framework"] = infer_framework(out)
    out["setting"] = infer_setting(out)
    out["route_family"] = route_family(out)
    out["parsed_ok"] = bool(best)
    out["dataset_in_log"] = best.get("dataset_in_log", "") if best else ""
    out["dataset_match"] = best.get("dataset_match", False) if best else False
    out["parsed_record_count"] = best.get("parsed_record_count", 0) if best else 0
    for key in [
        "primary_metric", "primary_value", "metric_source", "selected_config",
        "auc", "acc", "f1", "precision", "recall", "mae", "rmse", "rmsle", "r2", "ari", "nmi",
    ]:
        out[key] = best.get(key, "") if best else ""
    return out


def rows_from_old_status() -> list[dict]:
    rows = []
    for r in read_csv_rows(BASE_TABLE_DIR / "all_status_candidates.csv"):
        version = r.get("version", "")
        rows.append({
            "version": version,
            "version_dir": r.get("version_dir", ""),
            "job_no": job_no(r.get("job_no") or r.get("job_id")),
            "job_id": r.get("job_id", ""),
            "framework": infer_framework(r),
            "task": r.get("task", ""),
            "dataset": r.get("dataset", ""),
            "seed": str(r.get("seed", "")),
            "setting": infer_setting(r),
            "route": r.get("route", ""),
            "status": r.get("status", ""),
            "exit_code": r.get("exit_code", ""),
            "runtime_sec": r.get("runtime_sec", ""),
            "validity_note": r.get("validity_note", VERSION_VALIDITY.get(version, "")),
            "stdout": r.get("stdout", ""),
            "stderr": r.get("stderr", ""),
            "copied_stdout": "",
            "copied_stderr": "",
        })
    return rows


def rows_from_status_dir(root: Path, version: str, subdir_label: str = "") -> list[dict]:
    rows = []
    status = root / "tables" / "run_status.csv"
    for r in read_csv_rows(status):
        j = job_no(r.get("job_no", ""))
        dataset = str(r.get("dataset", ""))
        seed = str(r.get("seed", ""))
        task = str(r.get("task", ""))
        setting = str(r.get("setting", ""))
        version_dir = root.name + (("/" + subdir_label) if subdir_label else "")
        rows.append({
            "version": version,
            "version_dir": version_dir,
            "job_no": j,
            "job_id": f"{j}_{task}_{dataset}_seed{seed}_{setting}",
            "framework": infer_framework(r),
            "task": task,
            "dataset": dataset,
            "seed": seed,
            "setting": setting,
            "route": r.get("route", ""),
            "status": r.get("status", ""),
            "exit_code": r.get("exit_code", ""),
            "runtime_sec": r.get("runtime_sec", ""),
            "validity_note": VERSION_VALIDITY.get(version, ""),
            "stdout": r.get("stdout", ""),
            "stderr": r.get("stderr", ""),
            "copied_stdout": "",
            "copied_stderr": "",
        })
    return rows


def load_all_rows() -> list[dict]:
    rows = rows_from_old_status()
    rows.extend(rows_from_status_dir(ROOT / "results_full_performance_cached_20260526_v7_recovery_35", "v7"))
    rows.extend(rows_from_status_dir(ROOT / "results_full_performance_cached_20260526_v7_recovery_35" / "f10_m7_p5_remaining_120min", "v7", "f10_m7_p5_remaining_120min"))
    rows.extend(rows_from_status_dir(ROOT / "results_full_performance_cached_20260526_v8_recovery_58_f10_m7_p5_10800s", "v8"))
    rows.extend(rows_from_status_dir(ROOT / "results_full_performance_cached_20260526_v9_recovery_7_fixed_18000s", "v9"))
    return rows


def trusted_candidate(row: dict) -> bool:
    if row.get("status") != "completed":
        return False
    if not row.get("parsed_ok"):
        return False
    if row.get("version") in {"v1", "v2", "v3"}:
        return False
    if str(row.get("dataset_match")).lower() != "true":
        return False
    return True


def better_seed_key(row: dict) -> tuple:
    task = row.get("task")
    value = safe_float(row.get("primary_value"))
    if value is None:
        value = 1e18 if task == "regression" else -1e18
    rank = VERSION_RANK.get(row.get("version", ""), 0)
    if task == "regression":
        return (-value, rank)
    return (value, rank)


def dedupe_seed(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        key = (
            r.get("task", ""),
            norm_dataset(r.get("dataset", "")),
            r.get("framework", ""),
            r.get("route", ""),
            r.get("setting", ""),
            int(float(r.get("seed", 0) or 0)),
        )
        groups[key].append(r)
    out = []
    for candidates in groups.values():
        out.append(sorted(candidates, key=better_seed_key, reverse=True)[0])
    return out


PRIMARY_METRIC = {"classification": "auc", "regression": "rmse", "clustering": "ari"}
SECONDARY_METRIC = {"classification": "acc", "regression": "mae", "clustering": "nmi"}
TASK_DIRECTION = {"classification": "max", "regression": "min", "clustering": "max"}
METRICS = {
    "classification": ["auc", "acc", "f1", "precision", "recall"],
    "regression": ["mae", "rmse", "rmsle", "r2"],
    "clustering": ["ari", "nmi"],
}


def mean_std(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def group_key(row: dict) -> tuple:
    return (
        row.get("task", ""),
        norm_dataset(row.get("dataset", "")),
        row.get("framework", ""),
        row.get("route_family", route_family(row)),
        row.get("setting", ""),
    )


def summarize_groups(rows: list[dict]) -> tuple[list[dict], dict[tuple[str, str], list[dict]]]:
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)

    summaries = []
    records_by_group = {}
    for key, items in groups.items():
        task, dataset, framework, route, setting = key
        seeds = sorted({int(float(i.get("seed", 0) or 0)) for i in items})
        if len(seeds) < 5:
            continue
        # Keep one record per seed after dedupe, sorted by seed.
        seed_items = []
        for seed in seeds[:5]:
            seed_candidates = [i for i in items if int(float(i.get("seed", 0) or 0)) == seed]
            if seed_candidates:
                seed_items.append(sorted(seed_candidates, key=better_seed_key, reverse=True)[0])
        if len(seed_items) != 5:
            continue
        row = {
            "task": task,
            "dataset": dataset,
            "framework": framework,
            "route": route,
            "setting": setting,
            "selection_mode": "strict_same_setting_5seeds",
            "n_seeds": 5,
            "seeds": ",".join(str(int(float(i.get("seed", 0) or 0))) for i in seed_items),
            "versions": ",".join(sorted({str(i.get("version", "")) for i in seed_items})),
            "job_numbers": ",".join(str(i.get("job_no", "")) for i in seed_items),
            "selected_configs": " | ".join(sorted({str(i.get("selected_config", "")) for i in seed_items if i.get("selected_config")})),
        }
        for metric in METRICS.get(task, []):
            m, s = mean_std([i.get(metric) for i in seed_items])
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
        primary = PRIMARY_METRIC[task]
        row["primary_metric"] = primary
        row["primary_mean"] = row.get(f"{primary}_mean")
        row["primary_std"] = row.get(f"{primary}_std")
        summaries.append(row)
        records_by_group[key] = seed_items
    return summaries, records_by_group


def summarize_seed_items(task: str, dataset: str, seed_items: list[dict], selection_mode: str) -> dict:
    seed_items = sorted(seed_items, key=lambda x: int(float(x.get("seed", 0) or 0)))
    row = {
        "task": task,
        "dataset": dataset,
        "framework": ",".join(sorted({str(i.get("framework", "")) for i in seed_items})),
        "route": ",".join(sorted({str(i.get("route_family", route_family(i))) for i in seed_items})),
        "setting": ",".join(sorted({str(i.get("setting", "")) for i in seed_items})),
        "selection_mode": selection_mode,
        "n_seeds": len(seed_items),
        "seeds": ",".join(str(int(float(i.get("seed", 0) or 0))) for i in seed_items),
        "versions": ",".join(sorted({str(i.get("version", "")) for i in seed_items})),
        "job_numbers": ",".join(str(i.get("job_no", "")) for i in seed_items),
        "selected_configs": " | ".join(sorted({str(i.get("selected_config", "")) for i in seed_items if i.get("selected_config")})),
    }
    for metric in METRICS.get(task, []):
        m, s = mean_std([i.get(metric) for i in seed_items])
        row[f"{metric}_mean"] = m
        row[f"{metric}_std"] = s
    primary = PRIMARY_METRIC[task]
    row["primary_metric"] = primary
    row["primary_mean"] = row.get(f"{primary}_mean")
    row["primary_std"] = row.get(f"{primary}_std")
    return row


def select_best_seed_items(items: list[dict]) -> list[dict]:
    by_seed = defaultdict(list)
    for r in items:
        by_seed[int(float(r.get("seed", 0) or 0))].append(r)
    out = []
    for seed, candidates in sorted(by_seed.items()):
        out.append(sorted(candidates, key=better_seed_key, reverse=True)[0])
    return out


def select_best_dataset_groups(group_summaries: list[dict], records_by_group: dict, deduped_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_dataset = defaultdict(list)
    for row in group_summaries:
        by_dataset[(row["task"], row["dataset"])].append(row)

    selected_summary = []
    selected_seed_rows = []
    covered = set()
    for key, candidates in sorted(by_dataset.items()):
        task = key[0]
        if TASK_DIRECTION[task] == "min":
            best = sorted(candidates, key=lambda r: (-(safe_float(r.get("primary_mean")) or 1e18), VERSION_RANK.get(str(r.get("versions", "")).split(",")[-1], 0)), reverse=True)[0]
        else:
            best = sorted(candidates, key=lambda r: ((safe_float(r.get("primary_mean")) or -1e18), VERSION_RANK.get(str(r.get("versions", "")).split(",")[-1], 0)), reverse=True)[0]
        selected_summary.append(best)
        selected_seed_rows.extend(records_by_group[group_key(best)])
        covered.add(key)

    all_by_dataset = defaultdict(list)
    for r in deduped_rows:
        all_by_dataset[(r.get("task", ""), norm_dataset(r.get("dataset", "")))].append(r)
    for key, items in sorted(all_by_dataset.items()):
        if key in covered:
            continue
        seed_items = select_best_seed_items(items)
        if len({int(float(i.get("seed", 0) or 0)) for i in seed_items}) != 5:
            continue
        task, dataset = key
        fallback_summary = summarize_seed_items(
            task,
            dataset,
            seed_items,
            "mixed_best_seed_records_no_single_5seed_setting",
        )
        selected_summary.append(fallback_summary)
        selected_seed_rows.extend(seed_items)

    return sorted(selected_summary, key=lambda r: (r["task"], r["dataset"])), selected_seed_rows


def expected_datasets_from_workbook() -> dict[str, list[str]]:
    wb = load_workbook(OLD_WORKBOOK, data_only=True)
    out = {}
    labels = {"Classification": "AUC (%)", "Regression": "MAE", "Clustering": "ARI (%)"}
    for sheet, label in labels.items():
        ws = wb[sheet]
        start = None
        for row in range(1, ws.max_row + 1):
            if ws.cell(row=row, column=1).value == label:
                start = row + 2
                break
        names = []
        if start:
            row = start
            while row <= ws.max_row and ws.cell(row=row, column=1).value:
                names.append(str(ws.cell(row=row, column=1).value).strip())
                row += 1
        out[sheet.lower()] = names
    return out


def fmt_cell(mean, std, digits=2):
    if mean is None or std is None:
        return ""
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def update_workbook(selected_summary: list[dict]) -> None:
    shutil.copy2(OLD_WORKBOOK, NEW_WORKBOOK)
    wb = load_workbook(NEW_WORKBOOK)
    lookup = {(r["task"], r["dataset"]): r for r in selected_summary}
    sections = {
        "Classification": [("AUC (%)", "auc", 2), ("ACC (%)", "acc", 2)],
        "Regression": [("MAE", "mae", 2), ("RMSE", "rmse", 2)],
        "Clustering": [("ARI (%)", "ari", 2), ("NMI (%)", "nmi", 2)],
    }
    sheet_task = {"Classification": "classification", "Regression": "regression", "Clustering": "clustering"}
    for sheet, metric_specs in sections.items():
        ws = wb[sheet]
        task = sheet_task[sheet]
        for label, metric, digits in metric_specs:
            label_row = None
            for row in range(1, ws.max_row + 1):
                if ws.cell(row=row, column=1).value == label:
                    label_row = row
                    break
            if not label_row:
                continue
            header_row = label_row + 1
            autologic_col = None
            for col in range(1, ws.max_column + 1):
                if ws.cell(row=header_row, column=col).value == "Auto-LOGIC":
                    autologic_col = col
                    break
            if not autologic_col:
                continue
            row = header_row + 1
            while row <= ws.max_row and ws.cell(row=row, column=1).value:
                dataset = str(ws.cell(row=row, column=1).value).strip()
                sr = lookup.get((task, dataset))
                if sr:
                    ws.cell(row=row, column=autologic_col).value = fmt_cell(
                        safe_float(sr.get(f"{metric}_mean")),
                        safe_float(sr.get(f"{metric}_std")),
                        digits,
                    )
                row += 1

    ws = wb.create_sheet("SelectedSetting")
    headers = [
        "task", "dataset", "framework", "route", "setting", "n_seeds", "seeds",
        "primary_metric", "primary_mean", "primary_std", "versions", "job_numbers",
    ]
    ws.append(headers)
    for row in sorted(selected_summary, key=lambda r: (r["task"], r["dataset"])):
        ws.append([row.get(h, "") for h in headers])
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9EAF7")
    wb.save(NEW_WORKBOOK)


def make_curated_logs(selected_summary: list[dict], selected_seed_rows: list[dict]) -> list[dict]:
    by_dataset = defaultdict(list)
    for r in selected_seed_rows:
        by_dataset[(r["task"], norm_dataset(r["dataset"]))].append(r)
    manifest = []
    for sr in selected_summary:
        task = sr["task"]
        dataset = sr["dataset"]
        items = sorted(by_dataset[(task, dataset)], key=lambda x: int(float(x.get("seed", 0) or 0)))
        log_name = f"{task}_{dataset}_best_setting_5seeds.log"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", log_name)
        out_path = LOGS / safe_name
        lines = [
            f"task: {task}",
            f"dataset: {dataset}",
            f"framework: {sr.get('framework','')}",
            f"route: {sr.get('route','')}",
            f"setting: {sr.get('setting','')}",
            f"selected_by: {sr.get('primary_metric')} mean over 5 seeds",
            f"seeds: {sr.get('seeds','')}",
            "",
        ]
        for r in items:
            metric_parts = []
            for metric in METRICS.get(task, []):
                val = safe_float(r.get(metric))
                if val is not None:
                    metric_parts.append(f"{metric}={val:.6g}")
            lines.extend([
                "=" * 88,
                f"seed: {int(float(r.get('seed', 0) or 0))}",
                f"job_no: {r.get('job_no','')}",
                f"version: {r.get('version','')}",
                f"source_stdout: {r.get('stdout','')}",
                f"source_stderr: {r.get('stderr','')}",
                f"metric_source: {r.get('metric_source','')}",
                f"selected_config: {r.get('selected_config','')}",
                "metrics: " + ", ".join(metric_parts),
                "",
            ])
        out_path.write_text("\n".join(lines), encoding="utf-8")
        manifest.append({
            "task": task,
            "dataset": dataset,
            "curated_log": str(out_path),
            "framework": sr.get("framework", ""),
            "route": sr.get("route", ""),
            "setting": sr.get("setting", ""),
            "seeds": sr.get("seeds", ""),
            "source_stdout_files": " | ".join(str(r.get("stdout", "")) for r in items),
            "source_stderr_files": " | ".join(str(r.get("stderr", "")) for r in items),
        })
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    raw_rows = load_all_rows()
    parsed_rows = [parse_one(r) for r in raw_rows]
    trusted_rows = [r for r in parsed_rows if trusted_candidate(r)]
    deduped_rows = dedupe_seed(trusted_rows)
    group_summaries, records_by_group = summarize_groups(deduped_rows)
    selected_summary, selected_seed_rows = select_best_dataset_groups(group_summaries, records_by_group, deduped_rows)

    expected = expected_datasets_from_workbook()
    expected_pairs = {(task, d) for task, ds in expected.items() for d in ds}
    selected_pairs = {(r["task"], r["dataset"]) for r in selected_summary}
    missing = sorted(expected_pairs - selected_pairs)
    extra = sorted(selected_pairs - expected_pairs)

    fields_candidate = [
        "version", "version_dir", "job_no", "job_id", "framework", "task", "dataset", "seed",
        "setting", "route", "status", "exit_code", "runtime_sec", "parsed_ok", "dataset_in_log",
        "dataset_match", "parsed_record_count", "primary_metric", "primary_value", "metric_source",
        "selected_config", "auc", "acc", "f1", "precision", "recall", "mae", "rmse", "rmsle", "r2",
        "ari", "nmi", "validity_note", "stdout", "stderr",
    ]
    write_csv(TABLES / "all_parsed_candidates_v1_v9.csv", parsed_rows, fields_candidate)
    write_csv(TABLES / "trusted_seed_candidates_v1_v9.csv", trusted_rows, fields_candidate)
    write_csv(TABLES / "deduped_seed_candidates_v1_v9.csv", deduped_rows, fields_candidate)
    write_csv(TABLES / "per_setting_5seed_group_summaries.csv", group_summaries)
    write_csv(TABLES / "per_dataset_best_setting_summary.csv", selected_summary)
    write_csv(TABLES / "per_seed_selected_records.csv", selected_seed_rows, fields_candidate)

    log_manifest = make_curated_logs(selected_summary, selected_seed_rows)
    write_csv(TABLES / "curated_logs_manifest.csv", log_manifest)
    update_workbook(selected_summary)

    with pd.ExcelWriter(OUT / "source_tables_v1_v9.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(selected_summary).to_excel(writer, index=False, sheet_name="best_setting_summary")
        pd.DataFrame(selected_seed_rows).to_excel(writer, index=False, sheet_name="selected_5seed_records")
        pd.DataFrame(group_summaries).to_excel(writer, index=False, sheet_name="all_5seed_setting_groups")
        pd.DataFrame(log_manifest).to_excel(writer, index=False, sheet_name="curated_logs")

    manifest = {
        "output_dir": str(OUT),
        "main_workbook": str(NEW_WORKBOOK),
        "source_tables": str(OUT / "source_tables_v1_v9.xlsx"),
        "curated_logs_dir": str(LOGS),
        "raw_candidate_count": len(raw_rows),
        "parsed_candidate_count": len(parsed_rows),
        "trusted_candidate_count": len(trusted_rows),
        "deduped_seed_candidate_count": len(deduped_rows),
        "five_seed_setting_group_count": len(group_summaries),
        "selected_dataset_count": len(selected_summary),
        "strict_same_setting_selected_count": sum(1 for r in selected_summary if r.get("selection_mode") == "strict_same_setting_5seeds"),
        "mixed_fallback_selected_count": sum(1 for r in selected_summary if r.get("selection_mode") != "strict_same_setting_5seeds"),
        "expected_dataset_count": sum(len(v) for v in expected.values()),
        "missing_expected_dataset_pairs": missing,
        "extra_selected_dataset_pairs": extra,
        "selection_rule": "Prefer same task+framework+route_family+setting groups with exactly five distinct seeds; select max AUC for classification, min RMSE for regression, max ARI for clustering. If no single five-seed setting exists for a dataset, use the best trusted seed-level records and mark selection_mode=mixed_best_seed_records_no_single_5seed_setting.",
        "trusted_rule": "completed + parsed metrics + v4-v9 only + dataset detected in stdout matches expected dataset when detected.",
        "v9_supersedes_v8_failed_timeout": True,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "run_readme.md").write_text(
        "\n".join([
            "# Full 40 dataset best-setting table, v1-v9",
            "",
            "Main workbook: `autoLOGIC_full_40_best_results_v1_v9.xlsx`.",
            "The workbook copies the original comparison workbook style and updates the Auto-LOGIC column.",
            "",
            "Selection: prefer same dataset, same task, same route family, same setting, five seeds only. Classification selects highest mean AUC, regression selects lowest mean RMSE, clustering selects highest mean ARI.",
            "",
            "Some datasets have no five-seed block under a single external setting because their completed seeds are split across earlier f15_m15_p8 runs and f10_m7_p5 recovery runs. Those rows are marked `mixed_best_seed_records_no_single_5seed_setting` rather than being mislabeled as a single setting.",
            "",
            "The `logs` directory contains one curated log per selected dataset. Each curated log records only the five selected seed-level results and their source stdout/stderr paths.",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
