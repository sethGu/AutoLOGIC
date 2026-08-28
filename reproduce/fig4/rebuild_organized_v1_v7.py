from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("AUTOLOGIC_BASE_RUN_DIR", REPO_ROOT / "data" / "paper" / "base_run"))
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
WORKBOOK = TABLES / "reference" / "autologic_compare_summary.xlsx"

V7_ROOT = ROOT.parent / "results_full_performance_cached_20260526_v7_recovery_35"
V7_F10_ROOT = V7_ROOT / "f10_m7_p5_remaining_120min"


VERSION_VALIDITY = {
    "v1": "invalid_args_in_early_runner; job labels may not match actual script arguments",
    "v2": "early attempt; incomplete and not used for trusted comparison",
    "v3": "early attempt; incomplete and not used for trusted comparison",
    "v4": "completed candidate; some early classification failures existed but completed parsed logs are retained if dataset matches",
    "v5": "completed candidate; result txt side effects existed but copied stdout/stderr are retained if dataset matches",
    "v6": "most isolated v1-v6 run; preferred when tied",
    "v7": "recovery run; completed missing binary f10_m7_p5 jobs and one multiclass f15_m15_p8 job",
}


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def job_no_from_id(value: str) -> str:
    m = re.match(r"(\d{4})", str(value))
    if m:
        return m.group(1)
    try:
        return f"{int(float(value)):04d}"
    except Exception:
        return str(value)


def safe_float(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() == "N/A" or s.lower() == "nan":
        return None
    try:
        x = float(s)
    except Exception:
        return None
    if math.isnan(x):
        return None
    return x


def pct_if_fraction(x: float | None) -> float | None:
    if x is None:
        return None
    if -1.5 <= x <= 1.5:
        return x * 100.0
    return x


def norm_dataset(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    aliases = {
        "ds_credit": "credit-g",
        "credit": "credit-g",
        "balance_scale": "balance-scale",
        "jungle-chess": "jungle_chess",
    }
    return aliases.get(s, s)


def extract_dataset_names(text: str) -> list[str]:
    names: list[str] = []
    patterns = [
        r"=+\s*Dataset\s+([A-Za-z0-9_\-]+)\s*=+",
        r"=+\s*\u6570\u636e\u96c6\s+([A-Za-z0-9_\-]+)\s*=+",
        r"\bDataset\s+([A-Za-z0-9_\-]+)\b",
        r"\u6570\u636e\u96c6\s+([A-Za-z0-9_\-]+)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            name = m.group(1)
            if name and name not in names:
                names.append(name)
    return names


def first_number_after(label_regex: str, body: str) -> float | None:
    m = re.search(label_regex + r"\s*[:：]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)|N/A|nan)", body, re.I)
    return safe_float(m.group(1)) if m else None


CONFIG_BLOCK = re.compile(
    r"\u3010\u914d\u7f6e(\d+):\s*([^\u3011]+)\u3011(?P<body>.*?)(?=\n\u3010\u914d\u7f6e\d+:|\n=+\s*\n\nValid base model count|\n=+\s*\n\n\u5b8c\u6210|\nTotal experiment time|\Z)",
    re.S,
)


def parse_classification(text: str) -> tuple[dict | None, list[dict]]:
    records: list[dict] = []
    for m in CONFIG_BLOCK.finditer(text):
        body = m.group("body")
        acc = first_number_after(r"(?:\u51c6\u786e\u7387|Accuracy|acc)", body)
        f1 = first_number_after(r"(?:F1\u5206\u6570|F1 Score|f1)", body)
        auc = first_number_after(r"AUC", body)
        precision = first_number_after(r"(?:\u7cbe\u786e\u7387|Precision|pre)", body)
        recall = first_number_after(r"(?:\u53ec\u56de\u7387|Recall|rec)", body)
        if auc is not None:
            records.append(
                {
                    "metric_source": f"config_{m.group(1)}",
                    "selected_config": m.group(2).strip(),
                    "auc": pct_if_fraction(auc),
                    "acc": pct_if_fraction(acc),
                    "f1": pct_if_fraction(f1),
                    "precision": pct_if_fraction(precision),
                    "recall": pct_if_fraction(recall),
                }
            )

    if not records:
        tail = text[-20000:]
        low = tail.lower()
        if "stacking" in low:
            acc = first_number_after(r"acc", tail)
            f1 = first_number_after(r"f1", tail)
            auc = first_number_after(r"auc", tail)
            precision = first_number_after(r"pre", tail)
            recall = first_number_after(r"rec", tail)
            if auc is not None:
                records.append(
                    {
                        "metric_source": "stacking_summary",
                        "selected_config": "stacking",
                        "auc": pct_if_fraction(auc),
                        "acc": pct_if_fraction(acc),
                        "f1": pct_if_fraction(f1),
                        "precision": pct_if_fraction(precision),
                        "recall": pct_if_fraction(recall),
                    }
                )

    if not records:
        return None, records
    records.sort(key=lambda r: ((r.get("auc") is not None, r.get("auc") or -1e18), r.get("acc") or -1e18), reverse=True)
    best = dict(records[0])
    best["primary_metric"] = "auc"
    best["primary_value"] = best.get("auc")
    return best, records


def parse_regression(text: str) -> tuple[dict | None, list[dict]]:
    records: list[dict] = []
    for m in CONFIG_BLOCK.finditer(text):
        body = m.group("body")
        mae = first_number_after(r"MAE", body)
        rmse = first_number_after(r"RMSE", body)
        rmsle = first_number_after(r"RMSLE|RMLSE", body)
        r2 = first_number_after(r"R2|r2", body)
        if rmse is not None:
            records.append(
                {
                    "metric_source": f"config_{m.group(1)}",
                    "selected_config": m.group(2).strip(),
                    "mae": mae,
                    "rmse": rmse,
                    "rmsle": rmsle,
                    "r2": r2,
                }
            )

    for m in re.finditer(
        r"regression_([A-Za-z]+).*?MAE\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)).*?RMSE\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)).*?(?:RMSLE|RMLSE)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)|nan)",
        text,
        re.S | re.I,
    ):
        records.append(
            {
                "metric_source": f"regression_{m.group(1).lower()}",
                "selected_config": f"regression_{m.group(1).lower()}",
                "mae": safe_float(m.group(2)),
                "rmse": safe_float(m.group(3)),
                "rmsle": safe_float(m.group(4)),
                "r2": None,
            }
        )

    if not records:
        # Last-resort parser for raw ensemble blocks.
        for m in re.finditer(
            r"MAE\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)).*?RMSE\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)).*?(?:RMSLE|RMLSE)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)|nan)",
            text,
            re.S | re.I,
        ):
            records.append(
                {
                    "metric_source": "raw_regression_block",
                    "selected_config": "raw_regression_block",
                    "mae": safe_float(m.group(1)),
                    "rmse": safe_float(m.group(2)),
                    "rmsle": safe_float(m.group(3)),
                    "r2": None,
                }
            )

    records = [r for r in records if r.get("rmse") is not None]
    if not records:
        return None, records
    records.sort(key=lambda r: r.get("rmse") if r.get("rmse") is not None else 1e18)
    best = dict(records[0])
    best["primary_metric"] = "rmse"
    best["primary_value"] = best.get("rmse")
    return best, records


def parse_clustering(text: str) -> tuple[dict | None, list[dict]]:
    records: list[dict] = []
    pair_pat = re.compile(
        r"ARI\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\r?\n\s*NMI\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        re.I,
    )
    for i, m in enumerate(pair_pat.finditer(text), 1):
        ari = pct_if_fraction(safe_float(m.group(1)))
        nmi = pct_if_fraction(safe_float(m.group(2)))
        records.append(
            {
                "metric_source": f"ari_nmi_pair_{i}",
                "selected_config": f"ari_nmi_pair_{i}",
                "ari": ari,
                "nmi": nmi,
            }
        )
    records = [r for r in records if r.get("ari") is not None]
    if not records:
        return None, records
    records.sort(key=lambda r: r.get("ari") if r.get("ari") is not None else -1e18, reverse=True)
    best = dict(records[0])
    best["primary_metric"] = "ari"
    best["primary_value"] = best.get("ari")
    return best, records


def parse_log(row: dict) -> tuple[dict | None, list[dict]]:
    path = Path(row.get("copied_stdout") or row.get("stdout") or "")
    if not path.exists():
        return None, []
    text = path.read_text(encoding="utf-8", errors="ignore")
    task = row.get("task")
    if task == "classification":
        best, records = parse_classification(text)
    elif task == "regression":
        best, records = parse_regression(text)
    elif task == "clustering":
        best, records = parse_clustering(text)
    else:
        best, records = None, []
    names = extract_dataset_names(text)
    actual = names[-1] if names else ""
    expected = row.get("dataset", "")
    match = (not actual) or norm_dataset(actual) == norm_dataset(expected)
    if best is not None:
        best["dataset_in_log"] = actual
        best["dataset_match"] = match
        best["parsed_record_count"] = len(records)
    return best, records


def copy_log(src_value: str, dst_name: str) -> str:
    if not src_value:
        return ""
    src = Path(src_value)
    if not src.exists():
        return ""
    dst = LOGS / dst_name
    shutil.copy2(src, dst)
    return str(dst)


def v7_rows() -> list[dict]:
    rows: list[dict] = []
    sources = [
        (V7_ROOT / "tables" / "run_status.csv", "v7", "results_full_performance_cached_20260526_v7_recovery_35"),
        (V7_F10_ROOT / "tables" / "run_status.csv", "v7", "results_full_performance_cached_20260526_v7_recovery_35/f10_m7_p5_remaining_120min"),
    ]
    seen: set[tuple[str, str]] = set()
    for status_path, version, version_dir in sources:
        for r in read_csv_rows(status_path):
            job_no = job_no_from_id(r.get("job_no", ""))
            key = (job_no, str(r.get("stdout", "")))
            if key in seen:
                continue
            seen.add(key)
            dataset = r.get("dataset", "")
            seed = r.get("seed", "")
            task = r.get("task", "classification")
            route = r.get("route", "")
            if "multiclass" in route:
                suffix = "compare"
            else:
                suffix = "compare"
            job_id = f"{job_no}_{task}_{dataset}_{suffix}_seed{seed}"
            stdout_name = f"{job_no}_v7_{job_id}_{r.get('setting','')}.stdout.log"
            stderr_name = f"{job_no}_v7_{job_id}_{r.get('setting','')}.stderr.log"
            copied_stdout = copy_log(r.get("stdout", ""), stdout_name)
            copied_stderr = copy_log(r.get("stderr", ""), stderr_name)
            rows.append(
                {
                    "version": version,
                    "version_dir": version_dir,
                    "job_no": job_no,
                    "job_id": job_id,
                    "framework": "autoLOGIC_compare",
                    "task": task,
                    "dataset": dataset,
                    "seed": seed,
                    "setting": r.get("setting", ""),
                    "route": route,
                    "status": r.get("status", ""),
                    "exit_code": r.get("exit_code", ""),
                    "runtime_sec": r.get("runtime_sec", ""),
                    "validity_note": VERSION_VALIDITY[version],
                    "stdout": r.get("stdout", ""),
                    "stderr": r.get("stderr", ""),
                    "copied_stdout": copied_stdout,
                    "copied_stderr": copied_stderr,
                }
            )
    return rows


def load_candidates() -> list[dict]:
    rows: list[dict] = []
    for r in read_csv_rows(TABLES / "all_status_candidates.csv"):
        job_no = job_no_from_id(r.get("job_no") or r.get("job_id"))
        version = r.get("version", "")
        rows.append(
            {
                "version": version,
                "version_dir": r.get("version_dir", ""),
                "job_no": job_no,
                "job_id": r.get("job_id", ""),
                "framework": "",
                "task": r.get("task", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "setting": "",
                "route": r.get("route", ""),
                "status": r.get("status", ""),
                "exit_code": r.get("exit_code", ""),
                "runtime_sec": r.get("runtime_sec", ""),
                "validity_note": r.get("validity_note", VERSION_VALIDITY.get(version, "")),
                "stdout": r.get("stdout", ""),
                "stderr": r.get("stderr", ""),
                "copied_stdout": "",
                "copied_stderr": "",
            }
        )

    selected = {
        (job_no_from_id(r.get("job_no") or r.get("job_id")), r.get("version", "")): r
        for r in read_csv_rows(TABLES / "selected_best_completed_logs.csv")
    }
    for r in rows:
        old = selected.get((r["job_no"], r.get("version", "")))
        if old:
            r["copied_stdout"] = old.get("copied_stdout", "")
            r["copied_stderr"] = old.get("copied_stderr", "")
    rows.extend(v7_rows())
    return rows


def infer_setting(row: dict) -> str:
    if row.get("setting"):
        return str(row.get("setting"))
    task = row.get("task", "")
    route = row.get("route", "")
    if task == "classification":
        return "f15_m15_p8"
    if task == "clustering":
        return "f15_m15_p8"
    if task == "regression" and "sageloop" in route.lower():
        return "f10_m7"
    if task == "regression":
        return "f10_m7_p5"
    return ""


def enrich_candidates(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        best, _records = parse_log(r)
        e = dict(r)
        e["framework"] = e.get("framework") or infer_framework(e)
        e["setting"] = e.get("setting") or infer_setting(e)
        e["parsed_ok"] = bool(best)
        e["dataset_in_log"] = best.get("dataset_in_log", "") if best else ""
        e["dataset_match"] = best.get("dataset_match", False) if best else False
        e["parsed_record_count"] = best.get("parsed_record_count", 0) if best else 0
        for key in ["primary_metric", "primary_value", "metric_source", "selected_config", "auc", "acc", "f1", "precision", "recall", "mae", "rmse", "rmsle", "r2", "ari", "nmi"]:
            e[key] = best.get(key, "") if best else ""
        out.append(e)
    return out


def better_key(row: dict) -> tuple:
    value = safe_float(row.get("primary_value"))
    task = row.get("task")
    if value is None:
        value = -1e18 if task != "regression" else 1e18
    version_rank = {"v1": 1, "v2": 2, "v3": 3, "v4": 4, "v5": 5, "v6": 6, "v7": 7}.get(row.get("version", ""), 0)
    if task == "regression":
        return (-value, version_rank)
    return (value, version_rank)


def select_best(rows: list[dict], trusted: bool) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("status") != "completed" or not r.get("parsed_ok"):
            continue
        if trusted:
            if r.get("version") in {"v1", "v2", "v3"}:
                continue
            if str(r.get("dataset_match")).lower() != "true":
                continue
        grouped.setdefault(job_no_from_id(r.get("job_no")), []).append(r)
    selected: list[dict] = []
    for job_no, candidates in grouped.items():
        best = sorted(candidates, key=better_key, reverse=True)[0]
        selected.append(best)
    return sorted(selected, key=lambda r: int(job_no_from_id(r["job_no"])))


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return float(pd.Series(vals).mean()), float(pd.Series(vals).std(ddof=1))


def summarize_dataset(rows: list[dict]) -> list[dict]:
    metrics_by_task = {
        "classification": ["auc", "acc", "f1", "precision", "recall"],
        "regression": ["mae", "rmse", "rmsle", "r2"],
        "clustering": ["ari", "nmi"],
    }
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r.get("task", ""), r.get("dataset", ""))
        groups.setdefault(key, []).append(r)
    out: list[dict] = []
    for (task, dataset), items in sorted(groups.items()):
        row = {
            "task": task,
            "dataset": dataset,
            "framework": ",".join(sorted(set(str(i.get("framework") or infer_framework(i)) for i in items))),
            "route": ",".join(sorted(set(str(i.get("route", "")) for i in items))),
            "n_seeds": len(items),
            "seeds": ",".join(str(i.get("seed")) for i in sorted(items, key=lambda x: int(float(x.get("seed", 0) or 0)))),
            "versions": ",".join(sorted(set(str(i.get("version")) for i in items))),
            "settings": ",".join(sorted(set(str(i.get("setting")) for i in items if i.get("setting")))),
        }
        for metric in metrics_by_task.get(task, []):
            m, s = mean_std([safe_float(i.get(metric)) for i in items])
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
        out.append(row)
    return out


def infer_framework(row: dict) -> str:
    route = row.get("route", "")
    if "sage" in route.lower():
        return "SAGE-Loop"
    return "autoLOGIC_compare"


def parse_mean_std_cell(value) -> tuple[float | None, float | None]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, None
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", str(value))
    if not nums:
        return None, None
    mean = safe_float(nums[0])
    std = safe_float(nums[1]) if len(nums) > 1 else None
    return mean, std


def workbook_metric_tables() -> dict[tuple[str, str, str], dict]:
    xl = pd.ExcelFile(WORKBOOK)
    metric_labels = {
        "Classification": ["AUC (%)", "ACC (%)"],
        "Regression": ["MAE", "RMSE"],
        "Clustering": ["ARI (%)", "NMI (%)"],
    }
    out: dict[tuple[str, str, str], dict] = {}
    for sheet, labels in metric_labels.items():
        raw = pd.read_excel(WORKBOOK, sheet_name=sheet, header=None)
        for label in labels:
            matches = raw.index[raw.iloc[:, 0].astype(str).str.strip() == label].tolist()
            if not matches:
                continue
            start = matches[0] + 1
            headers = list(raw.iloc[start])
            try:
                dataset_col = headers.index("Dataset")
                autologic_col = headers.index("Auto-LOGIC")
            except ValueError:
                continue
            metric_name = label.replace(" (%)", "").lower()
            for i in range(start + 1, len(raw)):
                dataset = raw.iat[i, dataset_col]
                if pd.isna(dataset):
                    break
                dataset = str(dataset).strip()
                mean, std = parse_mean_std_cell(raw.iat[i, autologic_col])
                out[(sheet.lower(), dataset, metric_name)] = {
                    "workbook_mean": mean,
                    "workbook_std": std,
                    "workbook_cell": raw.iat[i, autologic_col],
                }
    return out


def compare_to_workbook(summary_rows: list[dict], trusted_label: str) -> list[dict]:
    wb = workbook_metric_tables()
    task_sheet = {"classification": "classification", "regression": "regression", "clustering": "clustering"}
    metrics = {
        "classification": ["auc", "acc"],
        "regression": ["mae", "rmse"],
        "clustering": ["ari", "nmi"],
    }
    by_dataset = {(r["task"], r["dataset"]): r for r in summary_rows}
    rows: list[dict] = []
    for task in ["classification", "regression", "clustering"]:
        datasets = sorted({d for (sheet, d, m) in wb.keys() if sheet == task_sheet[task]})
        for dataset in datasets:
            sr = by_dataset.get((task, dataset), {})
            for metric in metrics[task]:
                wbrow = wb.get((task_sheet[task], dataset, metric), {})
                run_mean = safe_float(sr.get(f"{metric}_mean"))
                run_std = safe_float(sr.get(f"{metric}_std"))
                wb_mean = wbrow.get("workbook_mean")
                wb_std = wbrow.get("workbook_std")
                delta = None if run_mean is None or wb_mean is None else run_mean - wb_mean
                if task == "regression":
                    direction = "lower_is_better"
                    status = "" if delta is None else ("better" if delta < 0 else "worse" if delta > 0 else "same")
                else:
                    direction = "higher_is_better"
                    status = "" if delta is None else ("better" if delta > 0 else "worse" if delta < 0 else "same")
                rows.append(
                    {
                        "comparison_set": trusted_label,
                        "task": task,
                        "dataset": dataset,
                        "metric": metric,
                        "direction": direction,
                        "rerun_mean": run_mean,
                        "rerun_std": run_std,
                        "rerun_n_seeds": sr.get("n_seeds", 0),
                        "rerun_versions": sr.get("versions", ""),
                        "rerun_framework": sr.get("framework", ""),
                        "rerun_route": sr.get("route", ""),
                        "workbook_mean": wb_mean,
                        "workbook_std": wb_std,
                        "workbook_cell": wbrow.get("workbook_cell", ""),
                        "delta_rerun_minus_workbook": delta,
                        "status_vs_workbook": status,
                    }
                )
    return rows


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    all_candidates = load_candidates()
    enriched = enrich_candidates(all_candidates)

    candidate_fields = [
        "version",
        "version_dir",
        "job_no",
        "job_id",
        "framework",
        "task",
        "dataset",
        "seed",
        "setting",
        "route",
        "status",
        "exit_code",
        "runtime_sec",
        "parsed_ok",
        "dataset_in_log",
        "dataset_match",
        "parsed_record_count",
        "primary_metric",
        "primary_value",
        "metric_source",
        "selected_config",
        "auc",
        "acc",
        "f1",
        "precision",
        "recall",
        "mae",
        "rmse",
        "rmsle",
        "r2",
        "ari",
        "nmi",
        "validity_note",
        "stdout",
        "stderr",
        "copied_stdout",
        "copied_stderr",
    ]
    write_csv(TABLES / "all_parsed_candidates_v1_v7.csv", enriched, candidate_fields)

    inclusive = select_best(enriched, trusted=False)
    trusted = select_best(enriched, trusted=True)
    write_csv(TABLES / "selected_best_inclusive_v1_v7.csv", inclusive, candidate_fields)
    write_csv(TABLES / "selected_best_trusted_no_v1_v3_datasetmatch_v1_v7.csv", trusted, candidate_fields)

    inclusive_summary = summarize_dataset(inclusive)
    trusted_summary = summarize_dataset(trusted)
    summary_fields = sorted(set().union(*(r.keys() for r in inclusive_summary + trusted_summary))) if (inclusive_summary or trusted_summary) else []
    write_csv(TABLES / "per_dataset_summary_inclusive_v1_v7.csv", inclusive_summary, summary_fields)
    write_csv(TABLES / "per_dataset_summary_trusted_no_v1_v3_datasetmatch_v1_v7.csv", trusted_summary, summary_fields)

    trusted_cmp = compare_to_workbook(trusted_summary, "trusted_no_v1_v3_datasetmatch")
    inclusive_cmp = compare_to_workbook(inclusive_summary, "inclusive_v1_v7")
    cmp_fields = list(trusted_cmp[0].keys()) if trusted_cmp else []
    write_csv(TABLES / "comparison_to_workbook_trusted_no_v1_v3_datasetmatch_v1_v7.csv", trusted_cmp, cmp_fields)
    write_csv(TABLES / "comparison_to_workbook_inclusive_v1_v7.csv", inclusive_cmp, cmp_fields)

    with pd.ExcelWriter(TABLES / "organized_best_results_v1_v7.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(enriched).to_excel(writer, index=False, sheet_name="all_candidates")
        pd.DataFrame(inclusive).to_excel(writer, index=False, sheet_name="best_inclusive")
        pd.DataFrame(trusted).to_excel(writer, index=False, sheet_name="best_trusted")
        pd.DataFrame(inclusive_summary).to_excel(writer, index=False, sheet_name="summary_inclusive")
        pd.DataFrame(trusted_summary).to_excel(writer, index=False, sheet_name="summary_trusted")
        pd.DataFrame(trusted_cmp).to_excel(writer, index=False, sheet_name="compare_workbook_trusted")
        pd.DataFrame(inclusive_cmp).to_excel(writer, index=False, sheet_name="compare_workbook_inclusive")

    manifest = {
        "output_dir": str(ROOT),
        "added_v7_completed_jobs": [r["job_no"] for r in v7_rows()],
        "candidate_count": len(enriched),
        "inclusive_selected_count": len(inclusive),
        "trusted_selected_count": len(trusted),
        "trusted_rule": "completed + parsed_ok + version not in v1/v2/v3 + dataset_in_log matches dataset when detected",
        "metric_selection_rule": "classification=max AUC; regression=min RMSE; clustering=max ARI; tie prefers later version",
        "remaining_jobs_without_trusted_selection": sorted(
            set(f"{i:04d}" for i in range(1, 201)) - set(r["job_no"] for r in trusted)
        ),
        "version_validity": VERSION_VALIDITY,
    }
    (TABLES / "organized_results_manifest_v1_v7.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = "\n".join(
        [
            "# Organized best results v1-v7",
            "",
            "This directory now contains copied v7 recovery logs and rebuilt result tables.",
            "",
            "Use `selected_best_trusted_no_v1_v3_datasetmatch_v1_v7.csv` and `comparison_to_workbook_trusted_no_v1_v3_datasetmatch_v1_v7.csv` for scientific comparison.",
            "",
            "The inclusive table is retained only for provenance because v1/v2/v3 include early runner problems; v1 in particular can have job labels that do not match the dataset actually run in stdout.",
            "",
            "Selection rule: classification=max AUC, regression=min RMSE, clustering=max ARI, with later versions used only as a tie-breaker.",
        ]
    )
    (TABLES / "README_organized_v1_v7.md").write_text(readme, encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
