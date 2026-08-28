import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fig5_probe_utils import (
    MeanProbEnsemble,
    MeanRegressorEnsemble,
    artifact_kb,
    classification_scores,
    load_classification_dataset,
    load_regression_dataset,
    median_predict_latency_ms,
)


ROOT = Path(__file__).resolve().parents[2]


def rmse(y_true, pred):
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y - p) ** 2)))


def corr(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def classification_candidates(seed):
    return [
        RandomForestClassifier(n_estimators=160, random_state=seed, n_jobs=-1, class_weight="balanced"),
        ExtraTreesClassifier(n_estimators=160, random_state=seed + 1, n_jobs=-1, class_weight="balanced"),
        HistGradientBoostingClassifier(max_iter=140, learning_rate=0.05, random_state=seed + 2),
        GradientBoostingClassifier(random_state=seed + 3),
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]


def regression_candidates(seed):
    return [
        RandomForestRegressor(n_estimators=160, random_state=seed, n_jobs=-1),
        ExtraTreesRegressor(n_estimators=160, random_state=seed + 1, n_jobs=-1),
        HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05, random_state=seed + 2),
        GradientBoostingRegressor(random_state=seed + 3),
        make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    ]


def run_classification(dataset, seed, args):
    x_train, y_train, x_test, y_test = load_classification_dataset(
        dataset, seed, max_train_rows=args.max_train_rows, max_test_rows=args.max_test_rows
    )
    strat = y_train if y_train.nunique() > 1 else None
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.25, random_state=seed, stratify=strat
    )

    teacher_start = time.perf_counter()
    fitted = []
    val_rows = []
    for model_id, model in enumerate(classification_candidates(seed), start=1):
        fitted_model = clone(model)
        fitted_model.fit(x_fit, y_fit)
        val_prob = fitted_model.predict_proba(x_val)[:, 1]
        score = classification_scores(y_val, val_prob)["auc"]
        fitted.append((score, model_id, fitted_model))
        val_rows.append({"candidate_id": model_id, "validation_auc": score})
    fitted = sorted(fitted, key=lambda x: x[0], reverse=True)
    teacher_members = [m for _, _, m in fitted[: args.teacher_top_k]]
    teacher = MeanProbEnsemble(teacher_members)
    teacher_build_seconds = time.perf_counter() - teacher_start

    pseudo_prob = teacher.predict_proba(x_fit)[:, 1]
    pseudo_y = (pseudo_prob >= 0.5).astype(int)
    if len(np.unique(pseudo_y)) < 2:
        pseudo_y = np.asarray(y_fit).astype(int)

    student = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15, learning_rate=0.08, random_state=seed + 11)
    student_start = time.perf_counter()
    student.fit(x_fit, pseudo_y)
    student_build_seconds = time.perf_counter() - student_start

    teacher_prob = teacher.predict_proba(x_test)[:, 1]
    student_prob = student.predict_proba(x_test)[:, 1]
    teacher_scores = classification_scores(y_test, teacher_prob)
    student_scores = classification_scores(y_test, student_prob)
    t_latency, batch_rows = median_predict_latency_ms(teacher, x_test, "classification", args.repeats, args.latency_batch_rows)
    s_latency, _ = median_predict_latency_ms(student, x_test, "classification", args.repeats, args.latency_batch_rows)

    return {
        "task": "classification",
        "dataset": dataset,
        "seed": seed,
        "teacher_top_k": args.teacher_top_k,
        "teacher_build_seconds": teacher_build_seconds,
        "student_build_seconds": student_build_seconds,
        "teacher_artifact_kb": artifact_kb(teacher),
        "student_artifact_kb": artifact_kb(student),
        "artifact_kb_ratio_student_over_teacher": artifact_kb(student) / artifact_kb(teacher),
        "teacher_latency_ms": t_latency,
        "student_latency_ms": s_latency,
        "latency_batch_rows": batch_rows,
        "latency_ratio_student_over_teacher": s_latency / t_latency if t_latency > 0 else np.nan,
        "teacher_auc": teacher_scores["auc"],
        "student_auc": student_scores["auc"],
        "auc_retention_ratio": student_scores["auc"] / teacher_scores["auc"] if teacher_scores["auc"] else np.nan,
        "auc_delta_student_minus_teacher": student_scores["auc"] - teacher_scores["auc"],
        "teacher_ece": teacher_scores["ece"],
        "student_ece": student_scores["ece"],
        "teacher_brier": teacher_scores["brier"],
        "student_brier": student_scores["brier"],
        "teacher_nll": teacher_scores["nll"],
        "student_nll": student_scores["nll"],
        "validation_candidates_json": json.dumps(val_rows, ensure_ascii=False),
    }


def run_regression(dataset, seed, args):
    x_train, y_train, x_test, y_test = load_regression_dataset(
        dataset, seed, max_train_rows=args.max_train_rows, max_test_rows=args.max_test_rows
    )
    x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=seed)

    teacher_start = time.perf_counter()
    fitted = []
    val_rows = []
    for model_id, model in enumerate(regression_candidates(seed), start=1):
        fitted_model = clone(model)
        fitted_model.fit(x_fit, y_fit)
        val_pred = np.asarray(fitted_model.predict(x_val), dtype=float).ravel()
        score = rmse(y_val, val_pred)
        fitted.append((score, model_id, fitted_model))
        val_rows.append({"candidate_id": model_id, "validation_rmse": score})
    fitted = sorted(fitted, key=lambda x: x[0])
    teacher_members = [m for _, _, m in fitted[: args.teacher_top_k]]
    teacher = MeanRegressorEnsemble(teacher_members)
    teacher_build_seconds = time.perf_counter() - teacher_start

    pseudo_y = teacher.predict(x_fit)
    student = HistGradientBoostingRegressor(max_iter=70, max_leaf_nodes=15, learning_rate=0.08, random_state=seed + 11)
    student_start = time.perf_counter()
    student.fit(x_fit, pseudo_y)
    student_build_seconds = time.perf_counter() - student_start

    teacher_pred = teacher.predict(x_test)
    student_pred = student.predict(x_test)
    teacher_rmse = rmse(y_test, teacher_pred)
    student_rmse = rmse(y_test, student_pred)
    t_latency, batch_rows = median_predict_latency_ms(teacher, x_test, "regression", args.repeats, args.latency_batch_rows)
    s_latency, _ = median_predict_latency_ms(student, x_test, "regression", args.repeats, args.latency_batch_rows)

    return {
        "task": "regression",
        "dataset": dataset,
        "seed": seed,
        "teacher_top_k": args.teacher_top_k,
        "teacher_build_seconds": teacher_build_seconds,
        "student_build_seconds": student_build_seconds,
        "teacher_artifact_kb": artifact_kb(teacher),
        "student_artifact_kb": artifact_kb(student),
        "artifact_kb_ratio_student_over_teacher": artifact_kb(student) / artifact_kb(teacher),
        "teacher_latency_ms": t_latency,
        "student_latency_ms": s_latency,
        "latency_batch_rows": batch_rows,
        "latency_ratio_student_over_teacher": s_latency / t_latency if t_latency > 0 else np.nan,
        "teacher_rmse": teacher_rmse,
        "student_rmse": student_rmse,
        "rmse_ratio_student_over_teacher": student_rmse / teacher_rmse if teacher_rmse > 0 else np.nan,
        "rmse_delta_student_minus_teacher": student_rmse - teacher_rmse,
        "teacher_mae": float(mean_absolute_error(y_test, teacher_pred)),
        "student_mae": float(mean_absolute_error(y_test, student_pred)),
        "prediction_corr_teacher_student": corr(teacher_pred, student_pred),
        "validation_candidates_json": json.dumps(val_rows, ensure_ascii=False),
    }


def summarize(rows: pd.DataFrame):
    summary_rows = []
    for keys, g in rows.groupby(["task"], dropna=False):
        task = keys
        row = {"task": task, "dataset": "ALL", "n": len(g)}
        for col in [
            "teacher_build_seconds",
            "student_build_seconds",
            "teacher_artifact_kb",
            "student_artifact_kb",
            "artifact_kb_ratio_student_over_teacher",
            "teacher_latency_ms",
            "student_latency_ms",
            "latency_ratio_student_over_teacher",
            "auc_retention_ratio",
            "rmse_ratio_student_over_teacher",
            "prediction_corr_teacher_student",
        ]:
            if col in g:
                row[f"mean_{col}"] = g[col].mean()
                row[f"std_{col}"] = g[col].std(ddof=1)
        summary_rows.append(row)
    for keys, g in rows.groupby(["task", "dataset"], dropna=False):
        task, dataset = keys
        row = {"task": task, "dataset": dataset, "n": len(g)}
        for col in [
            "teacher_build_seconds",
            "student_build_seconds",
            "teacher_artifact_kb",
            "student_artifact_kb",
            "artifact_kb_ratio_student_over_teacher",
            "teacher_latency_ms",
            "student_latency_ms",
            "latency_ratio_student_over_teacher",
            "auc_retention_ratio",
            "rmse_ratio_student_over_teacher",
            "prediction_corr_teacher_student",
        ]:
            if col in g:
                row[f"mean_{col}"] = g[col].mean()
                row[f"std_{col}"] = g[col].std(ddof=1)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-datasets", nargs="*", default=["cc1", "credit-g", "ld1"])
    parser.add_argument("--regression-datasets", nargs="*", default=["boston", "concrete", "california"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--teacher-top-k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--latency-batch-rows", type=int, default=1000)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--out-dir", default=str(ROOT / "detailed_results" / "fig5_distill_latency_complexity_probe_20260522"))
    args = parser.parse_args()
    args.max_train_rows = args.max_train_rows or None
    args.max_test_rows = args.max_test_rows or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.classification_datasets:
        for seed in args.seeds:
            print(f"[classification] {dataset} seed={seed}", flush=True)
            rows.append(run_classification(dataset, seed, args))
            pd.DataFrame(rows).to_csv(out_dir / "distill_latency_complexity_raw.csv", index=False, encoding="utf-8-sig")
    for dataset in args.regression_datasets:
        for seed in args.seeds:
            print(f"[regression] {dataset} seed={seed}", flush=True)
            rows.append(run_regression(dataset, seed, args))
            pd.DataFrame(rows).to_csv(out_dir / "distill_latency_complexity_raw.csv", index=False, encoding="utf-8-sig")

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "distill_latency_complexity_raw.csv", index=False, encoding="utf-8-sig")
    summarize(raw).to_csv(out_dir / "distill_latency_complexity_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(raw)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
