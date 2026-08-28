import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fig5_probe_utils import classification_scores, load_classification_dataset, load_regression_dataset


ROOT = Path(__file__).resolve().parents[2]


def rmse(y_true, pred):
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y - p) ** 2)))


def interval_metrics(y_true, lo, hi):
    y = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    width = hi - lo
    y_range = float(np.max(y) - np.min(y)) if len(y) else np.nan
    return {
        "picp": float(np.mean((y >= lo) & (y <= hi))),
        "mpiw": float(np.mean(width)),
        "pinaw": float(np.mean(width) / y_range) if y_range and np.isfinite(y_range) else np.nan,
    }


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def run_classification(dataset, seed, args):
    x_train, y_train, x_test, y_test = load_classification_dataset(
        dataset, seed, max_train_rows=args.max_train_rows, max_test_rows=args.max_test_rows
    )
    strat = y_train if y_train.nunique() > 1 else None
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        x_train, y_train, test_size=0.25, random_state=seed, stratify=strat
    )
    model = RandomForestClassifier(n_estimators=240, random_state=seed, n_jobs=-1, class_weight="balanced")
    model.fit(x_fit, y_fit)
    raw_cal = model.predict_proba(x_cal)[:, 1]
    raw_test = model.predict_proba(x_test)[:, 1]

    candidates = {"raw": (raw_cal, raw_test)}
    platt = LogisticRegression(max_iter=1000)
    platt.fit(logit(raw_cal), y_cal)
    candidates["platt"] = (
        platt.predict_proba(logit(raw_cal))[:, 1],
        platt.predict_proba(logit(raw_test))[:, 1],
    )
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_cal, y_cal)
    candidates["isotonic"] = (isotonic.predict(raw_cal), isotonic.predict(raw_test))

    val_scores = {name: classification_scores(y_cal, cal_prob)["nll"] for name, (cal_prob, _) in candidates.items()}
    selected = min(val_scores, key=val_scores.get)
    selected_test = candidates[selected][1]
    raw_scores = classification_scores(y_test, raw_test)
    cal_scores = classification_scores(y_test, selected_test)
    return {
        "task": "classification",
        "dataset": dataset,
        "seed": seed,
        "selected_calibrator": selected,
        "validation_nll_raw": val_scores["raw"],
        "validation_nll_platt": val_scores["platt"],
        "validation_nll_isotonic": val_scores["isotonic"],
        "auc_raw": raw_scores["auc"],
        "auc_cal": cal_scores["auc"],
        "ece_raw": raw_scores["ece"],
        "ece_cal": cal_scores["ece"],
        "ece_delta_cal_minus_raw": cal_scores["ece"] - raw_scores["ece"],
        "brier_raw": raw_scores["brier"],
        "brier_cal": cal_scores["brier"],
        "brier_delta_cal_minus_raw": cal_scores["brier"] - raw_scores["brier"],
        "nll_raw": raw_scores["nll"],
        "nll_cal": cal_scores["nll"],
        "nll_delta_cal_minus_raw": cal_scores["nll"] - raw_scores["nll"],
    }


def run_regression(dataset, seed, args):
    x_train, y_train, x_test, y_test = load_regression_dataset(
        dataset, seed, max_train_rows=args.max_train_rows, max_test_rows=args.max_test_rows
    )
    x_fit, x_cal, y_fit, y_cal = train_test_split(x_train, y_train, test_size=0.25, random_state=seed)
    model = RandomForestRegressor(n_estimators=240, random_state=seed, n_jobs=-1)
    model.fit(x_fit, y_fit)
    fit_pred = model.predict(x_fit)
    cal_pred = model.predict(x_cal)
    test_pred = model.predict(x_test)

    z = 1.6448536269514722
    raw_width = z * float(np.std(np.asarray(y_fit) - fit_pred, ddof=1))
    raw = interval_metrics(y_test, test_pred - raw_width, test_pred + raw_width)

    q = float(np.quantile(np.abs(np.asarray(y_cal) - cal_pred), 1.0 - args.alpha))
    conformal = interval_metrics(y_test, test_pred - q, test_pred + q)
    return {
        "task": "regression",
        "dataset": dataset,
        "seed": seed,
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "point_rmse": rmse(y_test, test_pred),
        "picp_raw_gaussian": raw["picp"],
        "mpiw_raw_gaussian": raw["mpiw"],
        "pinaw_raw_gaussian": raw["pinaw"],
        "picp_conformal": conformal["picp"],
        "mpiw_conformal": conformal["mpiw"],
        "pinaw_conformal": conformal["pinaw"],
        "coverage_error_raw_abs": abs(raw["picp"] - (1.0 - args.alpha)),
        "coverage_error_conformal_abs": abs(conformal["picp"] - (1.0 - args.alpha)),
    }


def summarize(raw):
    rows = []
    for keys, g in raw.groupby(["task"], dropna=False):
        task = keys
        row = {"task": task, "dataset": "ALL", "n": len(g)}
        for col in raw.columns:
            if col in {"task", "dataset", "seed", "selected_calibrator"}:
                continue
            if pd.api.types.is_numeric_dtype(g[col]):
                row[f"mean_{col}"] = g[col].mean()
                row[f"std_{col}"] = g[col].std(ddof=1)
        rows.append(row)
    for keys, g in raw.groupby(["task", "dataset"], dropna=False):
        task, dataset = keys
        row = {"task": task, "dataset": dataset, "n": len(g)}
        for col in raw.columns:
            if col in {"task", "dataset", "seed", "selected_calibrator"}:
                continue
            if pd.api.types.is_numeric_dtype(g[col]):
                row[f"mean_{col}"] = g[col].mean()
                row[f"std_{col}"] = g[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-datasets", nargs="*", default=["cc1", "credit-g", "ld1"])
    parser.add_argument("--regression-datasets", nargs="*", default=["boston", "concrete", "california"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--out-dir", default=str(ROOT / "detailed_results" / "fig5_calibration_probe_20260522"))
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
            pd.DataFrame(rows).to_csv(out_dir / "calibration_probe_raw.csv", index=False, encoding="utf-8-sig")
    for dataset in args.regression_datasets:
        for seed in args.seeds:
            print(f"[regression] {dataset} seed={seed}", flush=True)
            rows.append(run_regression(dataset, seed, args))
            pd.DataFrame(rows).to_csv(out_dir / "calibration_probe_raw.csv", index=False, encoding="utf-8-sig")

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "calibration_probe_raw.csv", index=False, encoding="utf-8-sig")
    summarize(raw).to_csv(out_dir / "calibration_probe_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(raw)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
