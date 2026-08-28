"""Reduced live Fig. 6b output-control run on autoLOGIC_compare.

The runner calls the live compare feature/model generation functions, captures
teacher/student source outputs once per dataset-seed, then expands calibration,
interval, and clustering post-processing settings offline. It avoids pairwise
sample matrices in clustering consensus.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    brier_score_loss,
    calinski_harabasz_score,
    davies_bouldin_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


WORKSPACE = Path(__file__).resolve().parents[2]
COMPARE_ROOT = WORKSPACE / "autologic"
DATA_ROOT = Path(os.environ.get("AUTOLOGIC_DATA_DIR", WORKSPACE / "data" / "pkl"))
OUT_DEFAULT = "outputs/fig6b_output_control"

CLASSIFICATION_DATASETS = ["cc1", "credit-g"]
REGRESSION_DATASETS = ["boston", "concrete"]
CLUSTERING_DATASETS = ["breast", "glass", "students"]
SEEDS = [42, 44, 46]
ALPHAS = [0.05, 0.10, 0.20]
N_BINS = 10
EPS = 1e-12

CLASS_CONFIGS = [
    ("teacher_stack_no_calibration", "teacher", "none"),
    ("teacher_stack_temperature_scaling", "teacher", "temperature_scaling"),
    ("teacher_stack_platt_sigmoid", "teacher", "platt_sigmoid"),
    ("teacher_stack_isotonic", "teacher", "isotonic"),
    ("distilled_student_no_calibration", "student", "none"),
    ("distilled_student_temperature_scaling", "student", "temperature_scaling"),
    ("distilled_student_platt_sigmoid", "student", "platt_sigmoid"),
    ("distilled_student_isotonic", "student", "isotonic"),
]

REG_CONFIGS = [
    ("teacher_point_prediction", "teacher", "point", None),
    ("teacher_split_conformal_alpha_0_05", "teacher", "split_conformal", 0.05),
    ("teacher_split_conformal_alpha_0_10", "teacher", "split_conformal", 0.10),
    ("teacher_split_conformal_alpha_0_20", "teacher", "split_conformal", 0.20),
    ("student_point_prediction", "student", "point", None),
    ("student_split_conformal_alpha_0_05", "student", "split_conformal", 0.05),
    ("student_split_conformal_alpha_0_10", "student", "split_conformal", 0.10),
    ("student_split_conformal_alpha_0_20", "student", "split_conformal", 0.20),
]

CLUSTER_CONFIGS = [
    ("teacher_no_consensus", "teacher", "best_single", 1, False),
    ("teacher_majority_voting", "teacher", "majority_voting", 3, False),
    ("teacher_weighted_voting", "teacher", "weighted_voting", 3, False),
    ("teacher_coassociation_consensus", "teacher", "coassociation_consensus", None, True),
    ("student_no_consensus", "student", "best_single", 1, False),
    ("student_majority_voting", "student", "majority_voting", 3, False),
    ("student_weighted_voting", "student", "weighted_voting", 3, False),
    ("student_coassociation_consensus", "student", "coassociation_consensus", None, True),
]


def load_module(name: str, path: Path):
    sys.path.insert(0, str(COMPARE_ROOT))
    sys.path.insert(0, str(WORKSPACE))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CLS = load_module(
    "compare_classification_live",
    COMPARE_ROOT / "ensemble" / "classification_ensemble" / "classification_auto_ensemble.py",
)
REG = load_module(
    "compare_regression_live",
    COMPARE_ROOT / "ensemble" / "regression_ensemble" / "regression_auto_ensemble.py",
)
CLU = load_module(
    "compare_clustering_live",
    COMPARE_ROOT / "ensemble" / "cluster_ensemble" / "cluster_auto_ensemble_cached_labels.py",
)


def ensure_api() -> Tuple[str, str]:
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY must be set.")
    return base_url, api_key


def finite_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 2:
        p = p[:, 1] if p.shape[1] > 1 else p[:, 0]
    return np.clip(p.reshape(-1), 1e-6, 1.0 - 1e-6)


def logit(p: np.ndarray) -> np.ndarray:
    p = finite_prob(p)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def ece_binary(y_true: np.ndarray, prob: np.ndarray, n_bins: int = N_BINS) -> float:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    prob = finite_prob(prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(prob, bins[1:-1], right=True)
    out = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        out += float(np.mean(mask)) * abs(float(np.mean(prob[mask])) - float(np.mean(y_true[mask])))
    return float(out)


def reliability_bins(dataset: str, seed: int, config: str, y_true: np.ndarray, prob: np.ndarray) -> List[Dict[str, Any]]:
    y_true = np.asarray(y_true).astype(int)
    prob = finite_prob(prob)
    bins = np.linspace(0.0, 1.0, N_BINS + 1)
    ids = np.digitize(prob, bins[1:-1], right=True)
    rows = []
    for b in range(N_BINS):
        mask = ids == b
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "config": config,
                "bin_id": b,
                "bin_left": float(bins[b]),
                "bin_right": float(bins[b + 1]),
                "bin_count": int(np.sum(mask)),
                "mean_confidence": float(np.mean(prob[mask])) if np.any(mask) else np.nan,
                "empirical_accuracy": float(np.mean(y_true[mask])) if np.any(mask) else np.nan,
            }
        )
    return rows


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    prob = finite_prob(prob)
    pred = (prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "acc": float(accuracy_score(y_true, pred)),
        "ece": ece_binary(y_true, prob),
        "brier": float(brier_score_loss(y_true, prob)),
        "nll": float(log_loss(y_true, np.column_stack([1.0 - prob, prob]), labels=[0, 1])),
    }


def calibrate_prob(method: str, cal_prob: np.ndarray, y_cal: np.ndarray, test_prob: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    cal_prob = finite_prob(cal_prob)
    test_prob = finite_prob(test_prob)
    if method == "none":
        return test_prob, {"calibration_temperature": np.nan}
    if method == "temperature_scaling":
        z_cal = logit(cal_prob)
        z_test = logit(test_prob)

        def objective(log_t: float) -> float:
            t = float(np.exp(log_t))
            p = sigmoid(z_cal / t)
            return log_loss(y_cal, np.column_stack([1.0 - p, p]), labels=[0, 1])

        res = minimize_scalar(objective, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
        temp = float(np.exp(res.x)) if res.success else 1.0
        return sigmoid(z_test / temp), {"calibration_temperature": temp}
    if method == "platt_sigmoid":
        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(logit(cal_prob).reshape(-1, 1), y_cal)
        return finite_prob(lr.predict_proba(logit(test_prob).reshape(-1, 1))), {"calibration_temperature": np.nan}
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(cal_prob, y_cal)
        return finite_prob(iso.predict(test_prob)), {"calibration_temperature": np.nan}
    raise ValueError(method)


def conformal_qhat(residuals: np.ndarray, alpha: float) -> float:
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    if n == 0:
        return np.nan
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(residuals, level, method="higher"))


def interval_metrics(y_true: np.ndarray, pred: np.ndarray, qhat: float, alpha: float) -> Dict[str, float]:
    lower = pred - qhat
    upper = pred + qhat
    width = upper - lower
    picp = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mpiw = float(np.mean(width))
    yr = float(np.max(y_true) - np.min(y_true))
    pinaw = float(mpiw / yr) if yr > EPS else np.nan
    target = 1.0 - alpha
    return {
        "picp": picp,
        "mpiw": mpiw,
        "pinaw": pinaw,
        "coverage_error": abs(picp - target),
        "target_coverage": target,
        "qhat": qhat,
    }


def reg_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
    }


def rename_class(code: str, prefix: str, new_name: str) -> str:
    return re.sub(rf"class\s+{prefix}[_\w]*\s*(\([^)]*\))?\s*:", f"class {new_name}:", code, count=1)


def class_from_code(mod: Any, code: str, class_name: str):
    err, scope = mod.code_exec(code)
    if err is not None:
        raise RuntimeError(err)
    return scope[class_name]


def fit_predict_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("model lacks predict_proba")
    return finite_prob(model.predict_proba(x))


def generate_class_models(
    dataset: str,
    seed: int,
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, List[Any], List[Any], Dict[str, Any]]:
    base_url, api_key = ensure_api()
    df_train, df_test, target, desc = CLS.load_origin_data(dataset, seed=seed)
    baseline = CLS.base_model(seed)
    if args.feat_iterations > 0:
        df_train, df_test = CLS.generate_feat(
            baseline,
            df_train,
            df_test,
            dataset,
            round_num=1,
            llm_model=args.llm,
            iterations=args.feat_iterations,
            target_column_name=target,
            dataset_description=desc,
            base_url=base_url,
            api_key=api_key,
        )
    train_main, hold = train_test_split(df_train, test_size=0.35, random_state=seed, stratify=df_train[target])
    meta_df, cal_df = train_test_split(hold, test_size=0.5, random_state=seed + 17, stratify=hold[target])
    x_train, y_train = CLS.to_pd(train_main, target)
    x_meta, y_meta = CLS.to_pd(meta_df, target)
    x_cal, y_cal = CLS.to_pd(cal_df, target)
    x_test, y_test = CLS.to_pd(df_test, target)
    prompt = CLS.get_model_prompt(target_column_name=target, samples=CLS.build_prompt_samples(train_main))
    messages = [
        {"role": "system", "content": "You are a top-level machine learning classification expert. Output only valid Python code for class myclassifier with fit, predict, and predict_proba."},
        {"role": "user", "content": prompt},
    ]
    teacher_models: List[Any] = []
    student_models: List[Any] = []
    token_count = 0
    call_count = 0
    for i in range(args.class_model_iterations):
        res = CLS.generate_model_2(args.llm, messages, base_url, api_key)
        call_count += 1
        token_count += int(res.get("total_tokens") or 0)
        code = rename_class(CLS.clean_llm_code(res["code"]), "myclassifier", f"myclassifier_{i+1}")
        try:
            model = class_from_code(CLS, code, f"myclassifier_{i+1}")()
            model.fit(x_train, y_train)
            best_model = deepcopy(model)
            best_auc = roc_auc_score(y_meta, fit_predict_proba(best_model, x_meta))
            if args.param_iterations > 0:
                p_prompt = CLS.get_classification_param_prompt(code, best_auc * 100.0, desc, x_meta, list(x_train.columns), dataset, max_rows=10)
                p_messages = [
                    {"role": "system", "content": "Optimize hyperparameters only. Output Python code only for class myclassifier."},
                    {"role": "user", "content": p_prompt},
                ]
                res_p = CLS.generate_model_2(args.llm, p_messages, base_url, api_key)
                call_count += 1
                token_count += int(res_p.get("total_tokens") or 0)
                p_code = rename_class(CLS.clean_llm_code(res_p["code"]), "myclassifier", f"myclassifier_{i+1}_param_1")
                try:
                    p_model = class_from_code(CLS, p_code, f"myclassifier_{i+1}_param_1")()
                    p_model.fit(x_train, y_train)
                    p_auc = roc_auc_score(y_meta, fit_predict_proba(p_model, x_meta))
                    if p_auc >= best_auc:
                        best_model = deepcopy(p_model)
                        code = p_code
                        best_auc = p_auc
                except Exception:
                    pass
            teacher_models.append(best_model)
            try:
                student = CLS.distill_to_student(best_model, x_train, y_train, x_meta, y_meta, device="cpu", epochs=args.distill_epochs)
                student_models.append(student)
            except Exception:
                student_models.append(best_model)
            messages += [
                {"role": "assistant", "content": code},
                {"role": "user", "content": f"Current validation AUC is {best_auc:.4f}. Generate a different classifier likely to improve it. Output code only."},
            ]
        except Exception as exc:
            messages += [
                {"role": "assistant", "content": code},
                {"role": "user", "content": f"Code failed with {type(exc).__name__}: {exc}. Fix it and output code only."},
            ]
    meta = {"llm_calls": call_count, "llm_tokens_logged": token_count, "valid_teacher_models": len(teacher_models), "valid_student_models": len(student_models)}
    return x_meta, x_cal, x_test, y_meta, y_cal, y_test, teacher_models, student_models, meta


def stacked_classifier_probs(models: List[Any], x_meta: pd.DataFrame, y_meta: pd.Series, x_cal: pd.DataFrame, x_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if not models:
        raise RuntimeError("no classification models")
    meta_train = np.column_stack([fit_predict_proba(m, x_meta) for m in models])
    meta_cal = np.column_stack([fit_predict_proba(m, x_cal) for m in models])
    meta_test = np.column_stack([fit_predict_proba(m, x_test) for m in models])
    if meta_train.shape[1] == 1:
        return finite_prob(meta_cal[:, 0]), finite_prob(meta_test[:, 0])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(meta_train, y_meta)
    return finite_prob(clf.predict_proba(meta_cal)), finite_prob(clf.predict_proba(meta_test))


def run_class_source(dataset: str, seed: int, args: argparse.Namespace, out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    x_meta, x_cal, x_test, y_meta, y_cal, y_test, t_models, s_models, run_meta = generate_class_models(dataset, seed, args, out_dir)
    source = {
        "teacher": stacked_classifier_probs(t_models, x_meta, y_meta, x_cal, x_test),
        "student": stacked_classifier_probs(s_models, x_meta, y_meta, x_cal, x_test),
    }
    rows: List[Dict[str, Any]] = []
    bins: List[Dict[str, Any]] = []
    y_cal_arr = np.asarray(y_cal).astype(int)
    y_test_arr = np.asarray(y_test).astype(int)
    for config, family, method in CLASS_CONFIGS:
        cal_prob, test_prob = source[family]
        final_prob, cal_meta = calibrate_prob(method, cal_prob, y_cal_arr, test_prob)
        row = {
            "task": "classification",
            "dataset": dataset,
            "seed": seed,
            "config": config,
            "model_family": family,
            "calibration_method": method,
            "n_meta": len(y_meta),
            "n_calibration": len(y_cal),
            "n_test": len(y_test),
            "failed_flag": 0,
            "failure_reason": "",
            **classification_metrics(y_test_arr, final_prob),
            **cal_meta,
        }
        rows.append(row)
        bins.extend(reliability_bins(dataset, seed, config, y_test_arr, final_prob))
    return rows, bins, run_meta


def generate_reg_models(dataset: str, seed: int, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, List[Any], List[Any], Dict[str, Any]]:
    base_url, api_key = ensure_api()
    loc = str(DATA_ROOT / f"{dataset}.pkl")
    df_train, df_test, target, desc = REG.load_origin_data(loc, seed)
    baseline = REG.base_model(seed)
    if args.feat_iterations > 0:
        df_train, df_test = REG.generate_feat(baseline, df_train, df_test, dataset, 1, args.llm, args.feat_iterations, target, desc, "regression", base_url, api_key)
    train_main, hold = train_test_split(df_train, test_size=0.35, random_state=seed)
    meta_df, cal_df = train_test_split(hold, test_size=0.5, random_state=seed + 17)
    x_train, y_train = REG.to_pd(train_main, target)
    x_meta, y_meta = REG.to_pd(meta_df, target)
    x_cal, y_cal = REG.to_pd(cal_df, target)
    x_test, y_test = REG.to_pd(df_test, target)
    prompt = REG.get_regression_model_prompt(target_column_name=target, samples=REG.build_prompt_samples(train_main))
    messages = [
        {"role": "system", "content": "You are a top-level regression expert. Output only valid Python code for class myregressor with fit and predict."},
        {"role": "user", "content": prompt},
    ]
    teacher_models: List[Any] = []
    student_models: List[Any] = []
    token_count = 0
    call_count = 0
    for i in range(args.reg_model_iterations):
        res = REG.generate_model_2(args.llm, messages, base_url, api_key)
        call_count += 1
        token_count += int(res.get("total_tokens") or 0)
        code = rename_class(REG.clean_llm_code(res["code"]), "myregressor", f"myregressor_{i+1}")
        try:
            model = class_from_code(REG, code, f"myregressor_{i+1}")()
            model.fit(x_train, y_train)
            best_model = deepcopy(model)
            best_rmse = math.sqrt(mean_squared_error(y_meta, np.asarray(best_model.predict(x_meta), dtype=float)))
            if args.param_iterations > 0:
                p_prompt = f"Current best RMSE: {best_rmse:.4f}\nCode:\n```python\n{code}\n```\nOptimize hyperparameters only. Output code only."
                res_p = REG.generate_model_2(args.llm, [{"role": "system", "content": "Optimize regressor hyperparameters only."}, {"role": "user", "content": p_prompt}], base_url, api_key)
                call_count += 1
                token_count += int(res_p.get("total_tokens") or 0)
                p_code = rename_class(REG.clean_llm_code(res_p["code"]), "myregressor", f"myregressor_{i+1}_param_1")
                try:
                    p_model = class_from_code(REG, p_code, f"myregressor_{i+1}_param_1")()
                    p_model.fit(x_train, y_train)
                    p_rmse = math.sqrt(mean_squared_error(y_meta, np.asarray(p_model.predict(x_meta), dtype=float)))
                    if p_rmse <= best_rmse:
                        best_model = deepcopy(p_model)
                        code = p_code
                        best_rmse = p_rmse
                except Exception:
                    pass
            teacher_models.append(best_model)
            try:
                student = REG.distill_to_student_regression(best_model, x_train, y_train, x_meta, y_meta, device="cpu", epochs=args.distill_epochs)
                student_models.append(student)
            except Exception:
                student_models.append(best_model)
            messages += [
                {"role": "assistant", "content": code},
                {"role": "user", "content": f"Current validation RMSE is {best_rmse:.4f}. Generate a different regressor likely to improve it. Output code only."},
            ]
        except Exception as exc:
            messages += [
                {"role": "assistant", "content": code},
                {"role": "user", "content": f"Code failed with {type(exc).__name__}: {exc}. Fix it and output code only."},
            ]
    meta = {"llm_calls": call_count, "llm_tokens_logged": token_count, "valid_teacher_models": len(teacher_models), "valid_student_models": len(student_models)}
    return x_meta, x_cal, x_test, y_meta, y_cal, y_test, teacher_models, student_models, meta


def stacked_regression_pred(models: List[Any], x_meta: pd.DataFrame, y_meta: pd.Series, x_cal: pd.DataFrame, x_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if not models:
        raise RuntimeError("no regression models")
    meta_train = np.column_stack([np.asarray(m.predict(x_meta), dtype=float) for m in models])
    meta_cal = np.column_stack([np.asarray(m.predict(x_cal), dtype=float) for m in models])
    meta_test = np.column_stack([np.asarray(m.predict(x_test), dtype=float) for m in models])
    if meta_train.shape[1] == 1:
        return meta_cal[:, 0], meta_test[:, 0]
    model = Ridge(alpha=1.0)
    model.fit(meta_train, y_meta)
    return np.asarray(model.predict(meta_cal), dtype=float), np.asarray(model.predict(meta_test), dtype=float)


def run_reg_source(dataset: str, seed: int, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    x_meta, x_cal, x_test, y_meta, y_cal, y_test, t_models, s_models, run_meta = generate_reg_models(dataset, seed, args)
    source = {
        "teacher": stacked_regression_pred(t_models, x_meta, y_meta, x_cal, x_test),
        "student": stacked_regression_pred(s_models, x_meta, y_meta, x_cal, x_test),
    }
    y_cal_arr = np.asarray(y_cal, dtype=float)
    y_test_arr = np.asarray(y_test, dtype=float)
    rows: List[Dict[str, Any]] = []
    curve: List[Dict[str, Any]] = []
    for config, family, method, alpha in REG_CONFIGS:
        cal_pred, test_pred = source[family]
        row = {
            "task": "regression",
            "dataset": dataset,
            "seed": seed,
            "config": config,
            "model_family": family,
            "interval_method": method,
            "alpha": alpha,
            "n_meta": len(y_meta),
            "n_calibration": len(y_cal),
            "n_test": len(y_test),
            "failed_flag": 0,
            "failure_reason": "",
            **reg_metrics(y_test_arr, test_pred),
            "picp": np.nan,
            "mpiw": np.nan,
            "pinaw": np.nan,
            "coverage_error": np.nan,
            "target_coverage": np.nan,
            "qhat": np.nan,
        }
        if method == "split_conformal":
            qhat = conformal_qhat(np.abs(y_cal_arr - cal_pred), float(alpha))
            im = interval_metrics(y_test_arr, test_pred, qhat, float(alpha))
            row.update(im)
            curve.append({"dataset": dataset, "seed": seed, "config": config, "model_family": family, "alpha": alpha, **im})
        rows.append(row)
    return rows, curve, run_meta


@dataclass
class LabelRecord:
    name: str
    labels: np.ndarray
    ari: float
    nmi: float
    selected_k: int
    silhouette: float
    db: float
    ch: float
    source: str


def cluster_internal(x: pd.DataFrame, labels: np.ndarray) -> Tuple[float, float, float, int]:
    labels = np.asarray(labels).reshape(-1)
    k = len(np.unique(labels))
    if k < 2 or k >= len(labels):
        return np.nan, np.nan, np.nan, 1
    try:
        return float(silhouette_score(x, labels)), float(davies_bouldin_score(x, labels)), float(calinski_harabasz_score(x, labels)), 0
    except Exception:
        return np.nan, np.nan, np.nan, 1


def make_label_record(name: str, labels: np.ndarray, y_true: np.ndarray, x: pd.DataFrame, source: str) -> LabelRecord:
    labels = np.asarray(labels).reshape(-1)
    sil, db, ch, _ = cluster_internal(x, labels)
    return LabelRecord(
        name=name,
        labels=labels,
        ari=float(adjusted_rand_score(y_true, labels)),
        nmi=float(normalized_mutual_info_score(y_true, labels)),
        selected_k=int(len(np.unique(labels))),
        silhouette=sil,
        db=db,
        ch=ch,
        source=source,
    )


def map_to_reference(reference: np.ndarray, labels: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference)
    labels = np.asarray(labels)
    out = np.empty_like(reference)
    refs = np.unique(reference)
    for lv in np.unique(labels):
        mask = labels == lv
        best = max(refs, key=lambda rv: int(np.sum(mask & (reference == rv))))
        out[mask] = best
    return out


def internal_score_values(labels: np.ndarray, sil: float, db: float, ch: float, expected_k: int) -> Tuple[float, float, float, float]:
    labels = np.asarray(labels).reshape(-1)
    k = int(len(np.unique(labels)))
    k_penalty = -abs(k - int(expected_k))
    sil_score = np.nan_to_num(sil, nan=-999.0)
    db_score = np.nan_to_num(db, nan=999.0)
    ch_score = np.nan_to_num(ch, nan=-999.0)
    return float(k_penalty), float(sil_score), float(-db_score), float(ch_score)


def label_selection_score(rec: LabelRecord, expected_k: int) -> Tuple[float, float, float, float]:
    return internal_score_values(rec.labels, rec.silhouette, rec.db, rec.ch, expected_k)


def k_compatible_records(records: List[LabelRecord], expected_k: int) -> List[LabelRecord]:
    valid = [r for r in records if r.selected_k > 1 and np.isfinite(r.silhouette)]
    if not valid:
        valid = [r for r in records if r.selected_k > 1]
    if not valid:
        return []
    exact = [r for r in valid if r.selected_k == int(expected_k)]
    if exact:
        return exact
    nearest_dist = min(abs(r.selected_k - int(expected_k)) for r in valid)
    return [r for r in valid if abs(r.selected_k - int(expected_k)) == nearest_dist]


def label_selection_score_v1(rec: LabelRecord) -> Tuple[float, float, float]:
    sil = np.nan_to_num(rec.silhouette, nan=-999.0)
    db = np.nan_to_num(rec.db, nan=999.0)
    ch = np.nan_to_num(rec.ch, nan=-999.0)
    return float(sil), float(-db), float(ch)


def select_records(records: List[LabelRecord], top_k: int, expected_k: int, target_k: Optional[int] = None) -> List[LabelRecord]:
    valid = k_compatible_records(records, expected_k)
    if target_k is not None:
        same_k = [r for r in valid if r.selected_k == int(target_k)]
        if same_k:
            valid = same_k
    valid.sort(key=lambda r: label_selection_score(r, expected_k), reverse=True)
    return valid[: max(1, min(top_k, len(valid)))] if valid else []


def vote_labels(records: List[LabelRecord], weighted: bool, expected_k: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    anchor = select_records(records, 1, expected_k)
    if not anchor:
        raise RuntimeError("no valid label records")
    selected = select_records(records, 3, expected_k, target_k=anchor[0].selected_k)
    if not selected:
        raise RuntimeError("no valid label records")
    reference = selected[0].labels
    values = np.unique(reference)
    value_to_idx = {v: i for i, v in enumerate(values)}
    votes = np.zeros((len(reference), len(values)), dtype=float)
    for rec in selected:
        aligned = map_to_reference(reference, rec.labels)
        weight = max(float(rec.silhouette) + 1.0, 0.05) if weighted and np.isfinite(rec.silhouette) else 1.0
        for i, v in enumerate(aligned):
            votes[i, value_to_idx[v]] += weight
    labels = values[np.argmax(votes, axis=1)]
    return labels, {
        "selected_model_count": len(selected),
        "selected_model_names": [r.name for r in selected],
        "selected_model_ari": [r.ari for r in selected],
        "selected_model_silhouette": [r.silhouette for r in selected],
        "selection_policy": "expected_k_first_same_k_voting_internal_metrics",
        "selection_expected_k": int(expected_k),
        "selection_target_k": int(len(np.unique(reference))),
    }


def best_single_labels(records: List[LabelRecord], expected_k: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    selected = select_records(records, 1, expected_k)
    if not selected:
        raise RuntimeError("no valid label records")
    r = selected[0]
    return r.labels, {
        "selected_model_count": 1,
        "selected_model_names": [r.name],
        "selected_model_ari": [r.ari],
        "selected_model_silhouette": [r.silhouette],
        "selection_policy": "expected_k_first_internal_metrics",
        "selection_expected_k": int(expected_k),
        "selection_target_k": int(r.selected_k),
    }


def pseudo_label_student(labels: np.ndarray, x: pd.DataFrame, seed: int) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if len(np.unique(labels)) < 2:
        raise ValueError("teacher pseudo-labels contain fewer than two clusters")
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=250,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
        ),
    )
    clf.fit(np.asarray(x), y)
    pred = clf.predict(np.asarray(x))
    return enc.inverse_transform(pred)


def cluster_row(dataset: str, seed: int, config: str, family: str, method: str, labels: Optional[np.ndarray], y_true: np.ndarray, x: pd.DataFrame, meta: Dict[str, Any], failed: int = 0, reason: str = "") -> Dict[str, Any]:
    if labels is None:
        return {
            "task": "clustering",
            "dataset": dataset,
            "seed": seed,
            "config": config,
            "model_family": family,
            "postprocessing_method": method,
            "selected_k": np.nan,
            "ari": np.nan,
            "nmi": np.nan,
            "silhouette": np.nan,
            "db": np.nan,
            "ch": np.nan,
            "failed_flag": 1,
            "failure_reason": reason,
            **meta,
        }
    labels = np.asarray(labels).reshape(-1)
    sil, db, ch, internal_failed = cluster_internal(x, labels)
    row_failed = int(failed or internal_failed)
    row_reason = reason or ("invalid_cluster_count_or_internal_metric_failure" if internal_failed else "")
    return {
        "task": "clustering",
        "dataset": dataset,
        "seed": seed,
        "config": config,
        "model_family": family,
        "postprocessing_method": method,
        "selected_k": int(len(np.unique(labels))),
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
        "silhouette": sil,
        "db": db,
        "ch": ch,
        "failed_flag": row_failed,
        "failure_reason": row_reason,
        **meta,
    }


def run_cluster_source(dataset: str, seed: int, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    base_url, api_key = ensure_api()
    loc = str(DATA_ROOT / f"{dataset}.pkl")
    df, n_clusters, target, desc = CLU.load_origin_data(loc)
    baseline = CLU.base_model(n_clusters, seed)
    x_aug, y_true = CLU.generate_feat(baseline, df, dataset, 1, args.llm, args.feat_iterations, target, desc, "clustering", base_url, api_key)
    y_true = np.asarray(y_true).reshape(-1)
    prompt = CLU.get_clustering_model_prompt(samples=CLU.build_prompt_samples(x_aug), n_clusters=n_clusters)
    messages = [
        {"role": "system", "content": "You are a top-level clustering expert. Output only valid Python code for class mycluster with fit_predict."},
        {"role": "user", "content": prompt},
    ]
    teacher_records: List[LabelRecord] = []
    token_count = 0
    call_count = 0
    for i in range(args.cluster_model_iterations):
        raw = CLU.generate_model(args.llm, messages, base_url, api_key)
        call_count += 1
        code = rename_class(CLU.clean_llm_code(raw), "mycluster", f"mycluster_{i+1}")
        try:
            model = CLU._instantiate_cluster_model(class_from_code(CLU, code, f"mycluster_{i+1}"), n_clusters)
            labels = CLU._validate_cluster_labels(model.fit_predict(x_aug), x_aug.shape[0])
            best_model = deepcopy(model)
            best_labels = np.asarray(labels).copy()
            best_sil, best_db, best_ch, best_failed = cluster_internal(x_aug, best_labels)
            best_score = internal_score_values(best_labels, best_sil, best_db, best_ch, n_clusters)
            if args.param_iterations > 0:
                p_prompt = (
                    "Improve only the clustering hyperparameters or preprocessing choices in the following class. "
                    "Ground-truth labels are unavailable during model selection. Use scalable operations only; do not build sample-by-sample similarity matrices. "
                    f"Current internal diagnostics: selected_k={len(np.unique(best_labels))}, silhouette={best_sil:.6f}, "
                    f"Davies-Bouldin={best_db:.6f}, Calinski-Harabasz={best_ch:.6f}. "
                    f"Expected cluster count is approximately {n_clusters}. Dataset description: {desc}. "
                    "Return only Python code defining class mycluster with fit and fit_predict.\n\n"
                    f"{code}"
                )
                raw_p = CLU.generate_model(args.llm, [{"role": "system", "content": "Optimize clustering hyperparameters only. Output code only."}, {"role": "user", "content": p_prompt}], base_url, api_key)
                call_count += 1
                p_code = rename_class(CLU.clean_llm_code(raw_p), "mycluster", f"mycluster_{i+1}_param_1")
                try:
                    p_model = CLU._instantiate_cluster_model(class_from_code(CLU, p_code, f"mycluster_{i+1}_param_1"), n_clusters)
                    p_labels = CLU._validate_cluster_labels(p_model.fit_predict(x_aug), x_aug.shape[0])
                    p_sil, p_db, p_ch, p_failed = cluster_internal(x_aug, p_labels)
                    p_score = internal_score_values(p_labels, p_sil, p_db, p_ch, n_clusters)
                    if (not p_failed) and p_score >= best_score:
                        best_model = deepcopy(p_model)
                        best_labels = np.asarray(p_labels).copy()
                        best_sil, best_db, best_ch = p_sil, p_db, p_ch
                        best_score = p_score
                        code = p_code
                except Exception:
                    pass
            teacher_records.append(make_label_record(f"teacher_{i+1}", best_labels, y_true, x_aug, "teacher"))
            messages += [
                {"role": "assistant", "content": code},
                {"role": "user", "content": f"Current internal diagnostics are silhouette={best_sil:.4f}, Davies-Bouldin={best_db:.4f}, Calinski-Harabasz={best_ch:.4f}. Generate a different stable clustering model. Output code only."},
            ]
        except Exception as exc:
            messages += [
                {"role": "assistant", "content": code},
                {"role": "user", "content": f"Code failed with {type(exc).__name__}: {exc}. Fix and output code only."},
            ]
    if not teacher_records:
        for idx, fb in enumerate(CLU.fallback_models(n_clusters=n_clusters, seed=seed), start=1):
            try:
                teacher_records.append(make_label_record(f"teacher_fallback_{idx}", fb.fit_predict(x_aug), y_true, x_aug, "teacher_fallback"))
            except Exception:
                pass
    student_records: List[LabelRecord] = []
    for idx, rec in enumerate(teacher_records, start=1):
        try:
            labels = CLU._validate_cluster_labels(pseudo_label_student(rec.labels, x_aug, seed + idx), x_aug.shape[0])
            student_records.append(make_label_record(f"student_{idx}", labels, y_true, x_aug, "student"))
        except Exception:
            student_records.append(make_label_record(f"student_fallback_{idx}", rec.labels, y_true, x_aug, "student_fallback"))
    rows: List[Dict[str, Any]] = []
    label_summary: List[Dict[str, Any]] = []
    for rec in teacher_records + student_records:
        label_summary.append({
            "dataset": dataset,
            "seed": seed,
            "name": rec.name,
            "source": rec.source,
            "selected_k": rec.selected_k,
            "ari": rec.ari,
            "nmi": rec.nmi,
            "silhouette": rec.silhouette,
            "db": rec.db,
            "ch": rec.ch,
        })
    for config, family, method, top_k, disabled in CLUSTER_CONFIGS:
        records = teacher_records if family == "teacher" else student_records
        if disabled:
            rows.append(cluster_row(dataset, seed, config, family, method, None, y_true, x_aug, {"disabled_reason": "pairwise_sample_matrix_disabled"}, 1, "disabled_pairwise_sample_matrix"))
            continue
        try:
            if method == "best_single":
                labels, meta = best_single_labels(records, n_clusters)
            elif method == "majority_voting":
                labels, meta = vote_labels(records, weighted=False, expected_k=n_clusters)
            elif method == "weighted_voting":
                labels, meta = vote_labels(records, weighted=True, expected_k=n_clusters)
            else:
                raise ValueError(method)
            if len(np.unique(labels)) < 2:
                fallback, fb_meta = best_single_labels(records, n_clusters)
                meta.update({"raw_selected_k": int(len(np.unique(labels))), "fallback_used": True, "fallback_reason": "single_cluster_vote"})
                meta.update({f"fallback_{k}": v for k, v in fb_meta.items()})
                labels = fallback
                rows.append(cluster_row(dataset, seed, config, family, method, labels, y_true, x_aug, meta, 1, "single_cluster_vote_fallback_to_best_single"))
            else:
                meta.update({"fallback_used": False, "requested_top_k": top_k})
                rows.append(cluster_row(dataset, seed, config, family, method, labels, y_true, x_aug, meta))
        except Exception as exc:
            rows.append(cluster_row(dataset, seed, config, family, method, None, y_true, x_aug, {}, 1, f"{type(exc).__name__}: {exc}"))
    run_meta = {"llm_calls": call_count, "llm_tokens_logged": token_count, "valid_teacher_models": len(teacher_records), "valid_student_models": len(student_records)}
    return rows, label_summary, run_meta


def failed_rows(task: str, dataset: str, seed: int, reason: str) -> List[Dict[str, Any]]:
    if task == "classification":
        return [{"task": task, "dataset": dataset, "seed": seed, "config": c, "model_family": f, "calibration_method": m, "failed_flag": 1, "failure_reason": reason} for c, f, m in CLASS_CONFIGS]
    if task == "regression":
        return [{"task": task, "dataset": dataset, "seed": seed, "config": c, "model_family": f, "interval_method": im, "alpha": a, "failed_flag": 1, "failure_reason": reason} for c, f, im, a in REG_CONFIGS]
    return [{"task": task, "dataset": dataset, "seed": seed, "config": c, "model_family": f, "postprocessing_method": m, "failed_flag": 1, "failure_reason": reason} for c, f, m, _, _ in CLUSTER_CONFIGS]


def aggregate(df: pd.DataFrame, groups: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for key, g in df.groupby(groups, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {k: v for k, v in zip(groups, key)}
        row["n_rows"] = int(len(g))
        row["n_seeds"] = int(g["seed"].nunique()) if "seed" in g else 0
        row["failed_count"] = int(g.get("failed_flag", pd.Series(dtype=float)).fillna(0).sum())
        for m in metrics:
            vals = pd.to_numeric(g.get(m, pd.Series(dtype=float)), errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else np.nan)
            row[f"{m}_n"] = int(len(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(out_dir: Path, tables: Dict[str, pd.DataFrame], manifest: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_dir / "fig6_panel_b_source_data.xlsx", engine="openpyxl") as writer:
        sheet_map = {
            "classification_per_seed": "class_per_seed",
            "classification_summary": "class_summary",
            "classification_reliability_bins": "class_bins",
            "regression_per_seed": "reg_per_seed",
            "regression_summary": "reg_summary",
            "regression_interval_curve": "reg_curve",
            "clustering_per_seed": "cluster_per_seed",
            "clustering_summary": "cluster_summary",
            "clustering_label_records": "cluster_label_records",
            "live_run_records": "live_run_records",
            "failed_runs": "failed_runs",
            "api_call_summary": "api_call_summary",
        }
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_map.get(name, name[:31]), index=False)
    (out_dir / "configs_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = f"""# Fig. 6b reduced live compare run

Generated: {datetime.now().isoformat(timespec='seconds')}

Scope: reduced live source validation for Fig. 6b output-control behavior.

Live source runs:
- classification: {CLASSIFICATION_DATASETS} x seeds {SEEDS}
- regression: {REGRESSION_DATASETS} x seeds {SEEDS}
- clustering: {CLUSTERING_DATASETS} x seeds {SEEDS}

The runner calls live feature/model generation once per dataset-seed, captures
teacher/student source outputs, and expands output-control settings offline.
Pairwise sample-matrix consensus is disabled; clustering reports best-single,
majority voting, weighted voting, and disabled coassociation rows.
For clustering, ground-truth labels are used only for final offline ARI/NMI.
Model refinement, top-k selection, and voting use expected-k-first internal
metrics and cached labels; student clustering uses teacher pseudo-label
distillation.
"""
    (out_dir / "run_readme.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUT_DEFAULT)
    parser.add_argument("--llm", default="gpt-4o-mini")
    parser.add_argument("--tasks", default="classification,regression,clustering", help="Comma-separated subset of classification, regression, clustering.")
    parser.add_argument("--feat-iterations", type=int, default=1)
    parser.add_argument("--class-model-iterations", type=int, default=2)
    parser.add_argument("--reg-model-iterations", type=int, default=2)
    parser.add_argument("--cluster-model-iterations", type=int, default=5)
    parser.add_argument("--param-iterations", type=int, default=1)
    parser.add_argument("--distill-epochs", type=int, default=20)
    args = parser.parse_args()
    task_set = {t.strip() for t in args.tasks.split(",") if t.strip()}

    out_dir = Path(args.output).expanduser()
    if not out_dir.is_absolute():
        out_dir = WORKSPACE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    class_rows: List[Dict[str, Any]] = []
    class_bins: List[Dict[str, Any]] = []
    reg_rows: List[Dict[str, Any]] = []
    reg_curve: List[Dict[str, Any]] = []
    cluster_rows: List[Dict[str, Any]] = []
    cluster_labels: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    start = time.time()

    if "classification" in task_set:
        for dataset in CLASSIFICATION_DATASETS:
            for seed in SEEDS:
                t0 = time.time()
                print(f"[classification] {dataset} seed={seed}", flush=True)
                try:
                    rows, bins, meta = run_class_source(dataset, seed, args, out_dir)
                    class_rows.extend(rows)
                    class_bins.extend(bins)
                    status = "success"
                    reason = ""
                except Exception as exc:
                    status = "failed"
                    reason = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                    class_rows.extend(failed_rows("classification", dataset, seed, reason))
                    failed.append({"task": "classification", "dataset": dataset, "seed": seed, "failure_reason": reason})
                    meta = {}
                run_records.append({"task": "classification", "dataset": dataset, "seed": seed, "status": status, "seconds": time.time() - t0, "failure_reason": reason, **meta})
                write_outputs(out_dir, make_tables(class_rows, class_bins, reg_rows, reg_curve, cluster_rows, cluster_labels, run_records, failed), build_manifest(args, start))

    if "regression" in task_set:
        for dataset in REGRESSION_DATASETS:
            for seed in SEEDS:
                t0 = time.time()
                print(f"[regression] {dataset} seed={seed}", flush=True)
                try:
                    rows, curve, meta = run_reg_source(dataset, seed, args)
                    reg_rows.extend(rows)
                    reg_curve.extend(curve)
                    status = "success"
                    reason = ""
                except Exception as exc:
                    status = "failed"
                    reason = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                    reg_rows.extend(failed_rows("regression", dataset, seed, reason))
                    failed.append({"task": "regression", "dataset": dataset, "seed": seed, "failure_reason": reason})
                    meta = {}
                run_records.append({"task": "regression", "dataset": dataset, "seed": seed, "status": status, "seconds": time.time() - t0, "failure_reason": reason, **meta})
                write_outputs(out_dir, make_tables(class_rows, class_bins, reg_rows, reg_curve, cluster_rows, cluster_labels, run_records, failed), build_manifest(args, start))

    if "clustering" in task_set:
        for dataset in CLUSTERING_DATASETS:
            for seed in SEEDS:
                t0 = time.time()
                print(f"[clustering] {dataset} seed={seed}", flush=True)
                try:
                    rows, labels, meta = run_cluster_source(dataset, seed, args)
                    cluster_rows.extend(rows)
                    cluster_labels.extend(labels)
                    status = "success"
                    reason = ""
                except Exception as exc:
                    status = "failed"
                    reason = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                    cluster_rows.extend(failed_rows("clustering", dataset, seed, reason))
                    failed.append({"task": "clustering", "dataset": dataset, "seed": seed, "failure_reason": reason})
                    meta = {}
                run_records.append({"task": "clustering", "dataset": dataset, "seed": seed, "status": status, "seconds": time.time() - t0, "failure_reason": reason, **meta})
                write_outputs(out_dir, make_tables(class_rows, class_bins, reg_rows, reg_curve, cluster_rows, cluster_labels, run_records, failed), build_manifest(args, start))

    write_outputs(out_dir, make_tables(class_rows, class_bins, reg_rows, reg_curve, cluster_rows, cluster_labels, run_records, failed), build_manifest(args, start))
    print(f"[done] output={out_dir}", flush=True)


def make_tables(class_rows, class_bins, reg_rows, reg_curve, cluster_rows, cluster_labels, run_records, failed) -> Dict[str, pd.DataFrame]:
    c = pd.DataFrame(class_rows)
    r = pd.DataFrame(reg_rows)
    cl = pd.DataFrame(cluster_rows)
    tables = {
        "classification_per_seed": c,
        "classification_summary": aggregate(c, ["dataset", "config", "model_family", "calibration_method"], ["auc", "acc", "ece", "brier", "nll"]) if not c.empty else pd.DataFrame(),
        "classification_reliability_bins": pd.DataFrame(class_bins),
        "regression_per_seed": r,
        "regression_summary": aggregate(r, ["dataset", "config", "model_family", "interval_method", "alpha"], ["mae", "rmse", "r2", "picp", "mpiw", "pinaw", "coverage_error"]) if not r.empty else pd.DataFrame(),
        "regression_interval_curve": pd.DataFrame(reg_curve),
        "clustering_per_seed": cl,
        "clustering_summary": aggregate(cl, ["dataset", "config", "model_family", "postprocessing_method"], ["ari", "nmi", "selected_k", "silhouette", "db", "ch"]) if not cl.empty else pd.DataFrame(),
        "clustering_label_records": pd.DataFrame(cluster_labels),
        "live_run_records": pd.DataFrame(run_records),
        "failed_runs": pd.DataFrame(failed),
    }
    api = pd.DataFrame(run_records)
    if not api.empty:
        tables["api_call_summary"] = pd.DataFrame([{
            "live_source_runs_completed": int((api["status"] == "success").sum()),
            "live_source_runs_failed": int((api["status"] != "success").sum()),
            "llm_calls_logged": int(pd.to_numeric(api.get("llm_calls", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "llm_tokens_logged": int(pd.to_numeric(api.get("llm_tokens_logged", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "seconds_total": float(pd.to_numeric(api.get("seconds", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        }])
    else:
        tables["api_call_summary"] = pd.DataFrame()
    return tables


def build_manifest(args: argparse.Namespace, start: float) -> Dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - start,
        "code_family": "autoLOGIC_compare",
        "entry_point": "experiments/fig6_panel_b_compare_live_reduced.py",
        "llm": args.llm,
        "tasks": [t.strip() for t in args.tasks.split(",") if t.strip()],
        "seeds": SEEDS,
        "datasets": {
            "classification": CLASSIFICATION_DATASETS,
            "regression": REGRESSION_DATASETS,
            "clustering": CLUSTERING_DATASETS,
        },
        "iterations": {
            "feat_iterations": args.feat_iterations,
            "class_model_iterations": args.class_model_iterations,
            "reg_model_iterations": args.reg_model_iterations,
            "cluster_model_iterations": args.cluster_model_iterations,
            "param_iterations": args.param_iterations,
            "distill_epochs": args.distill_epochs,
        },
        "classification_configs": CLASS_CONFIGS,
        "regression_configs": REG_CONFIGS,
        "clustering_configs": CLUSTER_CONFIGS,
        "clustering_ground_truth_policy": "labels only used for final offline ARI/NMI",
        "clustering_selection_policy": "expected-k-first internal metrics; same-k cached-label voting",
        "clustering_student_policy": "pseudo-label student trained from cached teacher labels",
        "scaling_rule": "pairwise sample-matrix consensus disabled",
        "postprocessing_policy": "source runs are live; calibration, conformal intervals, and voting are offline from captured source outputs",
    }


if __name__ == "__main__":
    main()
