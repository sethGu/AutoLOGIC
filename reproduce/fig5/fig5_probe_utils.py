import importlib.util
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]


def import_autologic_module(task: str):
    unit_path = ROOT / "experiments" / "fig5_strict_fixed_pool_unit.py"
    spec = importlib.util.spec_from_file_location("fig5_strict_fixed_pool_unit_probe", unit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {unit_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.import_module_for_task(task)


def _numeric_align(*frames):
    lengths = [len(df) for df in frames]
    combined = pd.concat([df.reset_index(drop=True) for df in frames], axis=0, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = pd.get_dummies(combined, dummy_na=True)
    combined = combined.fillna(0)
    out = []
    start = 0
    for length in lengths:
        out.append(combined.iloc[start : start + length].reset_index(drop=True))
        start += length
    return out


def maybe_subsample(x, y, max_rows, seed, stratify=False):
    if max_rows is None or max_rows <= 0 or len(x) <= max_rows:
        return x.reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    y_series = pd.Series(y).reset_index(drop=True)
    strat = y_series if stratify and y_series.nunique() > 1 else None
    x_sub, _, y_sub, _ = train_test_split(
        x.reset_index(drop=True),
        y_series,
        train_size=max_rows,
        random_state=seed,
        stratify=strat,
    )
    return x_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


def load_classification_dataset(dataset: str, seed: int, max_train_rows=None, max_test_rows=None):
    mod = import_autologic_module("classification")
    df_train, df_test, target, _ = mod.load_origin_data(dataset, seed)
    x_train, y_train = mod.to_pd(df_train, target)
    x_test, y_test = mod.to_pd(df_test, target)
    x_train, x_test = _numeric_align(x_train, x_test)
    x_train, y_train = maybe_subsample(x_train, y_train, max_train_rows, seed, stratify=True)
    x_test, y_test = maybe_subsample(x_test, y_test, max_test_rows, seed + 1000, stratify=True)
    return x_train, pd.Series(y_train).astype(int), x_test, pd.Series(y_test).astype(int)


def load_regression_dataset(dataset: str, seed: int, max_train_rows=None, max_test_rows=None):
    mod = import_autologic_module("regression")
    loc = ROOT / "AutoLogic" / "data" / f"{dataset}.pkl"
    df_train, df_test, target, _ = mod.load_origin_data(str(loc), seed)
    x_train, y_train = mod.to_pd(df_train, target)
    x_test, y_test = mod.to_pd(df_test, target)
    x_train, x_test = _numeric_align(x_train, x_test)
    x_train, y_train = maybe_subsample(x_train, y_train, max_train_rows, seed, stratify=False)
    x_test, y_test = maybe_subsample(x_test, y_test, max_test_rows, seed + 1000, stratify=False)
    return x_train, pd.Series(y_train).astype(float), x_test, pd.Series(y_test).astype(float)


class MeanProbEnsemble:
    def __init__(self, models):
        self.models = list(models)

    def predict_proba(self, x):
        probs = []
        for model in self.models:
            p = model.predict_proba(x)
            if p.ndim == 1:
                p = np.vstack([1.0 - p, p]).T
            if p.shape[1] == 1:
                p = np.hstack([1.0 - p, p])
            probs.append(p)
        return np.mean(probs, axis=0)

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class MeanRegressorEnsemble:
    def __init__(self, models):
        self.models = list(models)

    def predict(self, x):
        preds = [np.asarray(model.predict(x), dtype=float).ravel() for model in self.models]
        return np.mean(preds, axis=0)


def ece_score(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return ece


def classification_scores(y_true, prob):
    y_true = np.asarray(y_true).ravel()
    prob = np.clip(np.asarray(prob, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    return {
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "ece": float(ece_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "nll": float(log_loss(y_true, np.vstack([1.0 - prob, prob]).T, labels=[0, 1])),
    }


def artifact_kb(model):
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0


def median_predict_latency_ms(model, x, kind: str, repeats: int, batch_rows: int):
    x_batch = x.iloc[: min(batch_rows, len(x))].copy()
    if len(x_batch) == 0:
        return np.nan, 0
    timings = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        if kind == "classification":
            model.predict_proba(x_batch)
        else:
            model.predict(x_batch)
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings)), int(len(x_batch))
