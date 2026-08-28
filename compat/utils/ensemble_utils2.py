import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _as_np(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    return np.asarray(x)


def _encode_binary_y(y):
    y = _as_np(y).reshape(-1)
    uniq = np.unique(y[pd.notna(y)])
    if set(uniq.tolist()) <= {0, 1, 0.0, 1.0}:
        return y.astype(int)
    return LabelEncoder().fit_transform(y).astype(int)


def _positive_proba(model, x):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].astype(float)
        return p.reshape(-1).astype(float)
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(x), dtype=float).reshape(-1)
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))
    return np.asarray(model.predict(x), dtype=float).reshape(-1)


def _predict_with_fit_if_needed(model, x_train, y_train, x_pred):
    m = deepcopy(model)
    try:
        return m, _positive_proba(m, x_pred)
    except Exception:
        m.fit(x_train, y_train)
        return m, _positive_proba(m, x_pred)


def _ece_binary(y_true, p, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        if mask.any():
            out += (mask.sum() / n) * abs(float(y_true[mask].mean()) - float(p[mask].mean()))
    return float(out)


def _best_threshold(y, p):
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        acc = accuracy_score(y, (p >= t).astype(int))
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    return best_t


def _select_columns(val_pred, y_val, max_models=5, min_auc=0.5, corr_cut=0.995):
    scores = []
    for j in range(val_pred.shape[1]):
        p = np.nan_to_num(val_pred[:, j], nan=0.5, posinf=1.0, neginf=0.0)
        try:
            auc = roc_auc_score(y_val, p)
        except Exception:
            auc = np.nan
        try:
            brier = brier_score_loss(y_val, np.clip(p, 1e-6, 1 - 1e-6))
        except Exception:
            brier = np.inf
        if np.isfinite(auc):
            scores.append((j, float(auc), float(brier)))
    if not scores:
        raise ValueError("No usable cached classifier predictions.")
    scores.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    best_auc = scores[0][1]
    floor = max(float(min_auc), best_auc - 0.03)
    selected = []
    for j, auc, _ in scores:
        if auc < floor and len(selected) >= 1:
            continue
        candidate = val_pred[:, j]
        redundant = False
        for k in selected:
            c = np.corrcoef(candidate, val_pred[:, k])[0, 1]
            if np.isfinite(c) and abs(c) >= corr_cut:
                redundant = True
                break
        if not redundant:
            selected.append(j)
        if len(selected) >= max_models:
            break
    if not selected:
        selected = [scores[0][0]]
    return selected, scores


def stacking_ensemble_v2(
    base_models,
    X_train,
    y_train,
    X_test,
    y_test,
    X_val=None,
    y_val=None,
    weight_list=None,
    weight_handling="ignore",
    n_folds=5,
    meta_cv_repeats=3,
    meta_C_grid=(0.03, 0.1, 0.3, 1, 3),
    use_logit=False,
    scale_meta=True,
    class_weight=None,
    optimize_metric="accuracy",
    tune_threshold=True,
    threshold_metric="accuracy",
    calibrate_proba=False,
    calibration_method="sigmoid",
    calibration_cv=3,
    random_state=42,
    verbose=False,
    drop_x_na_in_train=False,
    drop_x_na_in_test=False,
):
    if X_val is None or y_val is None:
        X_val, y_val = X_train, y_train
    y_train_enc = _encode_binary_y(y_train)
    y_val_enc = _encode_binary_y(y_val)
    y_test_enc = _encode_binary_y(y_test)

    val_cols, test_cols, fitted = [], [], []
    for model in base_models:
        try:
            m, p_val = _predict_with_fit_if_needed(model, X_train, y_train_enc, X_val)
            p_test = _positive_proba(m, X_test)
            if len(p_val) != len(y_val_enc) or len(p_test) != len(y_test_enc):
                continue
            val_cols.append(np.clip(np.asarray(p_val, dtype=float), 1e-6, 1 - 1e-6))
            test_cols.append(np.clip(np.asarray(p_test, dtype=float), 1e-6, 1 - 1e-6))
            fitted.append(m)
        except Exception as e:
            if verbose:
                print(f"[cached-classification] skipped model: {type(e).__name__}: {e}")
            continue

    if not val_cols:
        raise ValueError("No usable base models for cached classification ensemble.")
    val_meta = np.column_stack(val_cols)
    test_meta = np.column_stack(test_cols)
    selected, val_scores = _select_columns(val_meta, y_val_enc, max_models=5)
    val_meta = val_meta[:, selected]
    test_meta = test_meta[:, selected]

    if val_meta.shape[1] == 1:
        final_proba = test_meta[:, 0]
        val_proba = val_meta[:, 0]
        meta_model = None
    else:
        steps = []
        if scale_meta:
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", LogisticRegression(max_iter=10000, class_weight=class_weight, random_state=random_state)))
        meta_model = Pipeline(steps)
        if calibrate_proba and len(y_val_enc) >= 20:
            cv = min(calibration_cv, max(2, np.bincount(y_val_enc).min()))
            if cv >= 2:
                meta_model = CalibratedClassifierCV(meta_model, method=calibration_method, cv=cv)
        meta_model.fit(val_meta, y_val_enc)
        final_proba = meta_model.predict_proba(test_meta)[:, 1]
        val_proba = meta_model.predict_proba(val_meta)[:, 1]

    threshold = _best_threshold(y_val_enc, val_proba) if tune_threshold else 0.5
    pred = (final_proba >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test_enc, pred)),
        "f1": float(f1_score(y_test_enc, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_enc, final_proba)),
        "precision": float(precision_score(y_test_enc, pred, zero_division=0)),
        "recall": float(recall_score(y_test_enc, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_test_enc, final_proba)),
        "ece": _ece_binary(y_test_enc, final_proba),
        "selected_model_count": int(len(selected)),
        "cached_prediction_mode": True,
        "val_best_auc": float(max(s[1] for s in val_scores)),
        "threshold": float(threshold),
    }
    return {
        "meta_model": meta_model,
        "base_models_fitted": fitted,
        "stacking_metrics": metrics,
        "final_proba": final_proba,
        "test_meta": test_meta,
        "train_meta": val_meta,
        "calibration_applied": bool(calibrate_proba),
        "calibration_method": calibration_method if calibrate_proba else None,
    }
