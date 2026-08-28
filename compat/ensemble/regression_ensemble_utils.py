import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score


class ConformalCalibrator:
    def __init__(self, alpha=0.1):
        self.alpha = float(alpha)
        self.q_hat = None

    def fit(self, val_pred, val_y):
        val_pred = np.asarray(val_pred, dtype=float).reshape(-1)
        val_y = np.asarray(val_y, dtype=float).reshape(-1)
        residuals = np.abs(val_y - val_pred)
        self.q_hat = np.quantile(residuals, 1.0 - self.alpha, method="higher")

    def predict(self, test_pred):
        test_pred = np.asarray(test_pred, dtype=float).reshape(-1)
        return test_pred - self.q_hat, test_pred + self.q_hat


def _as_np(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    return np.asarray(x)


def _predict_with_fit_if_needed(model, x_train, y_train, x_pred):
    m = deepcopy(model)
    try:
        pred = np.asarray(m.predict(x_pred), dtype=float).reshape(-1)
        return m, pred
    except Exception:
        m.fit(x_train, y_train)
        pred = np.asarray(m.predict(x_pred), dtype=float).reshape(-1)
        return m, pred


def _interval_metrics(y_true, lower, upper):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside)), float(np.mean(upper - lower))


def _select_columns(val_pred, y_val, max_models=5, rel_rmse=1.20, corr_cut=0.995):
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    scores = []
    for j in range(val_pred.shape[1]):
        p = np.asarray(val_pred[:, j], dtype=float).reshape(-1)
        if len(p) != len(y_val) or not np.isfinite(p).all():
            continue
        rmse = float(np.sqrt(np.mean((y_val - p) ** 2)))
        mae = float(mean_absolute_error(y_val, p))
        scores.append((j, rmse, mae))
    if not scores:
        raise ValueError("No usable cached regression predictions.")
    scores.sort(key=lambda x: (x[1], x[2]))
    best_rmse = scores[0][1]
    selected = []
    for j, rmse, _ in scores:
        if rmse > best_rmse * rel_rmse and len(selected) >= 1:
            continue
        redundant = False
        for k in selected:
            c = np.corrcoef(val_pred[:, j], val_pred[:, k])[0, 1]
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


def _inverse_rmse_weights(val_pred, y_val):
    y_val = np.asarray(y_val, dtype=float).reshape(-1)
    rmses = []
    for j in range(val_pred.shape[1]):
        rmse = np.sqrt(np.mean((y_val - val_pred[:, j]) ** 2))
        rmses.append(max(float(rmse), 1e-9))
    w = 1.0 / np.square(np.asarray(rmses, dtype=float))
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(val_pred.shape[1], dtype=float)
    return w / w.sum()


def stacking_regression_util(
    base_models,
    meta_model,
    X,
    y,
    X_test,
    y_test,
    X_val=None,
    y_val=None,
    calibrate=False,
    alpha=0.1,
    n_folds=5,
    verbose=True,
    random_state=42,
):
    if X_val is None or y_val is None:
        X_val, y_val = X, y
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    y_val_arr = np.asarray(y_val, dtype=float).reshape(-1)
    y_test_arr = np.asarray(y_test, dtype=float).reshape(-1)

    val_cols, test_cols, fitted = [], [], []
    for model in base_models:
        try:
            m, val_pred = _predict_with_fit_if_needed(model, X, y_arr, X_val)
            test_pred = np.asarray(m.predict(X_test), dtype=float).reshape(-1)
            if len(val_pred) != len(y_val_arr) or len(test_pred) != len(y_test_arr):
                continue
            if not np.isfinite(val_pred).all() or not np.isfinite(test_pred).all():
                continue
            val_cols.append(val_pred)
            test_cols.append(test_pred)
            fitted.append(m)
        except Exception as e:
            if verbose:
                print(f"[cached-regression] skipped model: {type(e).__name__}: {e}")
            continue

    if not val_cols:
        raise ValueError("No usable base models for cached regression ensemble.")
    val_meta = np.column_stack(val_cols)
    test_meta = np.column_stack(test_cols)
    selected, scores = _select_columns(val_meta, y_val_arr, max_models=5)
    val_meta = val_meta[:, selected]
    test_meta = test_meta[:, selected]
    weights = _inverse_rmse_weights(val_meta, y_val_arr)
    final_preds = test_meta @ weights
    val_preds = val_meta @ weights

    rmse = float(np.sqrt(np.mean((y_test_arr - final_preds) ** 2)))
    mae = float(mean_absolute_error(y_test_arr, final_preds))
    r2 = float(r2_score(y_test_arr, final_preds))
    try:
        if np.any(y_test_arr < 0) or np.any(final_preds < 0):
            rmsle = np.nan
        else:
            rmsle = float(np.sqrt(mean_squared_log_error(y_test_arr, final_preds)))
    except Exception:
        rmsle = np.nan
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmsle": rmsle,
        "selected_model_count": int(len(selected)),
        "cached_prediction_mode": True,
        "val_best_rmse": float(min(s[1] for s in scores)),
    }

    calibrator = None
    if calibrate:
        calibrator = ConformalCalibrator(alpha=alpha)
        calibrator.fit(val_preds, y_val_arr)
        lower, upper = calibrator.predict(final_preds)
        coverage, width = _interval_metrics(y_test_arr, lower, upper)
        metrics["coverage"] = coverage
        metrics["width"] = width

    if verbose:
        print("[cached-regression] selected models:", len(selected))
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                print(f"{k}: {v:.6f}" if np.isfinite(v) else f"{k}: NaN")

    return {
        "full_fit_models": fitted,
        "meta_model": None,
        "stacking_metrics": metrics,
        "final_predictions": final_preds,
        "calibrator": calibrator,
        "test_meta": test_meta,
        "val_meta": val_meta,
        "weights": weights,
    }


def voting_regression_util(base_models, X_test, y_test):
    preds = []
    for model in base_models:
        preds.append(np.asarray(model.predict(X_test), dtype=float).reshape(-1))
    pred = np.mean(np.column_stack(preds), axis=1)
    y = np.asarray(y_test, dtype=float).reshape(-1)
    return {"voting_metrics": {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "rmsle": float("nan") if np.any(y < 0) or np.any(pred < 0) else float(np.sqrt(mean_squared_log_error(y, pred))),
    }}


def bagging_regression_util(base_models, X_test, y_test):
    result = voting_regression_util(base_models, X_test, y_test)["voting_metrics"]
    return {"bagging_metrics": result}
