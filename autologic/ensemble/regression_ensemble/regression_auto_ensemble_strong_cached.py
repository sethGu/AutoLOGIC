from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import HuberRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
NEW_DATA_ROOT = Path(os.environ.get("AUTOLOGIC_REGRESSION_DATA_DIR", WORKSPACE_ROOT / "data" / "csv_data"))

DATASET_ALIASES = {
    "crab": "crab_age",
    "forest": "forest-fires",
    "puma8nh": "puma8NH",
}

BANNED_CODE_PATTERNS = [
    "subprocess",
    "requests",
    "socket",
    "open(",
    "input(",
    "exec(",
    "eval(",
    "gridsearchcv",
    "randomizedsearchcv",
    "cross_val_score",
    "cross_validate",
    "while true",
]


@dataclass
class Candidate:
    name: str
    source: str
    val_mae: float
    val_rmse: float
    val_r2: float
    test_mae: float
    test_rmse: float
    test_r2: float
    train_pred: List[float]
    val_pred: List[float]
    test_pred: List[float]
    code_path: str = ""
    selected_from: str = ""
    failed: bool = False
    failure_reason: str = ""


def log_event(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_pickle_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def read_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def wait_for_result_file(proc, result_path: str | Path, timeout_s: int, stage: str) -> Any:
    result_path = Path(result_path)
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if result_path.exists():
            proc.join(2)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(5)
            return read_pickle(result_path)
        if not proc.is_alive():
            break
        proc.join(min(0.25, max(0.01, deadline - time.time())))
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        raise TimeoutError(f"{stage} stage exceeded {timeout_s}s")
    if result_path.exists():
        return read_pickle(result_path)
    raise RuntimeError(f"{stage} worker returned no result")


def record_tokens(output_dir: Path, stage: str, model: str, usage: Any) -> None:
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "model": model,
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    with (output_dir / "token_ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    max_messages = int(os.environ.get("AUTOLOGIC_MAX_MESSAGES", "8"))
    max_chars = int(os.environ.get("AUTOLOGIC_MAX_MESSAGE_CHARS", "7000"))
    if len(messages) > max_messages:
        messages = messages[:1] + messages[-(max_messages - 1):]
    out = []
    for msg in messages:
        item = dict(msg)
        content = str(item.get("content", ""))
        if len(content) > max_chars:
            item["content"] = content[-max_chars:]
        out.append(item)
    return out


def call_llm(
    messages: List[Dict[str, str]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    output_dir: Path,
    stage: str,
    timeout_s: int,
    max_tokens: int,
) -> str:
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=float(timeout_s))
    kwargs = {
        "model": model,
        "messages": compact_messages(messages),
        "temperature": 0.7,
        "stop": ["```end"],
    }
    try:
        completion = client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
    except TypeError:
        completion = client.chat.completions.create(max_tokens=max_tokens, **kwargs)
    record_tokens(output_dir, stage, model, completion.usage)
    code = completion.choices[0].message.content or ""
    return clean_code(code)


def clean_code(text: str) -> str:
    code = text.replace("```python", "").replace("```", "").replace("<end>", "")
    lines = code.strip().splitlines()
    kept = []
    started = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from ") or stripped.startswith("class myregressor"):
            started = True
        if started:
            kept.append(line)
    return "\n".join(kept).strip() if kept else code.strip()


def validate_code(code: str) -> Optional[str]:
    lower = code.lower()
    if "class myregressor" not in lower:
        return "missing class myregressor"
    for pat in BANNED_CODE_PATTERNS:
        if pat in lower:
            return f"banned pattern: {pat}"
    return None


def save_code(output_dir: Path, stage: str, name: str, code: str, accepted: bool, reason: str = "") -> Path:
    code_dir = output_dir / "generated_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120]
    path = code_dir / f"{stage}_{safe}.py"
    path.write_text(code or "", encoding="utf-8")
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "name": name,
        "code_path": str(path),
        "accepted": bool(accepted),
        "reason": reason,
    }
    with (output_dir / "generated_code_manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return path


def encode_features(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df_train.copy()
    test = df_test.copy()
    n_train = len(train)
    for c in [x for x in train.columns if x != target]:
        if pd.api.types.is_bool_dtype(train[c]):
            train[c] = train[c].astype(int)
            test[c] = test[c].astype(int)
        elif pd.api.types.is_numeric_dtype(train[c]):
            train[c] = pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            test[c] = pd.to_numeric(test[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            all_col = pd.concat([train[c], test[c]], ignore_index=True).astype("string").fillna("__missing__").str.strip()
            cat = pd.Categorical(all_col)
            codes = pd.Series(cat.codes, dtype="int32")
            train[c] = codes.iloc[:n_train].to_numpy()
            test[c] = codes.iloc[n_train:].to_numpy()
    train[target] = pd.to_numeric(train[target], errors="coerce")
    test[target] = pd.to_numeric(test[target], errors="coerce")
    train = train[~train[target].isna()].copy()
    test = test[~test[target].isna()].copy()
    return train.replace([np.inf, -np.inf], np.nan).fillna(0.0), test.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_dataset(dataset: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    alias = DATASET_ALIASES.get(dataset, dataset)
    pkl_path = PROJECT_ROOT / "data" / f"{alias}.pkl"
    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            ds = pickle.load(f)
        df = ds[1].copy()
        target = str(ds[4][-1])
        desc = str(ds[-1] or "")
    else:
        csv_path = NEW_DATA_ROOT / f"{alias}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing regression dataset: {dataset}; checked {pkl_path} and {csv_path}")
        df = pd.read_csv(csv_path)
        target = str(df.columns[-1])
        desc_path = NEW_DATA_ROOT / f"{alias}-description.txt"
        desc = desc_path.read_text(encoding="utf-8", errors="replace") if desc_path.exists() else ""
    train, test = train_test_split(df, test_size=0.25, random_state=seed, shuffle=True)
    return (*encode_features(train, test, target), target, desc)


def to_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target]).reset_index(drop=True), df[target].reset_index(drop=True)


def metric_dict(y_true, pred) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    if len(p) != len(y) or not np.all(np.isfinite(p)):
        raise ValueError("invalid prediction vector")
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
    }


def make_builtin_candidates(seed: int, y_nonnegative: bool) -> List[Tuple[str, Any]]:
    candidates: List[Tuple[str, Any]] = [
        ("rf_600", RandomForestRegressor(n_estimators=600, min_samples_leaf=1, max_features=0.8, n_jobs=-1, random_state=seed)),
        ("rf_300_leaf2", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=seed + 1)),
        ("extra_800", ExtraTreesRegressor(n_estimators=800, min_samples_leaf=1, max_features=0.9, n_jobs=-1, random_state=seed + 2)),
        ("hgb_l2", HistGradientBoostingRegressor(max_iter=500, learning_rate=0.045, l2_regularization=0.02, random_state=seed + 3)),
        ("hgb_abs", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, loss="absolute_error", random_state=seed + 4)),
        ("gbr", GradientBoostingRegressor(n_estimators=450, learning_rate=0.035, max_depth=3, subsample=0.85, random_state=seed + 5)),
        ("huber_power", make_pipeline(PowerTransformer(), HuberRegressor(max_iter=1000, epsilon=1.35))),
    ]
    try:
        from xgboost import XGBRegressor

        candidates.append(
            (
                "xgb_depth4",
                XGBRegressor(
                    n_estimators=600,
                    max_depth=4,
                    learning_rate=0.035,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.5,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=seed + 6,
                    verbosity=0,
                ),
            )
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        candidates.append(
            (
                "lgbm",
                LGBMRegressor(
                    n_estimators=700,
                    learning_rate=0.035,
                    num_leaves=31,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    n_jobs=-1,
                    random_state=seed + 7,
                    verbose=-1,
                ),
            )
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor

        candidates.append(
            (
                "catboost",
                CatBoostRegressor(
                    iterations=700,
                    depth=6,
                    learning_rate=0.035,
                    loss_function="MAE",
                    random_seed=seed + 8,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                ),
            )
        )
    except Exception:
        pass
    if y_nonnegative:
        wrapped = []
        for name, model in candidates[:5]:
            wrapped.append((f"log1p_{name}", TransformedTargetRegressor(regressor=clone(model), func=np.log1p, inverse_func=np.expm1)))
        candidates.extend(wrapped)
    return candidates


def evaluate_prefit_candidate(name: str, source: str, model: Any, X_train, y_train, X_val, y_val, X_test, y_test) -> Candidate:
    train_pred = np.asarray(model.predict(X_train), dtype=float).ravel()
    val_pred = np.asarray(model.predict(X_val), dtype=float).ravel()
    test_pred = np.asarray(model.predict(X_test), dtype=float).ravel()
    val = metric_dict(y_val, val_pred)
    test = metric_dict(y_test, test_pred)
    return Candidate(
        name=name,
        source=source,
        val_mae=val["mae"],
        val_rmse=val["rmse"],
        val_r2=val["r2"],
        test_mae=test["mae"],
        test_rmse=test["rmse"],
        test_r2=test["r2"],
        train_pred=train_pred.tolist(),
        val_pred=val_pred.tolist(),
        test_pred=test_pred.tolist(),
    )


def model_code_worker(input_path: str, result_path: str) -> None:
    try:
        payload = read_pickle(input_path)
        code = payload["code"]
        class_name = payload["class_name"]
        X_train = payload["X_train"]
        y_train = payload["y_train"]
        X_val = payload["X_val"]
        y_val = payload["y_val"]
        X_test = payload["X_test"]
        y_test = payload["y_test"]
        ns: Dict[str, Any] = {
            "np": np,
            "pd": pd,
            "RandomForestRegressor": RandomForestRegressor,
            "ExtraTreesRegressor": ExtraTreesRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
            "StandardScaler": StandardScaler,
            "RobustScaler": RobustScaler,
            "PowerTransformer": PowerTransformer,
            "QuantileTransformer": QuantileTransformer,
            "Ridge": Ridge,
            "HuberRegressor": HuberRegressor,
            "make_pipeline": make_pipeline,
        }
        exec(code, ns, ns)
        cls = ns.get(class_name)
        if cls is None:
            raise ValueError(f"{class_name} not defined")
        model = cls()
        model.fit(X_train, y_train)
        train_pred = np.asarray(model.predict(X_train), dtype=float).ravel()
        val_pred = np.asarray(model.predict(X_val), dtype=float).ravel()
        test_pred = np.asarray(model.predict(X_test), dtype=float).ravel()
        val = metric_dict(y_val, val_pred)
        test = metric_dict(y_test, test_pred)
        write_pickle_atomic(
            result_path,
            {
                "ok": True,
                "train_pred": train_pred,
                "val_pred": val_pred,
                "test_pred": test_pred,
                "val": val,
                "test": test,
            },
        )
    except Exception as exc:
        write_pickle_atomic(result_path, {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(limit=5)})


def run_model_code_with_timeout(
    code: str,
    class_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    timeout_s: int,
) -> Dict[str, Any]:
    import multiprocessing as mp

    with tempfile.TemporaryDirectory(prefix="autologic_reg_model_") as tmpdir:
        input_path = Path(tmpdir) / "input.pkl"
        result_path = Path(tmpdir) / "result.pkl"
        write_pickle_atomic(
            input_path,
            {
                "code": code,
                "class_name": class_name,
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
            },
        )
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=model_code_worker, args=(str(input_path), str(result_path)))
        proc.start()
        result = wait_for_result_file(proc, result_path, timeout_s, "model")
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "model worker failed"))
    return result


def feature_worker(input_path: str, result_path: str) -> None:
    try:
        from regression_auto_ensemble import base_model, generate_feat

        payload = read_pickle(input_path)
        df_train_aug, df_test_aug = generate_feat(
            base_model=base_model(payload["seed"]),
            df_train=payload["df_train"],
            df_test=payload["df_test"],
            dataset_name=payload["dataset"],
            round_num=1,
            llm_model=payload["llm"],
            iterations=payload["iterations"],
            target_column_name=payload["target"],
            dataset_description=payload["description"],
            task_type="regression",
            base_url=payload["base_url"],
            api_key=payload["api_key"],
            logger=None,
        )
        write_pickle_atomic(result_path, {"ok": True, "df_train": df_train_aug, "df_test": df_test_aug})
    except Exception as exc:
        write_pickle_atomic(result_path, {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(limit=5)})


def run_feature_with_timeout(df_train, df_test, target, dataset, description, args, output_dir: Path):
    if args.feat_iterations <= 0:
        return df_train, df_test, {"feature_status": "skipped", "feature_error": ""}
    import multiprocessing as mp

    with tempfile.TemporaryDirectory(prefix="autologic_reg_feature_") as tmpdir:
        input_path = Path(tmpdir) / "input.pkl"
        result_path = Path(tmpdir) / "result.pkl"
        write_pickle_atomic(
            input_path,
            {
                "df_train": df_train,
                "df_test": df_test,
                "target": target,
                "dataset": dataset,
                "description": description,
                "seed": args.seed,
                "llm": args.llm,
                "iterations": args.feat_iterations,
                "base_url": args.api_base,
                "api_key": args.api_key,
            },
        )
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=feature_worker, args=(str(input_path), str(result_path)))
        proc.start()
        try:
            result = wait_for_result_file(proc, result_path, args.feature_timeout_s, "feature")
        except Exception as exc:
            log_event(output_dir, {"stage": "feature_failed", "error": repr(exc), "fallback": "raw_features"})
            return df_train, df_test, {"feature_status": "fallback_raw", "feature_error": repr(exc)}
    if not result.get("ok"):
        log_event(output_dir, {"stage": "feature_failed", "error": result.get("error"), "fallback": "raw_features"})
        return df_train, df_test, {"feature_status": "fallback_raw", "feature_error": str(result.get("error"))}
    return result["df_train"], result["df_test"], {"feature_status": "success", "feature_error": ""}


def make_model_prompt(dataset: str, description: str, target: str, df_train: pd.DataFrame, history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sample = df_train.head(8).round(4).to_dict(orient="list")
    summary = df_train.describe().round(4).to_string()
    hist = "\n".join(
        f"- {h['name']}: val MAE={h['val_mae']:.5f}, val RMSE={h['val_rmse']:.5f}, source={h['source']}"
        for h in history[-10:]
    )
    return [
        {
            "role": "system",
            "content": (
                "You write compact, executable Python regression estimators for tabular data. "
                "Primary objective: minimize validation MAE. Secondary objective: minimize RMSE. "
                "Use only train features and target supplied to fit; do not access test labels or files."
            ),
        },
        {
            "role": "user",
            "content": f"""
Dataset: {dataset}
Target column: {target}
Description:
{description[:1200]}

Feature/target sample:
{json.dumps(sample, ensure_ascii=False)}

Numeric summary:
{summary[:5000]}

Previous validation evidence:
{hist or "None"}

Return Python code only. Define exactly one class named myregressor with fit(self, X, y) and predict(self, X).

Guidance:
- Prefer strong tabular regressors: ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor.
- Use robust preprocessing only inside a Pipeline when appropriate.
- If using tree ensembles, keep runtime bounded: n_estimators <= 900, max_depth <= 10 when available, n_jobs=-1 when available.
- Do not use GridSearchCV, RandomizedSearchCV, cross-validation utilities, file/network calls, or sample-by-sample distance matrices.
- The class must be self-contained and deterministic with random_state where possible.
```end
""",
        },
    ]


def evaluate_code_candidate(
    name: str,
    source: str,
    code: str,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    timeout_s: int,
    output_dir: Path,
) -> Candidate:
    result = run_model_code_with_timeout(code, name, X_train, y_train, X_val, y_val, X_test, y_test, timeout_s)
    val = result["val"]
    test = result["test"]
    return Candidate(
        name=name,
        source=source,
        val_mae=float(val["mae"]),
        val_rmse=float(val["rmse"]),
        val_r2=float(val["r2"]),
        test_mae=float(test["mae"]),
        test_rmse=float(test["rmse"]),
        test_r2=float(test["r2"]),
        train_pred=np.asarray(result["train_pred"], dtype=float).tolist(),
        val_pred=np.asarray(result["val_pred"], dtype=float).tolist(),
        test_pred=np.asarray(result["test_pred"], dtype=float).tolist(),
    )


def selection_score(c: Candidate, metric: str) -> float:
    return float(c.val_mae if metric == "mae" else c.val_rmse)


def select_candidates(candidates: List[Candidate], args) -> List[Candidate]:
    valid = [c for c in candidates if np.isfinite(selection_score(c, args.selection_metric))]
    valid.sort(key=lambda c: selection_score(c, args.selection_metric))
    if not valid:
        return []
    best = selection_score(valid[0], args.selection_metric)
    eligible = [c for c in valid if selection_score(c, args.selection_metric) <= best * (1.0 + args.selection_gap)]
    if len(eligible) < args.min_selected and len(valid) >= args.min_selected:
        eligible = valid[: args.min_selected]
    return eligible[: max(1, min(args.top_k, len(eligible)))]


def weighted_average(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    return np.average(preds, axis=0, weights=weights)


def conformal_interval(val_pred, y_val, test_pred, alpha: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    val_pred = np.asarray(val_pred, dtype=float).ravel()
    y_val = np.asarray(y_val, dtype=float).ravel()
    test_pred = np.asarray(test_pred, dtype=float).ravel()
    q = np.quantile(np.abs(y_val - val_pred), 1 - alpha, method="higher")
    lower = test_pred - q
    upper = test_pred + q
    return lower, upper, float(q), float(np.mean(upper - lower))


def evaluate_prediction(config: str, pred, val_pred, y_val, y_test, args, source: str, used: List[Candidate]) -> Dict[str, Any]:
    test = metric_dict(y_test, pred)
    val = metric_dict(y_val, val_pred)
    row = {
        "config": config,
        "source": source,
        "selection_metric": args.selection_metric,
        "val_mae": val["mae"],
        "val_rmse": val["rmse"],
        "val_r2": val["r2"],
        "mae": test["mae"],
        "rmse": test["rmse"],
        "r2": test["r2"],
        "used_model_count": len(used),
        "used_model_names": json.dumps([c.name for c in used], ensure_ascii=False),
        "used_model_val_mae": json.dumps([round(float(c.val_mae), 6) for c in used]),
        "used_model_val_rmse": json.dumps([round(float(c.val_rmse), 6) for c in used]),
    }
    if args.calibrate_intervals:
        lower, upper, q, width = conformal_interval(val_pred, y_val, pred, args.alpha)
        y = np.asarray(y_test, dtype=float).ravel()
        row.update(
            {
                "alpha": args.alpha,
                "picp": float(np.mean((y >= lower) & (y <= upper))),
                "mpiw": width,
                "conformal_q": q,
            }
        )
    return row


def ensemble_rows(candidates: List[Candidate], y_val, y_test, args, prefix: str) -> List[Dict[str, Any]]:
    selected = select_candidates(candidates, args)
    if not selected:
        return []
    val_mat = np.vstack([np.asarray(c.val_pred, dtype=float) for c in selected])
    test_mat = np.vstack([np.asarray(c.test_pred, dtype=float) for c in selected])
    plans = []
    plans.append(("best_single", val_mat[0], test_mat[0], selected[:1]))
    if len(selected) > 1:
        plans.append(("mean", val_mat.mean(axis=0), test_mat.mean(axis=0), selected))
        weights = np.asarray([1.0 / max(selection_score(c, args.selection_metric), 1e-9) for c in selected])
        plans.append(("weighted", weighted_average(val_mat, weights), weighted_average(test_mat, weights), selected))
        try:
            ridge = RidgeCV(alphas=np.logspace(-4, 4, 21))
            ridge.fit(val_mat.T, np.asarray(y_val, dtype=float).ravel())
            plans.append(("ridge_stack", ridge.predict(val_mat.T), ridge.predict(test_mat.T), selected))
        except Exception:
            pass
    rows = []
    scored = []
    for mode, val_pred, test_pred, used in plans:
        row = evaluate_prediction(f"{prefix}_{mode}", test_pred, val_pred, y_val, y_test, args, prefix, used)
        rows.append(row)
        scored.append((row["val_mae" if args.selection_metric == "mae" else "val_rmse"], mode, row))
    scored.sort(key=lambda x: x[0])
    auto = dict(scored[0][2])
    auto["config"] = f"{prefix}_auto"
    auto["auto_selected_mode"] = scored[0][1]
    rows.append(auto)
    return rows


def distill_selected(selected: List[Candidate], X_train, X_val, X_test, y_val, y_test, args) -> List[Candidate]:
    out: List[Candidate] = []
    for i, c in enumerate(selected[: max(0, args.distill_top_k)]):
        try:
            scaler = StandardScaler()
            Xt = scaler.fit_transform(X_train)
            Xv = scaler.transform(X_val)
            Xs = scaler.transform(X_test)
            student = MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=args.distill_epochs,
                early_stopping=True,
                random_state=args.seed + 1000 + i,
            )
            pseudo = np.asarray(c.train_pred, dtype=float)
            student.fit(Xt, pseudo)
            train_pred = student.predict(Xt)
            val_pred = student.predict(Xv)
            test_pred = student.predict(Xs)
            val = metric_dict(y_val, val_pred)
            test = metric_dict(y_test, test_pred)
            out.append(
                Candidate(
                    name=f"student_{c.name}",
                    source="student_distilled",
                    val_mae=val["mae"],
                    val_rmse=val["rmse"],
                    val_r2=val["r2"],
                    test_mae=test["mae"],
                    test_rmse=test["rmse"],
                    test_r2=test["r2"],
                    train_pred=train_pred.tolist(),
                    val_pred=val_pred.tolist(),
                    test_pred=test_pred.tolist(),
                    selected_from=c.name,
                )
            )
        except Exception:
            continue
    return out


def run_one(args) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir).absolute() / f"{args.dataset}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_event(out_dir, {"stage": "start", "dataset": args.dataset, "seed": args.seed, "setting": f"f{args.feat_iterations}_m{args.model_iterations}_p{args.param_iterations}"})

    df_train, df_test, target, desc = load_dataset(args.dataset, args.seed)
    df_train, df_test, feat_info = run_feature_with_timeout(df_train, df_test, target, args.dataset, desc, args, out_dir)
    df_model_train, df_val = train_test_split(df_train, test_size=args.val_fraction, random_state=args.seed + 17, shuffle=True)
    X_train, y_train = to_xy(df_model_train, target)
    X_val, y_val = to_xy(df_val, target)
    X_test, y_test = to_xy(df_test, target)

    candidates: List[Candidate] = []
    y_nonnegative = bool((np.asarray(y_train) >= 0).all())
    for name, model in make_builtin_candidates(args.seed, y_nonnegative):
        try:
            start = time.time()
            model.fit(X_train, y_train)
            cand = evaluate_prefit_candidate(name, "builtin", model, X_train, y_train, X_val, y_val, X_test, y_test)
            candidates.append(cand)
            log_event(out_dir, {"stage": "candidate", "name": name, "source": "builtin", "val_mae": cand.val_mae, "val_rmse": cand.val_rmse, "seconds": time.time() - start})
        except Exception as exc:
            log_event(out_dir, {"stage": "candidate_failed", "name": name, "source": "builtin", "error": repr(exc)})

    history = [{"name": c.name, "source": c.source, "val_mae": c.val_mae, "val_rmse": c.val_rmse} for c in sorted(candidates, key=lambda c: c.val_mae)[:8]]
    for i in range(args.model_iterations):
        code = ""
        try:
            code = call_llm(
                make_model_prompt(args.dataset, desc, target, df_model_train, history),
                model=args.llm,
                base_url=args.api_base,
                api_key=args.api_key,
                output_dir=out_dir,
                stage="model_generation",
                timeout_s=args.api_timeout_s,
                max_tokens=args.max_tokens,
            )
            reason = validate_code(code)
            if reason:
                raise ValueError(reason)
            class_name = f"myregressor_{i + 1}"
            code = re.sub(r"class\s+myregressor[_\w]*\s*(\([^)]*\))?\s*:", f"class {class_name}:", code, count=1)
            path = save_code(out_dir, "model", f"llm_model_{i + 1}", code, accepted=True)
            best_code = code
            best_name = class_name
            best = evaluate_code_candidate(class_name, "llm_model", code, X_train, y_train, X_val, y_val, X_test, y_test, args.model_timeout_s, out_dir)
            best.code_path = str(path)
            param_messages = [
                {
                    "role": "system",
                    "content": "You tune hyperparameters of an existing Python regressor. Return executable code only. Primary objective: lower validation MAE; secondary: lower RMSE.",
                },
                {
                    "role": "user",
                    "content": f"Current code:\n```python\n{best_code}\n```\nCurrent validation MAE={best.val_mae:.6f}, RMSE={best.val_rmse:.6f}. Modify only hyperparameters or lightweight preprocessing. Keep class name myregressor.",
                },
            ]
            for j in range(args.param_iterations):
                p_code = ""
                try:
                    p_code = call_llm(
                        param_messages,
                        model=args.llm,
                        base_url=args.api_base,
                        api_key=args.api_key,
                        output_dir=out_dir,
                        stage="param_refinement",
                        timeout_s=args.api_timeout_s,
                        max_tokens=args.max_tokens,
                    )
                    reason = validate_code(p_code)
                    if reason:
                        raise ValueError(reason)
                    p_name = f"myregressor_{i + 1}_param_{j + 1}"
                    p_code = re.sub(r"class\s+myregressor[_\w]*\s*(\([^)]*\))?\s*:", f"class {p_name}:", p_code, count=1)
                    p_path = save_code(out_dir, "param", f"llm_model_{i + 1}_param_{j + 1}", p_code, accepted=True)
                    p_cand = evaluate_code_candidate(p_name, "llm_param", p_code, X_train, y_train, X_val, y_val, X_test, y_test, args.model_timeout_s, out_dir)
                    p_cand.code_path = str(p_path)
                    if selection_score(p_cand, args.selection_metric) < selection_score(best, args.selection_metric):
                        best = p_cand
                        best_code = p_code
                        best_name = p_name
                    param_messages += [
                        {"role": "assistant", "content": p_code},
                        {"role": "user", "content": f"Validation MAE={p_cand.val_mae:.6f}, RMSE={p_cand.val_rmse:.6f}. Best MAE={best.val_mae:.6f}, best RMSE={best.val_rmse:.6f}. Improve again."},
                    ]
                except Exception as exc:
                    if p_code:
                        save_code(out_dir, "param", f"llm_model_{i + 1}_param_{j + 1}", p_code, accepted=False, reason=repr(exc))
                    log_event(out_dir, {"stage": "param_failed", "model_iter": i + 1, "param_iter": j + 1, "error": repr(exc)})
            candidates.append(best)
            history.append({"name": best.name, "source": best.source, "val_mae": best.val_mae, "val_rmse": best.val_rmse})
            log_event(out_dir, {"stage": "candidate", "name": best.name, "source": best.source, "val_mae": best.val_mae, "val_rmse": best.val_rmse})
        except Exception as exc:
            if code:
                save_code(out_dir, "model", f"llm_model_{i + 1}", code, accepted=False, reason=repr(exc))
            log_event(out_dir, {"stage": "model_failed", "model_iter": i + 1, "error": repr(exc)})

    candidates = sorted(candidates, key=lambda c: selection_score(c, args.selection_metric))
    selected_teachers = select_candidates(candidates, args)
    students = distill_selected(selected_teachers, X_train, X_val, X_test, y_val, y_test, args)

    rows = []
    rows.extend(ensemble_rows(candidates, y_val, y_test, args, "teacher_no_distill"))
    if students:
        rows.extend(ensemble_rows(students, y_val, y_test, args, "student_distilled"))
    if not rows:
        raise RuntimeError("no valid regression candidates")
    for row in rows:
        row.update(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "llm": args.llm,
                "feat_iterations": args.feat_iterations,
                "model_iterations": args.model_iterations,
                "param_iterations": args.param_iterations,
                "top_k": args.top_k,
                "selection_gap": args.selection_gap,
                "feature_status": feat_info.get("feature_status"),
                "feature_error": feat_info.get("feature_error"),
                "candidate_count": len(candidates),
            }
        )

    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "target": target,
        "split": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "configs": rows,
        "candidates": [asdict(c) for c in candidates],
    }
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (out_dir / "per_config.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame([asdict(c) for c in candidates]).drop(columns=["train_pred", "val_pred", "test_pred"], errors="ignore").to_csv(out_dir / "candidates.csv", index=False)
    print(json.dumps({"dataset": args.dataset, "seed": args.seed, "best_config": min(rows, key=lambda r: r["val_mae"])}, ensure_ascii=False), flush=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--llm", default="gpt-3.5-turbo")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "https://api.laozhang.ai/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--feat-iterations", type=int, default=10)
    parser.add_argument("--model-iterations", type=int, default=8)
    parser.add_argument("--param-iterations", type=int, default=8)
    parser.add_argument("--selection-metric", choices=["mae", "rmse"], default="mae")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-selected", type=int, default=2)
    parser.add_argument("--selection-gap", type=float, default=0.08)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--feature-timeout-s", type=int, default=900)
    parser.add_argument("--api-timeout-s", type=int, default=180)
    parser.add_argument("--model-timeout-s", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--distill-top-k", type=int, default=2)
    parser.add_argument("--distill-epochs", type=int, default=250)
    parser.add_argument("--calibrate-intervals", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "results_regression_strong_cached_f10_m8_p8_20260601" / "raw_runs"))
    args = parser.parse_args()
    if (args.feat_iterations > 0 or args.model_iterations > 0 or args.param_iterations > 0) and not args.api_key:
        raise ValueError("OPENAI_API_KEY or --api-key is required for live LLM runs")
    return args


if __name__ == "__main__":
    run_one(parse_args())
