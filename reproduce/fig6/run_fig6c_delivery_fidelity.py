from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pickle
import random
import re
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
COMPARE_ROOT = WORKSPACE / "autologic"
CLASS_SCRIPT = COMPARE_ROOT / "ensemble" / "classification_ensemble" / "classification_auto_ensemble.py"
REG_SCRIPT = COMPARE_ROOT / "ensemble" / "regression_ensemble" / "regression_auto_ensemble.py"

if str(COMPARE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARE_ROOT))

try:
    import cloudpickle as serializer
except Exception:  # pragma: no cover
    serializer = pickle


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CLS = import_module_from_path("panelc_cls_compare", CLASS_SCRIPT)
REG = import_module_from_path("panelc_reg_compare", REG_SCRIPT)
MODEL_GEN = importlib.import_module("utils.model_generate")
STAGE1 = importlib.import_module("mystage1.stage1")


def parse_csv_arg(value: str) -> List[str]:
    if value is None or value == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(x) for x in parse_csv_arg(value)]


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return list(args.seeds)
    return list(range(int(args.default_seed), int(args.default_seed) + int(args.exam_iterations)))


def ensure_api() -> Tuple[str, str]:
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY must be set in the runtime environment.")
    return base_url, api_key


class TokenTracker:
    def __init__(self, max_tokens: Optional[int] = None, max_tokens_per_call: Optional[int] = None):
        self.max_tokens = max_tokens
        self.max_tokens_per_call = max_tokens_per_call
        self.current_context: Dict[str, Any] = {}
        self.calls: List[Dict[str, Any]] = []

    @contextmanager
    def scope(self, **context):
        previous = self.current_context.copy()
        self.current_context.update(context)
        try:
            yield
        finally:
            self.current_context = previous

    def record(self, model: Optional[str], usage: Any):
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        row = {
            "call_index": len(self.calls) + 1,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        row.update(self.current_context)
        self.calls.append(row)
        if self.max_tokens_per_call is not None and total_tokens is not None:
            if int(total_tokens) > int(self.max_tokens_per_call):
                raise RuntimeError(
                    f"LLM token limit exceeded for one call: {int(total_tokens)} > {int(self.max_tokens_per_call)}"
                )
        if self.max_tokens is not None:
            observed = sum(int(c.get("total_tokens") or 0) for c in self.calls)
            if observed > self.max_tokens:
                raise RuntimeError(f"LLM token limit exceeded for this run: {observed} > {self.max_tokens}")

    def summary_since(self, start_index: int = 0) -> Dict[str, Any]:
        rows = self.calls[start_index:]
        return {
            "llm_calls": len(rows),
            "llm_prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in rows),
            "llm_completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
            "llm_total_tokens": sum(int(r.get("total_tokens") or 0) for r in rows),
        }


def install_openai_tracker(tracker: TokenTracker):
    original_model_gen_openai = MODEL_GEN.OpenAI
    original_stage1_openai = STAGE1.OpenAI

    class CompletionProxy:
        def __init__(self, inner):
            self._inner = inner

        def create(self, *args, **kwargs):
            response = self._inner.create(*args, **kwargs)
            tracker.record(kwargs.get("model"), getattr(response, "usage", None))
            return response

        def __getattr__(self, item):
            return getattr(self._inner, item)

    class ChatProxy:
        def __init__(self, inner):
            self._inner = inner
            self.completions = CompletionProxy(inner.completions)

        def __getattr__(self, item):
            return getattr(self._inner, item)

    class TrackingOpenAI:
        def __init__(self, *args, **kwargs):
            self._client = original_model_gen_openai(*args, **kwargs)
            self.chat = ChatProxy(self._client.chat)

        def __getattr__(self, item):
            return getattr(self._client, item)

    MODEL_GEN.OpenAI = TrackingOpenAI
    STAGE1.OpenAI = TrackingOpenAI
    return original_model_gen_openai, original_stage1_openai


def restore_openai_tracker(originals):
    original_model_gen_openai, original_stage1_openai = originals
    MODEL_GEN.OpenAI = original_model_gen_openai
    STAGE1.OpenAI = original_stage1_openai


class MockUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


def install_mock_llm(tracker: TokenTracker):
    original_cls_generate = CLS.generate_model_2
    original_reg_generate = REG.generate_model_2
    original_model_gen_generate = MODEL_GEN.generate_model_2

    class_code = """
from sklearn.ensemble import RandomForestClassifier
class myclassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=24, max_depth=6, random_state=42, n_jobs=-1, class_weight='balanced')
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)
"""
    reg_code = """
from sklearn.ensemble import RandomForestRegressor
class myregressor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=24, max_depth=6, random_state=42, n_jobs=-1)
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)
"""

    def fake_generate(model: str, messages: List[Dict[str, str]], base_url: str = None, api_key: str = None):
        task = str(tracker.current_context.get("task", "")).lower()
        joined = "\n".join(str(m.get("content", "")) for m in messages[-3:]).lower()
        is_reg = task == "regression" or "myregressor" in joined or "rmse" in joined
        prompt_tokens = max(1, sum(len(str(m.get("content", ""))) for m in messages) // 4)
        completion_tokens = len(reg_code if is_reg else class_code) // 4
        tracker.record(model or "mock-llm", MockUsage(prompt_tokens, completion_tokens))
        return {
            "code": reg_code if is_reg else class_code,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    CLS.generate_model_2 = fake_generate
    REG.generate_model_2 = fake_generate
    MODEL_GEN.generate_model_2 = fake_generate
    return original_cls_generate, original_reg_generate, original_model_gen_generate


def restore_mock_llm(originals):
    original_cls_generate, original_reg_generate, original_model_gen_generate = originals
    CLS.generate_model_2 = original_cls_generate
    REG.generate_model_2 = original_reg_generate
    MODEL_GEN.generate_model_2 = original_model_gen_generate


def append_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header:
            fieldnames = list(dict.fromkeys(existing_header + fieldnames))
    tmp = path.with_suffix(path.suffix + ".tmp")
    if exists:
        existing = pd.read_csv(path)
        for col in fieldnames:
            if col not in existing.columns:
                existing[col] = np.nan
        new_df = pd.DataFrame(rows)
        for col in fieldnames:
            if col not in new_df.columns:
                new_df[col] = np.nan
        out = pd.concat([existing[fieldnames], new_df[fieldnames]], ignore_index=True)
        out.to_csv(tmp, index=False)
    else:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    tmp.replace(path)


def append_jsonl(path: Path, row: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path, args: argparse.Namespace):
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv[1:],
        "workspace": str(WORKSPACE),
        "compare_root": str(COMPARE_ROOT),
        "classification_script": str(CLASS_SCRIPT),
        "classification_script_sha256": file_sha256(CLASS_SCRIPT),
        "regression_script": str(REG_SCRIPT),
        "regression_script_sha256": file_sha256(REG_SCRIPT),
        "tasks": args.tasks,
        "classification_datasets": args.classification_datasets,
        "regression_datasets": args.regression_datasets,
        "seeds": args.seeds,
        "llm": args.llm,
        "feat_iterations": args.feat_iterations,
        "model_iterations": args.model_iterations,
        "param_iterations": args.param_iterations,
        "enable_optimization": args.enable_optimization,
        "enable_feedback": args.enable_feedback,
        "distill_epochs": args.distill_epochs,
        "latency_repeats": args.latency_repeats,
        "latency_sample_size": args.latency_sample_size,
        "stage_timeout_s": args.stage_timeout_s,
        "max_run_seconds": args.max_run_seconds,
        "max_llm_tokens_per_run": args.max_llm_tokens_per_run,
        "max_llm_tokens_per_call": args.max_llm_tokens_per_call,
        "mock_llm": args.mock_llm,
        "api_base_url_from_env": bool(os.getenv("OPENAI_BASE_URL")),
        "api_key_from_env": bool(os.getenv("OPENAI_API_KEY")),
        "note": "API keys are intentionally not written to this manifest.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    latest_path = path.parent / "configs_manifest_latest.json"
    latest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not path.exists():
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    append_jsonl(path.parent / "configs_manifest_runs.jsonl", manifest)


def completed_keys(metrics_path: Path) -> set:
    if not metrics_path.exists():
        return set()
    df = pd.read_csv(metrics_path)
    required = {"task", "dataset", "seed", "status"}
    if not required.issubset(df.columns):
        return set()
    done = df[df["status"].astype(str).str.lower().eq("success")]
    return set(zip(done["task"].astype(str), done["dataset"].astype(str), done["seed"].astype(int)))


def safe_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim > 1 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    return arr


def check_runtime(args: argparse.Namespace, start_time: float, stage: str):
    if getattr(args, "max_run_seconds", 0) and args.max_run_seconds > 0:
        elapsed = time.time() - start_time
        if elapsed > args.max_run_seconds:
            raise RuntimeError(f"Run time limit exceeded at {stage}: {elapsed:.1f}s > {args.max_run_seconds}s")


def finite_probs(prob: Any) -> np.ndarray:
    arr = np.asarray(prob, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] >= 2:
            arr = arr[:, 1]
        else:
            arr = arr[:, 0]
    arr = arr.reshape(-1)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(arr, 1e-6, 1 - 1e-6)


def safe_predict_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    try:
        return finite_probs(model.predict_proba(x))
    except Exception:
        return finite_probs(model.predict_proba(np.asarray(x, dtype=float)))


def safe_predict(model: Any, x: pd.DataFrame) -> np.ndarray:
    try:
        pred = model.predict(x)
    except Exception:
        pred = model.predict(np.asarray(x, dtype=float))
    return safe_array(pred).astype(float)


def ece_binary(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = finite_probs(prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def binary_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = finite_probs(p)
    q = finite_probs(q)
    p2 = np.vstack([1 - p, p]).T
    q2 = np.vstack([1 - q, q]).T
    m = 0.5 * (p2 + q2)
    js = 0.5 * np.sum(p2 * np.log(p2 / m), axis=1) + 0.5 * np.sum(q2 * np.log(q2 / m), axis=1)
    return float(np.mean(js))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    val = spearmanr(a, b, nan_policy="omit").correlation
    return float(val) if val is not None else float("nan")


def top_overlap(a: np.ndarray, b: np.ndarray, frac: float = 0.10) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    k = max(1, int(round(n * frac)))
    top_a = set(np.argsort(np.asarray(a))[-k:].tolist())
    top_b = set(np.argsort(np.asarray(b))[-k:].tolist())
    return float(len(top_a & top_b) / k)


def classification_summary(y_true: np.ndarray, prob: np.ndarray, prefix: str) -> Dict[str, Any]:
    p = finite_probs(prob)
    pred = (p >= 0.5).astype(int)
    out: Dict[str, Any] = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, pred)),
        f"{prefix}_brier": float(brier_score_loss(y_true, p)),
        f"{prefix}_ece": float(ece_binary(y_true, p)),
        f"{prefix}_nll": float(log_loss(y_true, p, labels=[0, 1])),
    }
    try:
        out[f"{prefix}_auc"] = float(roc_auc_score(y_true, p))
    except Exception:
        out[f"{prefix}_auc"] = float("nan")
    return out


def regression_summary(y_true: np.ndarray, pred: np.ndarray, prefix: str) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    rmse = math.sqrt(float(mean_squared_error(y, p)))
    out = {
        f"{prefix}_mae": float(mean_absolute_error(y, p)),
        f"{prefix}_rmse": rmse,
    }
    try:
        out[f"{prefix}_r2"] = float(r2_score(y, p))
    except Exception:
        out[f"{prefix}_r2"] = float("nan")
    return out


def object_size_bytes(obj: Any) -> Optional[int]:
    payload = portable_payload(obj)
    try:
        return len(serializer.dumps(payload))
    except Exception:
        return None


def portable_payload(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: portable_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [portable_payload(v) for v in obj]
    if isinstance(obj, torch.nn.Module):
        return {
            "type": type(obj).__name__,
            "state_dict": {k: v.detach().cpu().numpy() for k, v in obj.state_dict().items()},
        }
    if hasattr(obj, "model") and isinstance(getattr(obj, "model"), torch.nn.Module):
        return {
            "type": type(obj).__name__,
            "model": portable_payload(getattr(obj, "model")),
            "x_scaler": getattr(obj, "x_scaler", None),
            "y_scaler": getattr(obj, "y_scaler", None),
        }
    if hasattr(obj, "model"):
        inner = getattr(obj, "model")
        if inner is not obj:
            return {"type": type(obj).__name__, "model": portable_payload(inner)}
    return obj


def maybe_sample_x(x: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(x) <= max_rows:
        return x
    return x.sample(n=max_rows, random_state=seed)


def measure_latency_ms_per_sample(predict_fn, x: pd.DataFrame, repeats: int) -> Optional[float]:
    if repeats <= 0 or len(x) == 0:
        return None
    try:
        predict_fn(x)
        start = time.perf_counter()
        for _ in range(repeats):
            predict_fn(x)
        elapsed = time.perf_counter() - start
        return float(1000.0 * elapsed / (repeats * len(x)))
    except Exception:
        return None


def code_hash(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def save_generated_code(args: argparse.Namespace, task: str, dataset: str, seed: int, model_index: int, code: str) -> Optional[str]:
    out_dir = getattr(args, "_current_output_dir", None)
    if out_dir is None or not code:
        return None
    code_dir = Path(out_dir) / "generated_code" / safe_name(task) / safe_name(dataset) / f"seed_{seed}"
    code_dir.mkdir(parents=True, exist_ok=True)
    path = code_dir / f"model_{model_index:02d}_{code_hash(code)}.py"
    path.write_text(code, encoding="utf-8")
    return str(path)


def rename_class(code: str, base_name: str, new_name: str) -> str:
    pattern = rf"class\s+{base_name}[_\w]*\s*(\([^)]*\))?\s*:"
    return re.sub(pattern, f"class {new_name}:", code, count=1)


def fit_class_stack(models: Sequence[Any], x_meta: pd.DataFrame, y_meta: Sequence[int]) -> Dict[str, Any]:
    if len(models) == 0:
        raise RuntimeError("No classification models were generated.")
    meta = np.column_stack([safe_predict_proba(model, x_meta) for model in models])
    if meta.shape[1] == 1:
        stack = None
    else:
        stack = LogisticRegression(max_iter=1000, class_weight="balanced")
        stack.fit(meta, np.asarray(y_meta, dtype=int))
    return {"models": list(models), "stack": stack}


def predict_class_stack(artifact: Dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    meta = np.column_stack([safe_predict_proba(model, x) for model in artifact["models"]])
    stack = artifact.get("stack")
    if stack is None:
        return finite_probs(meta[:, 0])
    return finite_probs(stack.predict_proba(meta))


def fit_reg_stack(models: Sequence[Any], x_meta: pd.DataFrame, y_meta: Sequence[float]) -> Dict[str, Any]:
    if len(models) == 0:
        raise RuntimeError("No regression models were generated.")
    meta = np.column_stack([safe_predict(model, x_meta) for model in models])
    if meta.shape[1] == 1:
        stack = None
    else:
        stack = RidgeCV(alphas=np.array([0.01, 0.1, 1.0, 10.0, 100.0]))
        stack.fit(meta, np.asarray(y_meta, dtype=float))
    return {"models": list(models), "stack": stack}


def predict_reg_stack(artifact: Dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    meta = np.column_stack([safe_predict(model, x) for model in artifact["models"]])
    stack = artifact.get("stack")
    if stack is None:
        return np.asarray(meta[:, 0], dtype=float)
    return np.asarray(stack.predict(meta), dtype=float).reshape(-1)


def model_prompt_system(task: str) -> str:
    if task == "classification":
        return (
            "You are a top-level machine learning classification expert.\n"
            "Your task is to iteratively search for the most suitable binary classifier.\n"
            "The primary goal is to maximize validation AUC.\n"
            "Your answer must contain only valid executable Python code for a class named myclassifier.\n"
            "The class must implement fit, predict, and predict_proba.\n"
            "Use efficient hyperparameters and parallel CPU settings such as n_jobs=-1 where applicable.\n"
            "The sklearn version may be recent, so avoid obsolete parameters such as base_estimator.\n"
        )
    return (
        "You are a top-level regression algorithm expert.\n"
        "Your task is to iteratively search for the most suitable regressor.\n"
        "The primary goal is to minimize validation RMSE.\n"
        "Your answer must contain only valid executable Python code for a class named myregressor.\n"
        "The class must implement fit and predict.\n"
        "Use efficient hyperparameters and parallel CPU settings such as n_jobs=-1 where applicable.\n"
        "The sklearn version may be recent, so avoid obsolete parameters such as base_estimator.\n"
    )


def make_reg_param_prompt(best_code: str, rmse: float, mae: float) -> str:
    return f"""
Here is the best regression model code so far.
Current validation RMSE: {rmse:.6f}
Current validation MAE: {mae:.6f}

Model code:
```python
{best_code}
```

Optimize only hyperparameters to improve validation RMSE.
Do not change the required class API.
Return only executable Python code for a class named myregressor.
"""


def optimization_system(task: str) -> str:
    if task == "classification":
        return (
            "You are a classification optimization assistant.\n"
            "Tune hyperparameters only to improve validation AUC.\n"
            "Return only executable Python code for a class named myclassifier.\n"
            "Preserve or add n_jobs=-1 where supported.\n"
            "Use bounded values: n_estimators <= 300, max_depth <= 10, num_leaves <= 64, max_iter <= 500.\n"
        )
    return (
        "You are a regression optimization assistant.\n"
        "Tune hyperparameters only to improve validation RMSE.\n"
        "Return only executable Python code for a class named myregressor.\n"
        "Preserve or add n_jobs=-1 where supported.\n"
        "Use bounded values: n_estimators <= 300, max_depth <= 10, num_leaves <= 64, max_iter <= 500.\n"
    )


def run_generated_classifier(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    base_url: str,
    api_key: str,
    tracker: TokenTracker,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    start = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df_train, df_test, target, desc = CLS.load_origin_data(dataset, seed=seed)
    baseline = CLS.base_model(seed)

    with tracker.scope(task="classification", dataset=dataset, seed=seed, stage="feature_generation"):
        if args.feat_iterations > 0:
            df_train, df_test = CLS.generate_feat(
                base_classifier=baseline,
                df_train=df_train,
                df_test=df_test,
                dataset_name=dataset,
                round_num=1,
                llm_model=args.llm,
                iterations=args.feat_iterations,
                target_column_name=target,
                dataset_description=desc,
                base_url=base_url,
                api_key=api_key,
                logger=None,
            )
    check_runtime(args, start, "classification_feature_generation")

    df_train, df_meta = train_test_split(
        df_train,
        test_size=args.meta_fraction,
        random_state=seed,
        stratify=df_train[target],
    )
    x_train, y_train = CLS.to_pd(df_train, target)
    x_meta, y_meta = CLS.to_pd(df_meta, target)
    x_test, y_test = CLS.to_pd(df_test, target)

    samples = CLS.build_prompt_samples(df_train)
    prompt = CLS.get_model_prompt(target_column_name=target, samples=samples)
    messages = [{"role": "system", "content": model_prompt_system("classification")}, {"role": "user", "content": prompt}]

    teacher_models: List[Any] = []
    student_models: List[Any] = []
    inventory: List[Dict[str, Any]] = []
    best_auc = 0.0
    best_code = ""
    model_index = 0
    attempts_for_current = 0
    i = 0
    while i < args.model_iterations:
        check_runtime(args, start, f"classification_model_iter_{i + 1}_start")
        attempts_for_current += 1
        if attempts_for_current > args.max_retries_per_model:
            inventory.append(
                {
                    "task": "classification",
                    "dataset": dataset,
                    "seed": seed,
                    "model_index": i + 1,
                    "role": "teacher",
                    "status": "skipped",
                    "skip_reason": "retry_limit",
                }
            )
            i += 1
            attempts_for_current = 0
            continue

        with tracker.scope(task="classification", dataset=dataset, seed=seed, stage="model_generation"):
            res = CLS.generate_model_2(args.llm, messages, base_url, api_key)
        check_runtime(args, start, f"classification_model_iter_{i + 1}_llm_done")
        raw_code = res["code"]
        code = rename_class(CLS.clean_llm_code(raw_code), "myclassifier", f"myclassifier_{i + 1}")
        err, scope = CLS.code_exec(code)
        if err is not None or scope is None:
            messages += [{"role": "assistant", "content": code}, {"role": "user", "content": f"Code execution failed with error: {err}. Fix it and return only code."}]
            continue
        try:
            model_cls = scope[f"myclassifier_{i + 1}"]
            model = model_cls()
            model.fit(x_train, y_train)
            model_copy = copy.deepcopy(model)
            prob = safe_predict_proba(model_copy, x_meta)
            val_auc = float(roc_auc_score(y_meta, prob))
            val_brier = float(brier_score_loss(y_meta, prob))
        except Exception as exc:
            messages += [
                {"role": "assistant", "content": code},
                {"role": "user", "content": f"Model runtime failed with error: {type(exc).__name__}: {exc}. Fix it and return only code."},
            ]
            continue

        attempts_for_current = 0
        param_best_model = model_copy
        param_best_code = code
        param_best_auc = val_auc

        if args.enable_optimization and args.param_iterations > 0:
            param_prompt = CLS.get_classification_param_prompt(
                best_code=param_best_code,
                best_auc=param_best_auc * 100.0,
                dataset_description=desc,
                X_test=x_meta,
                feature_columns=x_train.columns.tolist(),
                dataset_name=dataset,
                max_rows=10,
            )
            param_messages = [{"role": "system", "content": optimization_system("classification")}, {"role": "user", "content": param_prompt}]
            for p_iter in range(args.param_iterations):
                check_runtime(args, start, f"classification_model_iter_{i + 1}_param_{p_iter + 1}_start")
                with tracker.scope(task="classification", dataset=dataset, seed=seed, stage="param_optimization"):
                    pres = CLS.generate_model_2(args.llm, param_messages, base_url, api_key)
                check_runtime(args, start, f"classification_model_iter_{i + 1}_param_{p_iter + 1}_llm_done")
                pcode = rename_class(CLS.clean_llm_code(pres["code"]), "myclassifier", f"myclassifier_{i + 1}_param_{p_iter + 1}")
                perr, pscope = CLS.code_exec(pcode)
                if perr is not None or pscope is None:
                    param_messages += [{"role": "assistant", "content": pcode}, {"role": "user", "content": f"Compilation failed: {perr}. Fix it and return only code."}]
                    continue
                try:
                    pcls = pscope[f"myclassifier_{i + 1}_param_{p_iter + 1}"]
                    tuned = pcls()
                    tuned.fit(x_train, y_train)
                    tuned_copy = copy.deepcopy(tuned)
                    tuned_prob = safe_predict_proba(tuned_copy, x_meta)
                    tuned_auc = float(roc_auc_score(y_meta, tuned_prob))
                    if tuned_auc > param_best_auc:
                        param_best_auc = tuned_auc
                        param_best_code = pcode
                        param_best_model = tuned_copy
                    param_messages += [
                        {"role": "assistant", "content": pcode},
                        {"role": "user", "content": f"Current AUC: {tuned_auc:.6f}; best AUC: {param_best_auc:.6f}. Improve further by tuning hyperparameters only."},
                    ]
                except Exception as exc:
                    param_messages += [
                        {"role": "assistant", "content": pcode},
                        {"role": "user", "content": f"Runtime failed: {type(exc).__name__}: {exc}. Fix it and return only code."},
                    ]

        teacher_models.append(param_best_model)
        model_index += 1
        distill_success = False
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            student = CLS.distill_to_student(param_best_model, x_train, y_train, x_meta, y_meta, device=device, epochs=args.distill_epochs)
            student_models.append(student)
            distill_success = True
        except Exception:
            student_models.append(param_best_model)
        check_runtime(args, start, f"classification_model_iter_{i + 1}_distill_done")

        best_auc = max(best_auc, param_best_auc)
        best_code = param_best_code if param_best_auc >= best_auc else best_code
        code_path = save_generated_code(args, "classification", dataset, seed, model_index, param_best_code)
        inventory.append(
            {
                "task": "classification",
                "dataset": dataset,
                "seed": seed,
                "model_index": model_index,
                "role": "teacher_student_pair",
                "status": "success",
                "val_auc": param_best_auc,
                "val_brier_initial": val_brier,
                "distill_success": distill_success,
                "code_len": len(param_best_code),
                "code_sha256": code_hash(param_best_code),
                "code_path": code_path,
                "teacher_type": type(param_best_model).__name__,
                "student_type": type(student_models[-1]).__name__,
            }
        )

        if args.enable_feedback:
            feedback = f"Classifier executed successfully. Current validation AUC: {param_best_auc:.6f}; best historical AUC: {best_auc:.6f}. Propose a different classifier likely to improve AUC. Return only code."
        else:
            feedback = "Classifier executed successfully. Propose a different classifier. Return only code."
        messages += [{"role": "assistant", "content": param_best_code}, {"role": "user", "content": feedback}]
        i += 1

    check_runtime(args, start, "classification_before_stack")
    teacher_artifact = fit_class_stack(teacher_models, x_meta, y_meta)
    student_artifact = fit_class_stack(student_models, x_meta, y_meta)
    teacher_prob = predict_class_stack(teacher_artifact, x_test)
    student_prob = predict_class_stack(student_artifact, x_test)

    y_arr = np.asarray(y_test, dtype=int)
    latency_x = maybe_sample_x(x_test, args.latency_sample_size, seed)
    teacher_latency = measure_latency_ms_per_sample(lambda z: predict_class_stack(teacher_artifact, z), latency_x, args.latency_repeats)
    student_latency = measure_latency_ms_per_sample(lambda z: predict_class_stack(student_artifact, z), latency_x, args.latency_repeats)
    teacher_size = object_size_bytes(teacher_artifact)
    student_size = object_size_bytes(student_artifact)

    metrics: Dict[str, Any] = {
        "task": "classification",
        "dataset": dataset,
        "seed": seed,
        "status": "success",
        "seconds": time.time() - start,
        "n_train_base": len(x_train),
        "n_meta": len(x_meta),
        "n_test": len(x_test),
        "n_features": x_train.shape[1],
        "teacher_model_count": len(teacher_models),
        "student_model_count": len(student_models),
        "distill_success_count": sum(1 for r in inventory if r.get("distill_success") is True),
        "teacher_artifact_bytes": teacher_size,
        "student_artifact_bytes": student_size,
        "artifact_size_ratio_student_over_teacher": (student_size / teacher_size) if teacher_size and student_size else None,
        "teacher_latency_ms_per_sample": teacher_latency,
        "student_latency_ms_per_sample": student_latency,
        "latency_ratio_student_over_teacher": (student_latency / teacher_latency) if teacher_latency and student_latency else None,
    }
    metrics.update(classification_summary(y_arr, teacher_prob, "teacher"))
    metrics.update(classification_summary(y_arr, student_prob, "student"))
    metrics.update(
        {
            "fidelity_prob_pearson": pearson_corr(teacher_prob, student_prob),
            "fidelity_prob_spearman": spearman_corr(teacher_prob, student_prob),
            "fidelity_prob_mae": float(np.mean(np.abs(teacher_prob - student_prob))),
            "fidelity_prob_rmse": float(math.sqrt(np.mean((teacher_prob - student_prob) ** 2))),
            "fidelity_prob_js": binary_js_divergence(teacher_prob, student_prob),
            "fidelity_top10_overlap": top_overlap(teacher_prob, student_prob, 0.10),
            "auc_delta_student_minus_teacher": None,
            "ece_delta_student_minus_teacher": None,
            "nll_delta_student_minus_teacher": None,
        }
    )
    metrics["auc_delta_student_minus_teacher"] = metrics["student_auc"] - metrics["teacher_auc"]
    metrics["ece_delta_student_minus_teacher"] = metrics["student_ece"] - metrics["teacher_ece"]
    metrics["nll_delta_student_minus_teacher"] = metrics["student_nll"] - metrics["teacher_nll"]

    predictions = [
        {
            "task": "classification",
            "dataset": dataset,
            "seed": seed,
            "sample_id": idx,
            "y_true": int(y_arr[idx]),
            "teacher_prob": float(teacher_prob[idx]),
            "student_prob": float(student_prob[idx]),
            "teacher_pred": int(teacher_prob[idx] >= 0.5),
            "student_pred": int(student_prob[idx] >= 0.5),
        }
        for idx in range(len(y_arr))
    ]
    return metrics, predictions, inventory


def run_generated_regressor(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    base_url: str,
    api_key: str,
    tracker: TokenTracker,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    start = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    loc = str(COMPARE_ROOT / "data" / f"{dataset}.pkl")
    df_train, df_test, target, desc = REG.load_origin_data(loc, seed)
    baseline = REG.base_model(seed)

    with tracker.scope(task="regression", dataset=dataset, seed=seed, stage="feature_generation"):
        if args.feat_iterations > 0:
            df_train, df_test = REG.generate_feat(
                base_model=baseline,
                df_train=df_train,
                df_test=df_test,
                dataset_name=dataset,
                round_num=1,
                llm_model=args.llm,
                iterations=args.feat_iterations,
                target_column_name=target,
                dataset_description=desc,
                task_type="regression",
                base_url=base_url,
                api_key=api_key,
                logger=None,
            )
    check_runtime(args, start, "regression_feature_generation")

    df_train, df_meta = train_test_split(df_train, test_size=args.meta_fraction, random_state=seed)
    x_train, y_train = REG.to_pd(df_train, target)
    x_meta, y_meta = REG.to_pd(df_meta, target)
    x_test, y_test = REG.to_pd(df_test, target)

    samples = REG.build_prompt_samples(df_train)
    prompt = REG.get_regression_model_prompt(target_column_name=target, samples=samples)
    messages = [{"role": "system", "content": model_prompt_system("regression")}, {"role": "user", "content": prompt}]

    teacher_models: List[Any] = []
    student_models: List[Any] = []
    inventory: List[Dict[str, Any]] = []
    best_rmse = float("inf")
    model_index = 0
    attempts_for_current = 0
    i = 0
    while i < args.model_iterations:
        check_runtime(args, start, f"regression_model_iter_{i + 1}_start")
        attempts_for_current += 1
        if attempts_for_current > args.max_retries_per_model:
            inventory.append(
                {
                    "task": "regression",
                    "dataset": dataset,
                    "seed": seed,
                    "model_index": i + 1,
                    "role": "teacher",
                    "status": "skipped",
                    "skip_reason": "retry_limit",
                }
            )
            i += 1
            attempts_for_current = 0
            continue

        with tracker.scope(task="regression", dataset=dataset, seed=seed, stage="model_generation"):
            res = REG.generate_model_2(args.llm, messages, base_url, api_key)
        check_runtime(args, start, f"regression_model_iter_{i + 1}_llm_done")
        raw_code = res["code"]
        code = rename_class(REG.clean_llm_code(raw_code), "myregressor", f"myregressor_{i + 1}")
        err, scope = REG.code_exec(code)
        if err is not None or scope is None:
            messages += [{"role": "assistant", "content": code}, {"role": "user", "content": f"Code execution failed with error: {err}. Fix it and return only code."}]
            continue
        try:
            model_cls = scope[f"myregressor_{i + 1}"]
            model = model_cls()
            model.fit(x_train, y_train)
            model_copy = copy.deepcopy(model)
            pred = safe_predict(model_copy, x_meta)
            val_rmse = float(math.sqrt(mean_squared_error(y_meta, pred)))
            val_mae = float(mean_absolute_error(y_meta, pred))
        except Exception as exc:
            messages += [
                {"role": "assistant", "content": code},
                {"role": "user", "content": f"Model runtime failed with error: {type(exc).__name__}: {exc}. Fix it and return only code."},
            ]
            continue

        attempts_for_current = 0
        param_best_model = model_copy
        param_best_code = code
        param_best_rmse = val_rmse
        param_best_mae = val_mae

        if args.enable_optimization and args.param_iterations > 0:
            param_messages = [
                {"role": "system", "content": optimization_system("regression")},
                {"role": "user", "content": make_reg_param_prompt(param_best_code, val_rmse, val_mae)},
            ]
            for p_iter in range(args.param_iterations):
                check_runtime(args, start, f"regression_model_iter_{i + 1}_param_{p_iter + 1}_start")
                with tracker.scope(task="regression", dataset=dataset, seed=seed, stage="param_optimization"):
                    pres = REG.generate_model_2(args.llm, param_messages, base_url, api_key)
                check_runtime(args, start, f"regression_model_iter_{i + 1}_param_{p_iter + 1}_llm_done")
                pcode = rename_class(REG.clean_llm_code(pres["code"]), "myregressor", f"myregressor_{i + 1}_param_{p_iter + 1}")
                perr, pscope = REG.code_exec(pcode)
                if perr is not None or pscope is None:
                    param_messages += [{"role": "assistant", "content": pcode}, {"role": "user", "content": f"Compilation failed: {perr}. Fix it and return only code."}]
                    continue
                try:
                    pcls = pscope[f"myregressor_{i + 1}_param_{p_iter + 1}"]
                    tuned = pcls()
                    tuned.fit(x_train, y_train)
                    tuned_copy = copy.deepcopy(tuned)
                    tuned_pred = safe_predict(tuned_copy, x_meta)
                    tuned_rmse = float(math.sqrt(mean_squared_error(y_meta, tuned_pred)))
                    tuned_mae = float(mean_absolute_error(y_meta, tuned_pred))
                    if tuned_rmse < param_best_rmse:
                        param_best_rmse = tuned_rmse
                        param_best_mae = tuned_mae
                        param_best_code = pcode
                        param_best_model = tuned_copy
                    param_messages += [
                        {"role": "assistant", "content": pcode},
                        {"role": "user", "content": f"Current RMSE: {tuned_rmse:.6f}; best RMSE: {param_best_rmse:.6f}. Improve further by tuning hyperparameters only."},
                    ]
                except Exception as exc:
                    param_messages += [
                        {"role": "assistant", "content": pcode},
                        {"role": "user", "content": f"Runtime failed: {type(exc).__name__}: {exc}. Fix it and return only code."},
                    ]

        teacher_models.append(param_best_model)
        model_index += 1
        distill_success = False
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            student = REG.distill_to_student_regression(param_best_model, x_train, y_train, x_meta, y_meta, device=device, epochs=args.distill_epochs)
            student_models.append(student)
            distill_success = True
        except Exception:
            student_models.append(param_best_model)
        check_runtime(args, start, f"regression_model_iter_{i + 1}_distill_done")

        best_rmse = min(best_rmse, param_best_rmse)
        code_path = save_generated_code(args, "regression", dataset, seed, model_index, param_best_code)
        inventory.append(
            {
                "task": "regression",
                "dataset": dataset,
                "seed": seed,
                "model_index": model_index,
                "role": "teacher_student_pair",
                "status": "success",
                "val_rmse": param_best_rmse,
                "val_mae": param_best_mae,
                "distill_success": distill_success,
                "code_len": len(param_best_code),
                "code_sha256": code_hash(param_best_code),
                "code_path": code_path,
                "teacher_type": type(param_best_model).__name__,
                "student_type": type(student_models[-1]).__name__,
            }
        )

        if args.enable_feedback:
            feedback = f"Regressor executed successfully. Current validation RMSE: {param_best_rmse:.6f}; best historical RMSE: {best_rmse:.6f}. Propose a different regressor likely to improve RMSE. Return only code."
        else:
            feedback = "Regressor executed successfully. Propose a different regressor. Return only code."
        messages += [{"role": "assistant", "content": param_best_code}, {"role": "user", "content": feedback}]
        i += 1

    check_runtime(args, start, "regression_before_stack")
    teacher_artifact = fit_reg_stack(teacher_models, x_meta, y_meta)
    student_artifact = fit_reg_stack(student_models, x_meta, y_meta)
    teacher_pred = predict_reg_stack(teacher_artifact, x_test)
    student_pred = predict_reg_stack(student_artifact, x_test)

    y_arr = np.asarray(y_test, dtype=float)
    latency_x = maybe_sample_x(x_test, args.latency_sample_size, seed)
    teacher_latency = measure_latency_ms_per_sample(lambda z: predict_reg_stack(teacher_artifact, z), latency_x, args.latency_repeats)
    student_latency = measure_latency_ms_per_sample(lambda z: predict_reg_stack(student_artifact, z), latency_x, args.latency_repeats)
    teacher_size = object_size_bytes(teacher_artifact)
    student_size = object_size_bytes(student_artifact)
    pred_rmse = float(math.sqrt(np.mean((teacher_pred - student_pred) ** 2)))
    y_std = float(np.std(y_arr)) if float(np.std(y_arr)) > 1e-12 else float("nan")

    metrics: Dict[str, Any] = {
        "task": "regression",
        "dataset": dataset,
        "seed": seed,
        "status": "success",
        "seconds": time.time() - start,
        "n_train_base": len(x_train),
        "n_meta": len(x_meta),
        "n_test": len(x_test),
        "n_features": x_train.shape[1],
        "teacher_model_count": len(teacher_models),
        "student_model_count": len(student_models),
        "distill_success_count": sum(1 for r in inventory if r.get("distill_success") is True),
        "teacher_artifact_bytes": teacher_size,
        "student_artifact_bytes": student_size,
        "artifact_size_ratio_student_over_teacher": (student_size / teacher_size) if teacher_size and student_size else None,
        "teacher_latency_ms_per_sample": teacher_latency,
        "student_latency_ms_per_sample": student_latency,
        "latency_ratio_student_over_teacher": (student_latency / teacher_latency) if teacher_latency and student_latency else None,
    }
    metrics.update(regression_summary(y_arr, teacher_pred, "teacher"))
    metrics.update(regression_summary(y_arr, student_pred, "student"))
    metrics.update(
        {
            "fidelity_pred_pearson": pearson_corr(teacher_pred, student_pred),
            "fidelity_pred_spearman": spearman_corr(teacher_pred, student_pred),
            "fidelity_pred_mae": float(np.mean(np.abs(teacher_pred - student_pred))),
            "fidelity_pred_rmse": pred_rmse,
            "fidelity_pred_nrmse_y_std": float(pred_rmse / y_std) if np.isfinite(y_std) else float("nan"),
            "teacher_student_top10_pred_overlap": top_overlap(teacher_pred, student_pred, 0.10),
            "rmse_delta_student_minus_teacher": None,
            "mae_delta_student_minus_teacher": None,
            "r2_delta_student_minus_teacher": None,
        }
    )
    metrics["rmse_delta_student_minus_teacher"] = metrics["student_rmse"] - metrics["teacher_rmse"]
    metrics["mae_delta_student_minus_teacher"] = metrics["student_mae"] - metrics["teacher_mae"]
    metrics["r2_delta_student_minus_teacher"] = metrics["student_r2"] - metrics["teacher_r2"]

    predictions = [
        {
            "task": "regression",
            "dataset": dataset,
            "seed": seed,
            "sample_id": idx,
            "y_true": float(y_arr[idx]),
            "teacher_pred": float(teacher_pred[idx]),
            "student_pred": float(student_pred[idx]),
        }
        for idx in range(len(y_arr))
    ]
    return metrics, predictions, inventory


def run_one(
    args: argparse.Namespace,
    task: str,
    dataset: str,
    seed: int,
    base_url: str,
    api_key: str,
    output_dir: Path,
):
    metrics_path = output_dir / "per_run_metrics.csv"
    class_pred_path = output_dir / "classification_predictions.csv"
    reg_pred_path = output_dir / "regression_predictions.csv"
    inventory_path = output_dir / "model_inventory.csv"
    llm_path = output_dir / "llm_call_records.csv"
    status_path = output_dir / "run_status.jsonl"

    key = (task, dataset, seed)
    if not args.force and key in completed_keys(metrics_path):
        append_jsonl(status_path, {"event": "skip_completed", "task": task, "dataset": dataset, "seed": seed, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        return

    append_jsonl(status_path, {"event": "start", "task": task, "dataset": dataset, "seed": seed, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    tracker = TokenTracker(max_tokens=args.max_llm_tokens_per_run, max_tokens_per_call=args.max_llm_tokens_per_call)
    args._current_output_dir = output_dir
    if args.mock_llm:
        originals = install_mock_llm(tracker)
        restore_fn = restore_mock_llm
    else:
        originals = install_openai_tracker(tracker)
        restore_fn = restore_openai_tracker
    start_call = len(tracker.calls)
    try:
        if task == "classification":
            metrics, predictions, inventory = run_generated_classifier(args, dataset, seed, base_url, api_key, tracker)
            pred_path = class_pred_path
        elif task == "regression":
            metrics, predictions, inventory = run_generated_regressor(args, dataset, seed, base_url, api_key, tracker)
            pred_path = reg_pred_path
        else:
            raise ValueError(f"Unsupported task: {task}")
        metrics.update(tracker.summary_since(start_call))
        append_rows_csv(metrics_path, [metrics])
        append_rows_csv(pred_path, predictions)
        append_rows_csv(inventory_path, inventory)
        append_rows_csv(llm_path, tracker.calls[start_call:])
        append_jsonl(status_path, {"event": "success", "task": task, "dataset": dataset, "seed": seed, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "llm_total_tokens": metrics.get("llm_total_tokens")})
    except Exception as exc:
        error_row = {
            "task": task,
            "dataset": dataset,
            "seed": seed,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        error_row.update(tracker.summary_since(start_call))
        append_rows_csv(metrics_path, [error_row])
        append_rows_csv(llm_path, tracker.calls[start_call:])
        append_jsonl(
            status_path,
            {
                "event": "failed",
                "task": task,
                "dataset": dataset,
                "seed": seed,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            },
        )
        raise
    finally:
        restore_fn(originals)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Panel c delivery-fidelity runner for AutoLOGIC compare.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tasks", type=parse_csv_arg, default=["classification", "regression"])
    parser.add_argument("--classification-datasets", type=parse_csv_arg, default=["cc1", "credit-g", "ld1"])
    parser.add_argument("--regression-datasets", type=parse_csv_arg, default=["boston", "concrete", "california"])
    parser.add_argument("--seeds", type=parse_int_csv, default=None, help="Comma-separated explicit seeds. Overrides -s/-e.")
    parser.add_argument("-s", "--default-seed", "--default_seed", dest="default_seed", type=int, default=42)
    parser.add_argument("-e", "--exam-iterations", "--exam_iterations", dest="exam_iterations", type=int, default=1)
    parser.add_argument("-l", "--llm", default="gpt-3.5-turbo")
    parser.add_argument("-f", "--feat-iterations", "--feat_iterations", dest="feat_iterations", type=int, default=10)
    parser.add_argument("-m", "--model-iterations", "--model_iterations", dest="model_iterations", type=int, default=10)
    parser.add_argument("-p", "--param-iterations", "--param_iterations", dest="param_iterations", type=int, default=5)
    parser.add_argument("--enable-optimization", "--enable_optimization", dest="enable_optimization", action="store_true", default=True)
    parser.add_argument("--disable-optimization", "--disable_optimization", dest="enable_optimization", action="store_false")
    parser.add_argument("--enable-feedback", "--enable_feedback", dest="enable_feedback", action="store_true", default=True)
    parser.add_argument("--disable-feedback", "--disable_feedback", dest="enable_feedback", action="store_false")
    parser.add_argument("--distill-epochs", type=int, default=30)
    parser.add_argument("--meta-fraction", type=float, default=0.25)
    parser.add_argument("--latency-repeats", type=int, default=30)
    parser.add_argument("--latency-sample-size", type=int, default=512)
    parser.add_argument("--stage-timeout-s", type=int, default=300)
    parser.add_argument("--max-run-seconds", type=int, default=3600)
    parser.add_argument("--max-retries-per-model", type=int, default=3)
    parser.add_argument("--max-llm-tokens-per-run", type=int, default=200000)
    parser.add_argument("--max-llm-tokens-per-call", type=int, default=30000)
    parser.add_argument("--mock-llm", action="store_true", help="Use local deterministic model code for offline smoke only.")
    parser.add_argument("--force", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.seeds = resolve_seeds(args)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "configs_manifest.json", args)

    CLS.STAGE_TIMEOUT_S = int(args.stage_timeout_s)
    REG.STAGE_TIMEOUT_S = int(args.stage_timeout_s)

    if args.mock_llm:
        if args.feat_iterations != 0:
            raise RuntimeError("--mock-llm is only valid with --feat-iterations 0; feature generation uses the external API.")
        base_url, api_key = "mock", "mock"
    else:
        base_url, api_key = ensure_api()
    tasks = args.tasks
    for seed in args.seeds:
        if "classification" in tasks:
            for dataset in args.classification_datasets:
                run_one(args, "classification", dataset, seed, base_url, api_key, output_dir)
        if "regression" in tasks:
            for dataset in args.regression_datasets:
                run_one(args, "regression", dataset, seed, base_url, api_key, output_dir)


if __name__ == "__main__":
    main()
