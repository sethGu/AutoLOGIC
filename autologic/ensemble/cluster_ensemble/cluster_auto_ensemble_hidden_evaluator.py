import argparse
import ast
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
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_LARGE_DATASETS = {"cd2", "ld2", "cc3"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import AgenticReasoningAgent  # noqa: E402
from utils.task_protocol import (  # noqa: E402
    build_feature_provenance,
    materialize_task_protocol,
    write_task_protocol,
)


BANNED_MODEL_PATTERNS = [
    "spectralclustering",
    "agglomerativeclustering",
    "dbscan",
    "optics",
    "affinity=\"precomputed\"",
    "affinity='precomputed'",
    "metric=\"precomputed\"",
    "metric='precomputed'",
    "pairwise_distances",
    "distance_matrix",
    "kneighbors_graph",
    "radius_neighbors_graph",
    "np.equal.outer",
    ".equal.outer",
    "coassociation",
    "co_association",
]


@dataclass
class CandidateRecord:
    name: str
    source: str
    code: str
    val_ari: float
    val_nmi: float
    test_ari: float
    test_nmi: float
    train_labels: List[int]
    val_labels: List[int]
    test_labels: List[int]
    raw_val_ari: Optional[float] = None
    selection_repeats: int = 1
    failed: bool = False
    failure_reason: str = ""


def log_event(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        "messages": messages[-8:],
        "temperature": 0.7,
        "stop": ["```end"],
    }
    try:
        completion = client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
    except TypeError:
        completion = client.chat.completions.create(max_tokens=max_tokens, **kwargs)
    except Exception as exc:
        if "max_completion_tokens" in str(exc):
            completion = client.chat.completions.create(max_tokens=max_tokens, **kwargs)
        else:
            raise
    record_tokens(output_dir, stage, model, completion.usage)
    code = completion.choices[0].message.content or ""
    return clean_code(code)


def save_generated_code(
    output_dir: Path,
    *,
    stage: str,
    name: str,
    code: str,
    accepted: bool,
    reason: str = "",
) -> Path:
    code_dir = output_dir / "generated_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120]
    path = code_dir / f"{stage}_{safe_name}.py"
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


def write_worker_result(result_path: str, payload: Dict[str, Any]) -> None:
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def read_worker_result(result_path: str) -> Dict[str, Any]:
    with Path(result_path).open("rb") as f:
        return pickle.load(f)


def wait_for_worker_result(proc, result_path: str, timeout_s: int, stage_name: str) -> Dict[str, Any]:
    path = Path(result_path)
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if path.exists():
            proc.join(2)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(5)
            return read_worker_result(result_path)
        if not proc.is_alive():
            break
        proc.join(min(0.25, max(0.01, deadline - time.time())))
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        raise TimeoutError(f"{stage_name} stage exceeded {timeout_s}s")
    if path.exists():
        return read_worker_result(result_path)
    raise RuntimeError(f"{stage_name} worker returned no result")


def clean_code(text: str) -> str:
    code = text.replace("```python", "").replace("```", "").replace("<end>", "")
    return code.strip()


def _encode_non_numeric_features(df: pd.DataFrame, target_column_name: str) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in df.columns if c != target_column_name]
    for c in feature_cols:
        col = df[c]
        if pd.api.types.is_bool_dtype(col):
            df[c] = col.astype(int)
        elif pd.api.types.is_numeric_dtype(col):
            df[c] = pd.to_numeric(col, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
        else:
            dt = pd.to_datetime(col, errors="coerce")
            if dt.notna().any():
                df[c] = dt.astype("int64").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
            else:
                cat = pd.Categorical(col.astype("string").fillna("__missing__").str.strip())
                df[c] = pd.Series(cat.codes, index=df.index, dtype="float32")
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_dataset(dataset: str) -> Tuple[pd.DataFrame, np.ndarray, str, int, str]:
    loc = PROJECT_ROOT / "data" / f"{dataset}.pkl"
    if not loc.exists():
        raise FileNotFoundError(f"Missing dataset pkl: {loc}")
    with loc.open("rb") as f:
        ds = pickle.load(f)
    df = ds[1].copy()
    target_column_name = ds[4][-1]
    description = str(ds[-1] or "")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = _encode_non_numeric_features(df, target_column_name)
    y = LabelEncoder().fit_transform(df[target_column_name].to_numpy())
    X = df.drop(columns=[target_column_name]).copy()
    n_clusters = int(len(np.unique(y)))
    return X, y, target_column_name, n_clusters, description


def _name_set(value: str) -> set:
    return {x.strip() for x in str(value or "").split(",") if x.strip()}


def is_large_dataset_mode(args, dataset: str, n_samples: int) -> bool:
    if getattr(args, "large_dataset_strategy", "auto") == "off":
        return False
    names = _name_set(getattr(args, "large_datasets", "")) or DEFAULT_LARGE_DATASETS
    return dataset in names or int(n_samples) >= int(getattr(args, "large_sample_threshold", 12000))


def effective_selection_repeats(args) -> int:
    repeats = int(getattr(args, "selection_repeats", 1) or 1)
    if getattr(args, "large_dataset_active", False):
        repeats = min(repeats, int(getattr(args, "large_selection_repeats", 1) or 1))
    return max(1, repeats)


def _stratified_take(y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    idx = np.arange(len(y))
    max_n = int(max_n)
    if max_n <= 0 or len(idx) <= max_n:
        return idx
    counts = np.bincount(y)
    strat = y if len(counts) > 1 and counts.min() >= 2 and max_n >= len(counts) * 2 else None
    _, take = train_test_split(idx, test_size=max_n, random_state=seed, stratify=strat)
    return np.asarray(take, dtype=int)


def selection_subsets(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args,
    seed: int,
):
    if not getattr(args, "large_dataset_active", False):
        return X_train, X_val, X_test, y_train, y_val, y_test
    tr_idx = _stratified_take(y_train, getattr(args, "large_selection_train_size", 3000), seed + 101)
    va_idx = _stratified_take(y_val, getattr(args, "large_selection_val_size", 1500), seed + 202)
    te_idx = _stratified_take(y_test, getattr(args, "large_selection_test_size", 1500), seed + 303)
    return (
        X_train.iloc[tr_idx].reset_index(drop=True),
        X_val.iloc[va_idx].reset_index(drop=True),
        X_test.iloc[te_idx].reset_index(drop=True),
        np.asarray(y_train)[tr_idx],
        np.asarray(y_val)[va_idx],
        np.asarray(y_test)[te_idx],
    )


def split_hidden_labels(X: pd.DataFrame, y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    strat = y if min(np.bincount(y)) >= 3 else None
    train_val_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=seed, stratify=strat)
    y_train_val = y[train_val_idx]
    strat2 = y_train_val if min(np.bincount(y_train_val)) >= 3 else None
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=seed + 17, stratify=strat2)
    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[val_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )


def make_feature_prompt(
    X_train: pd.DataFrame,
    dataset: str,
    description: str,
    n_clusters: int,
    history: List[Tuple[str, float]],
) -> List[Dict[str, str]]:
    sample = X_train.head(8).round(4).to_dict(orient="list")
    numeric_summary = X_train.describe().round(4).to_string()
    history_text = "\n".join(f"- previous validation ARI: {score:.4f}; note: {note}" for note, score in history[-5:])
    return [
        {
            "role": "system",
            "content": (
                "You write compact Python feature-engineering code for a clustering task. "
                "You only see input features and cluster count metadata. Do not assume access to labels, target values, test labels, or evaluation labels."
            ),
        },
        {
            "role": "user",
            "content": f"""
Dataset: {dataset}
Cluster count metadata: {n_clusters}
Dataset description:
{description[:1200]}

Feature columns:
{list(X_train.columns)}

Training feature sample:
{json.dumps(sample, ensure_ascii=False)}

Training numeric summary:
{numeric_summary[:4000]}

Past controller feedback, if any:
{history_text or "None"}

Return Python code only. Define exactly this function:

def generate_features(df):
    # df contains features only, with no target/label column
    ...
    return out_df

Constraints:
- Do not reference labels, target columns, y_true, y, test labels, or evaluation metrics inside the function.
- Do not import external packages beyond numpy and pandas.
- Keep the original rows unchanged in count and order.
- Produce only deterministic, vectorized, numeric features.
- Avoid loops, apply, file/network calls, and expensive row-wise operations.
- Keep total feature count below 4 times the original feature count.
```end
""",
        },
    ]


def make_model_prompt(
    X_train: pd.DataFrame,
    dataset: str,
    description: str,
    n_clusters: int,
    history: List[Tuple[str, float]],
    previous_code: str = "",
) -> List[Dict[str, str]]:
    sample = X_train.head(8).round(4).to_dict(orient="list")
    summary = X_train.describe().round(4).to_string()
    history_text = "\n".join(f"- validation ARI: {score:.4f}; note: {note}" for note, score in history[-8:])
    previous = f"\nPrevious code to refine:\n{previous_code[:6000]}\n" if previous_code else ""
    return [
        {
            "role": "system",
            "content": (
                "You write a scikit-learn compatible clustering estimator. "
                "The code must not access labels. A hidden controller will evaluate validation ARI outside your code."
            ),
        },
        {
            "role": "user",
            "content": f"""
Dataset: {dataset}
Cluster count metadata: {n_clusters}
Dataset description:
{description[:1200]}

Feature columns:
{list(X_train.columns)}

Training feature sample:
{json.dumps(sample, ensure_ascii=False)}

Training numeric summary:
{summary[:4000]}

Hidden-controller feedback:
{history_text or "None"}
{previous}
Return Python code only. Define a class named mycluster with methods:
- __init__(self, n_clusters={n_clusters}, random_state=0)
- fit(self, X, y=None)
- predict(self, X)
- fit_predict(self, X, y=None)

Allowed core model families: KMeans, MiniBatchKMeans, GaussianMixture, Birch, PCA, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, FeatureAgglomeration only as a transformer.
Use correct imports, for example GaussianMixture must come from sklearn.mixture, not sklearn.cluster.

Constraints:
- Do not reference labels, target columns, y_true, ground truth, ARI, NMI, or test labels inside the code.
- Do not use algorithms or code paths requiring all sample-by-sample distance/affinity matrices.
- Do not use SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, precomputed affinity, pairwise_distances, or kneighbors_graph.
- The estimator must support predict on held-out validation/test features after fit on training features.
- Keep runtime lightweight and deterministic.
```end
""",
        },
    ]


def validate_generated_code(code: str, target_column_name: str, kind: str) -> Optional[str]:
    lower = code.lower()
    for pat in BANNED_MODEL_PATTERNS:
        if pat in lower:
            return f"banned pattern: {pat}"

    def explicit_target_access(name: str) -> bool:
        if not name:
            return False
        escaped = re.escape(name)
        patterns = [
            rf"\[\s*['\"]{escaped}['\"]\s*\]",
            rf"\.\s*{escaped}\b",
            rf"drop\s*\([^)]*['\"]{escaped}['\"]",
            rf"pop\s*\(\s*['\"]{escaped}['\"]",
        ]
        return any(re.search(p, code, flags=re.IGNORECASE) for p in patterns)

    if explicit_target_access(target_column_name):
        return f"explicit target column access: {target_column_name}"
    if kind == "feature":
        for token in ["y_true", "target", "labels", "ground_truth", "species"]:
            if token in lower:
                return f"label-related token in feature code: {token}"
        if "def generate_features" not in code:
            return "missing generate_features"
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"syntax error: {exc}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                return "loop in feature code"
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                return "comprehension in feature code"
            if isinstance(node, ast.Call):
                func_text = ""
                if isinstance(node.func, ast.Attribute):
                    func_text = node.func.attr.lower()
                elif isinstance(node.func, ast.Name):
                    func_text = node.func.id.lower()
                if func_text in {"apply", "iterrows", "itertuples", "sleep", "open", "exec", "eval"}:
                    return f"unsafe feature call: {func_text}"
        unsafe_feature_tokens = ["subprocess", "multiprocessing", "requests", "socket", "sklearn"]
        for token in unsafe_feature_tokens:
            if token in lower:
                return f"unsafe feature code token: {token}"
    else:
        for token in ["y_true", "ground_truth", "adjusted_rand", "normalized_mutual_info", "species"]:
            if token in lower:
                return f"label/evaluation token in model code: {token}"
        if "class mycluster" not in lower:
            return "missing mycluster class"
    return None


def _feature_worker(code: str, train_json: str, val_json: str, test_json: str, result_path: str):
    try:
        ns = {"np": np, "pd": pd}
        exec(code, ns, ns)
        fn = ns.get("generate_features")
        if fn is None:
            raise ValueError("generate_features not defined")
        outs = []
        for js in [train_json, val_json, test_json]:
            df = pd.read_json(js, orient="split")
            out = fn(df.copy())
            if not isinstance(out, pd.DataFrame):
                raise ValueError("generate_features must return a DataFrame")
            if len(out) != len(df):
                raise ValueError("row count changed")
            out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            for c in out.columns:
                if not pd.api.types.is_numeric_dtype(out[c]):
                    out[c] = pd.Categorical(out[c].astype("string").fillna("__missing__")).codes.astype("float32")
            outs.append(out.astype("float32").to_json(orient="split"))
        write_worker_result(result_path, {"ok": True, "outs": outs})
    except Exception as exc:
        write_worker_result(result_path, {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(limit=3)})


def run_feature_code_with_timeout(
    code: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    timeout_s: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import multiprocessing as mp

    with tempfile.TemporaryDirectory(prefix="autologic_feature_") as tmpdir:
        result_path = str(Path(tmpdir) / "result.pkl")
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_feature_worker,
            args=(
                code,
                X_train.to_json(orient="split"),
                X_val.to_json(orient="split"),
                X_test.to_json(orient="split"),
                result_path,
            ),
        )
        proc.start()
        res = wait_for_worker_result(proc, result_path, timeout_s, "feature")
    if not res.get("ok"):
        raise RuntimeError(res.get("error", "feature worker failed"))
    return tuple(pd.read_json(js, orient="split") for js in res["outs"])


def _align_and_score_labels(y_ref: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    unique_labels = np.unique(labels)
    contingency = np.zeros((n_clusters, max(n_clusters, len(unique_labels))), dtype=np.int64)
    label_to_col = {lab: i for i, lab in enumerate(unique_labels)}
    for true_label, pred_label in zip(y_ref, labels):
        if int(true_label) < n_clusters:
            contingency[int(true_label), label_to_col[pred_label]] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if c < len(unique_labels):
            mapping[unique_labels[c]] = int(r)
    next_label = 0
    out = []
    for lab in labels:
        if lab not in mapping:
            while next_label in mapping.values():
                next_label += 1
            mapping[lab] = next_label
        out.append(mapping[lab])
    return np.asarray(out, dtype=int)


def _model_worker(
    code: str,
    class_name: str,
    X_train_json: str,
    X_val_json: str,
    X_test_json: str,
    y_train_json: str,
    y_val_json: str,
    y_test_json: str,
    n_clusters: int,
    seed: int,
    result_path: str,
):
    try:
        X_train = pd.read_json(X_train_json, orient="split").to_numpy(dtype=float)
        X_val = pd.read_json(X_val_json, orient="split").to_numpy(dtype=float)
        X_test = pd.read_json(X_test_json, orient="split").to_numpy(dtype=float)
        y_train = np.asarray(json.loads(y_train_json), dtype=int)
        y_val = np.asarray(json.loads(y_val_json), dtype=int)
        y_test = np.asarray(json.loads(y_test_json), dtype=int)
        ns: Dict[str, Any] = {
            "np": np,
            "pd": pd,
            "Pipeline": Pipeline,
            "StandardScaler": StandardScaler,
            "RobustScaler": RobustScaler,
            "PowerTransformer": PowerTransformer,
            "QuantileTransformer": QuantileTransformer,
            "PCA": PCA,
            "KMeans": KMeans,
            "MiniBatchKMeans": MiniBatchKMeans,
            "GaussianMixture": GaussianMixture,
            "Birch": Birch,
        }
        exec(code, ns, ns)
        cls = ns.get(class_name)
        if cls is None:
            raise ValueError(f"{class_name} not defined")
        try:
            model = cls(n_clusters=n_clusters, random_state=seed)
        except TypeError:
            try:
                model = cls(n_clusters=n_clusters)
            except TypeError:
                model = cls()
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise ValueError("model must implement fit and predict")
        model.fit(X_train)
        train_labels = np.asarray(model.predict(X_train), dtype=int)
        val_labels = np.asarray(model.predict(X_val), dtype=int)
        test_labels = np.asarray(model.predict(X_test), dtype=int)
        if len(np.unique(val_labels)) <= 1 or len(np.unique(test_labels)) <= 1:
            raise ValueError("single-cluster prediction")
        write_worker_result(
            result_path,
            {
                "ok": True,
                "train_labels": train_labels.tolist(),
                "val_labels": val_labels.tolist(),
                "test_labels": test_labels.tolist(),
                "val_ari": float(adjusted_rand_score(y_val, val_labels)),
                "val_nmi": float(normalized_mutual_info_score(y_val, val_labels)),
                "test_ari": float(adjusted_rand_score(y_test, test_labels)),
                "test_nmi": float(normalized_mutual_info_score(y_test, test_labels)),
            },
        )
    except Exception as exc:
        write_worker_result(result_path, {"ok": False, "error": repr(exc), "traceback": traceback.format_exc(limit=3)})


def run_model_code_with_timeout(
    code: str,
    class_name: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    n_clusters: int,
    seed: int,
    timeout_s: int,
) -> Dict[str, Any]:
    import multiprocessing as mp

    with tempfile.TemporaryDirectory(prefix="autologic_model_") as tmpdir:
        result_path = str(Path(tmpdir) / "result.pkl")
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_model_worker,
            args=(
                code,
                class_name,
                X_train.to_json(orient="split"),
                X_val.to_json(orient="split"),
                X_test.to_json(orient="split"),
                json.dumps(y_train.astype(int).tolist()),
                json.dumps(y_val.astype(int).tolist()),
                json.dumps(y_test.astype(int).tolist()),
                n_clusters,
                seed,
                result_path,
            ),
        )
        proc.start()
        res = wait_for_worker_result(proc, result_path, timeout_s, "model")
    if not res.get("ok"):
        raise RuntimeError(res.get("error", "model worker failed"))
    return res


def builtin_candidate_code(kind: str) -> str:
    if kind.startswith("kmeans"):
        scaler = "StandardScaler()"
        prefix_import = "from sklearn.preprocessing import StandardScaler"
        steps = '("scaler", StandardScaler()),\n            ("cluster", KMeans(n_clusters=n_clusters, n_init=80, random_state=random_state))'
        if kind == "kmeans_robust":
            prefix_import = "from sklearn.preprocessing import RobustScaler"
            steps = '("scaler", RobustScaler()),\n            ("cluster", KMeans(n_clusters=n_clusters, n_init=80, random_state=random_state))'
        elif kind == "kmeans_power":
            prefix_import = "from sklearn.preprocessing import PowerTransformer"
            steps = '("scaler", PowerTransformer(method="yeo-johnson", standardize=True)),\n            ("cluster", KMeans(n_clusters=n_clusters, n_init=80, random_state=random_state))'
        elif kind == "kmeans_quantile":
            prefix_import = "from sklearn.preprocessing import QuantileTransformer"
            steps = '("scaler", QuantileTransformer(output_distribution="normal", random_state=random_state)),\n            ("cluster", KMeans(n_clusters=n_clusters, n_init=80, random_state=random_state))'
        elif kind == "kmeans_pca":
            prefix_import = "from sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA"
            steps = '("scaler", StandardScaler()),\n            ("pca", PCA(n_components=0.95, random_state=random_state)),\n            ("cluster", KMeans(n_clusters=n_clusters, n_init=80, random_state=random_state))'
        return f"""
import numpy as np
from sklearn.pipeline import Pipeline
{prefix_import}
from sklearn.cluster import KMeans

class mycluster:
    def __init__(self, n_clusters=3, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = Pipeline([
            {steps}
        ])
    def fit(self, X, y=None):
        self.model.fit(X)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
"""
    if kind.startswith("gmm"):
        fast = kind.endswith("_fast")
        base_kind = kind[:-5] if fast else kind
        cov = base_kind.replace("gmm_", "") if "_" in base_kind else "full"
        if cov not in {"full", "diag", "tied", "spherical"}:
            cov = "full"
        scaler_import = "from sklearn.preprocessing import StandardScaler"
        scaler = "StandardScaler()"
        if base_kind.endswith("_robust"):
            cov = "full"
            scaler_import = "from sklearn.preprocessing import RobustScaler"
            scaler = "RobustScaler()"
        n_init = 2 if fast else 20
        max_iter = 100 if fast else 200
        return f"""
import numpy as np
{scaler_import}
from sklearn.mixture import GaussianMixture

class mycluster:
    def __init__(self, n_clusters=3, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = {scaler}
        self.model = GaussianMixture(n_components=n_clusters, covariance_type="{cov}", n_init={n_init}, max_iter={max_iter}, random_state=random_state, reg_covar=1e-6)
    def fit(self, X, y=None):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        return self
    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
"""
    if kind.startswith("birch"):
        threshold = "0.5"
        if kind == "birch_tight":
            threshold = "0.25"
        elif kind == "birch_loose":
            threshold = "0.8"
        elif kind == "birch_power":
            return """
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import Birch

class mycluster:
    def __init__(self, n_clusters=3, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = Pipeline([
            ("scaler", PowerTransformer(method="yeo-johnson", standardize=True)),
            ("cluster", Birch(n_clusters=n_clusters, threshold=0.5))
        ])
    def fit(self, X, y=None):
        self.model.fit(X)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
"""
        return f"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch

class mycluster:
    def __init__(self, n_clusters=3, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("cluster", Birch(n_clusters=n_clusters, threshold={threshold}))
        ])
    def fit(self, X, y=None):
        self.model.fit(X)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
"""
    if kind == "kmeans":
        return """
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class mycluster:
    def __init__(self, n_clusters=3, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("cluster", KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state))
        ])
    def fit(self, X, y=None):
        self.model.fit(X)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
"""
    if kind == "mbkmeans":
        return """
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

class mycluster:
    def __init__(self, n_clusters=3, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("cluster", MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=20, batch_size=2048))
        ])
    def fit(self, X, y=None):
        self.model.fit(X)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
"""
    if kind == "mbkmeans_fast":
        return """
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

class mycluster:
    def __init__(self, n_clusters=3, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("cluster", MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=5,
                max_iter=100,
                batch_size=4096,
                reassignment_ratio=0.01
            ))
        ])
    def fit(self, X, y=None):
        self.model.fit(X)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
"""
    if kind == "gmm":
        return """
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

class mycluster:
    def __init__(self, n_clusters=3, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = GaussianMixture(n_components=n_clusters, covariance_type="full", n_init=10, random_state=random_state)
    def fit(self, X, y=None):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        return self
    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
"""
    return """
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch

class mycluster:
    def __init__(self, n_clusters=3, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("cluster", Birch(n_clusters=n_clusters))
        ])
    def fit(self, X, y=None):
        self.model.fit(X)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)
"""


def candidate_from_code(
    name: str,
    source: str,
    code: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    n_clusters: int,
    seed: int,
    timeout_s: int,
) -> CandidateRecord:
    res = run_model_code_with_timeout(
        code,
        "mycluster",
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        n_clusters,
        seed,
        timeout_s,
    )
    return CandidateRecord(
        name=name,
        source=source,
        code=code,
        val_ari=float(res["val_ari"]),
        val_nmi=float(res["val_nmi"]),
        test_ari=float(res["test_ari"]),
        test_nmi=float(res["test_nmi"]),
        train_labels=res["train_labels"],
        val_labels=res["val_labels"],
        test_labels=res["test_labels"],
        raw_val_ari=float(res["val_ari"]),
    )


def candidate_from_code_for_selection(
    name: str,
    source: str,
    code: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    n_clusters: int,
    seed: int,
    timeout_s: int,
    args,
) -> CandidateRecord:
    sx_train, sx_val, sx_test, sy_train, sy_val, sy_test = selection_subsets(
        X_train, X_val, X_test, y_train, y_val, y_test, args, seed
    )
    return candidate_from_code(
        name,
        source,
        code,
        sx_train,
        sx_val,
        sx_test,
        sy_train,
        sy_val,
        sy_test,
        n_clusters,
        seed,
        timeout_s,
    )


def deterministic_projection_candidate(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    n_clusters: int,
) -> CandidateRecord:
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train.to_numpy(dtype=float))
    Xv = scaler.transform(X_val.to_numpy(dtype=float))
    Xs = scaler.transform(X_test.to_numpy(dtype=float))
    if Xt.shape[1] > 1:
        _, _, vt = np.linalg.svd(Xt[: min(len(Xt), 5000)], full_matrices=False)
        direction = vt[0]
    else:
        direction = np.ones(Xt.shape[1], dtype=float)

    train_score = Xt @ direction
    thresholds = np.quantile(train_score, np.linspace(0, 1, n_clusters + 1)[1:-1])

    def assign(arr: np.ndarray) -> np.ndarray:
        scores = arr @ direction
        labels = np.digitize(scores, thresholds, right=False).astype(int)
        if len(np.unique(labels)) <= 1 and len(labels) >= n_clusters:
            order = np.argsort(scores, kind="mergesort")
            labels = np.zeros(len(scores), dtype=int)
            bins = np.array_split(order, n_clusters)
            for lab, idx in enumerate(bins):
                labels[idx] = lab
        return labels

    train_labels = assign(Xt)
    val_labels = assign(Xv)
    test_labels = assign(Xs)
    return CandidateRecord(
        name="fallback_projection_quantile",
        source="last_resort_deterministic",
        code="StandardScaler + first principal direction + quantile bins",
        val_ari=float(adjusted_rand_score(y_val, val_labels)),
        val_nmi=float(normalized_mutual_info_score(y_val, val_labels)),
        test_ari=float(adjusted_rand_score(y_test, test_labels)),
        test_nmi=float(normalized_mutual_info_score(y_test, test_labels)),
        train_labels=train_labels.tolist(),
        val_labels=val_labels.tolist(),
        test_labels=test_labels.tolist(),
        raw_val_ari=float(adjusted_rand_score(y_val, val_labels)),
        selection_repeats=1,
    )


def materialize_large_candidates(
    candidates: List[CandidateRecord],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    n_clusters: int,
    seed: int,
    args,
    output_dir: Path,
) -> List[CandidateRecord]:
    if not getattr(args, "large_dataset_active", False):
        return candidates
    valid = [c for c in candidates if not c.failed and np.isfinite(c.val_ari)]
    valid.sort(key=lambda c: c.val_ari, reverse=True)
    if not valid:
        return []
    best_val = valid[0].val_ari
    eligible = [c for c in valid if c.val_ari >= args.min_val_ari and c.val_ari >= best_val - args.ensemble_val_gap]
    # Large-data selection is done on a fixed hidden-validation subsample. A model that
    # looks best on that subsample can still collapse to one cluster on the full split.
    # Keep trying ranked candidates instead of letting one collapsed model abort the seed.
    pool = eligible or valid
    max_attempts = max(int(getattr(args, "top_k", 3) or 3) * 4, 12)
    pool = pool[: min(len(pool), max_attempts)]
    materialized: List[CandidateRecord] = []
    for rec in pool:
        if len(materialized) >= max(1, int(getattr(args, "top_k", 3) or 3)):
            break
        try:
            full = candidate_from_code(
                rec.name,
                rec.source,
                rec.code,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                n_clusters,
                seed,
                args.model_code_timeout_s,
            )
            full_val_ari = full.val_ari
            full.raw_val_ari = rec.val_ari
            full.val_ari = rec.val_ari
            full.selection_repeats = rec.selection_repeats
            materialized.append(full)
            log_event(
                output_dir,
                {
                    "stage": "large_candidate_materialized",
                    "name": rec.name,
                    "selection_val_ari": rec.val_ari,
                    "full_val_ari": full_val_ari,
                    "test_ari": full.test_ari,
                },
            )
        except Exception as exc:
            log_event(output_dir, {"stage": "large_candidate_materialize_failed", "name": rec.name, "error": repr(exc)})
    if materialized:
        return sorted(materialized, key=lambda c: c.val_ari, reverse=True)
    fallback_kinds = ["mbkmeans_fast", "gmm_diag_fast", "gmm_spherical_fast", "birch_loose"]
    for kind in fallback_kinds:
        try:
            code = builtin_candidate_code(kind)
            full = candidate_from_code(
                f"fallback_{kind}",
                "fallback_builtin",
                code,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                n_clusters,
                seed,
                args.model_code_timeout_s,
            )
            full.raw_val_ari = full.val_ari
            full.selection_repeats = 1
            materialized.append(full)
            log_event(
                output_dir,
                {
                    "stage": "large_candidate_fallback_materialized",
                    "name": full.name,
                    "full_val_ari": full.val_ari,
                    "test_ari": full.test_ari,
                },
            )
            break
        except Exception as exc:
            log_event(output_dir, {"stage": "large_candidate_fallback_failed", "kind": kind, "error": repr(exc)})
    if materialized:
        return sorted(materialized, key=lambda c: c.val_ari, reverse=True)
    try:
        full = deterministic_projection_candidate(X_train, X_val, X_test, y_val, y_test, n_clusters)
        materialized.append(full)
        log_event(
            output_dir,
            {
                "stage": "large_candidate_last_resort_materialized",
                "name": full.name,
                "full_val_ari": full.val_ari,
                "test_ari": full.test_ari,
            },
        )
        return materialized
    except Exception as exc:
        log_event(
            output_dir,
            {
                "stage": "large_candidate_materialize_empty",
                "attempted": len(pool),
                "fallback_kinds": fallback_kinds,
                "last_resort_error": repr(exc),
            },
        )
    return []


def apply_repeated_selection_score(
    rec: CandidateRecord,
    code: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    n_clusters: int,
    seed: int,
    args,
) -> CandidateRecord:
    repeats = effective_selection_repeats(args)
    if repeats <= 1:
        rec.raw_val_ari = rec.val_ari
        rec.selection_repeats = 1
        return rec
    X_dev = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_dev = np.concatenate([np.asarray(y_train), np.asarray(y_val)])
    vals = []
    for rep in range(repeats):
        idx = np.arange(len(y_dev))
        strat = y_dev if min(np.bincount(y_dev.astype(int))) >= 3 else None
        tr_idx, va_idx = train_test_split(
            idx,
            test_size=float(getattr(args, "selection_val_fraction", 0.35)),
            random_state=seed + 7919 * (rep + 1),
            stratify=strat,
        )
        try:
            tr_keep = _stratified_take(y_dev[tr_idx], getattr(args, "large_selection_train_size", len(tr_idx)), seed + 1234 + rep)
            va_keep = _stratified_take(y_dev[va_idx], getattr(args, "large_selection_val_size", len(va_idx)), seed + 2345 + rep)
            tr_idx_use = np.asarray(tr_idx)[tr_keep]
            va_idx_use = np.asarray(va_idx)[va_keep]
            tmp = candidate_from_code(
                f"{rec.name}_repeat{rep + 1}",
                rec.source,
                code,
                X_dev.iloc[tr_idx_use].reset_index(drop=True),
                X_dev.iloc[va_idx_use].reset_index(drop=True),
                X_dev.iloc[va_idx_use].reset_index(drop=True),
                y_dev[tr_idx_use],
                y_dev[va_idx_use],
                y_dev[va_idx_use],
                n_clusters,
                seed + 7919 * (rep + 1),
                args.model_code_timeout_s,
            )
            vals.append(float(tmp.val_ari))
        except Exception:
            continue
    if vals:
        rec.raw_val_ari = rec.val_ari
        rec.val_ari = float(np.mean(vals))
        rec.selection_repeats = len(vals)
    return rec


def select_features(
    args,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    dataset: str,
    description: str,
    n_clusters: int,
    target_column_name: str,
    output_dir: Path,
):
    best = (X_train, X_val, X_test)
    best_score = -1.0
    history: List[Tuple[str, float]] = []
    feature_eval_kind = "mbkmeans_fast" if getattr(args, "large_dataset_active", False) else "kmeans"
    try:
        base = candidate_from_code_for_selection(
            "feature_base_kmeans",
            "feature_baseline",
            builtin_candidate_code(feature_eval_kind),
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            n_clusters,
            args.seed,
            args.model_code_timeout_s,
            args,
        )
        best_score = base.val_ari
        history.append(("raw features baseline", best_score))
    except Exception as exc:
        history.append((f"raw features baseline failed: {str(exc)[:120]}", best_score))
        log_event(output_dir, {"stage": "feature_baseline_failed", "error": repr(exc)})
    if args.feat_iterations <= 0:
        return best, {"best_feature_val_ari": best_score, "valid_feature_candidates": 0}

    valid_count = 0
    consecutive_feature_failures = 0
    for i in range(args.feat_iterations):
        if getattr(args, "max_feature_failures", 2) >= 0 and consecutive_feature_failures >= args.max_feature_failures:
            log_event(
                output_dir,
                {
                    "stage": "feature_skipped",
                    "reason": "max_consecutive_feature_failures",
                    "max_feature_failures": args.max_feature_failures,
                },
            )
            break
        code = ""
        try:
            code = call_llm(
                make_feature_prompt(X_train, dataset, description, n_clusters, history),
                model=args.llm,
                base_url=args.api_base,
                api_key=args.api_key,
                output_dir=output_dir,
                stage="feature_generation",
                timeout_s=args.api_timeout_s,
                max_tokens=args.max_tokens,
            )
            reason = validate_generated_code(code, target_column_name, "feature")
            if reason:
                raise ValueError(reason)
            tr, va, te = run_feature_code_with_timeout(code, X_train, X_val, X_test, args.feature_code_timeout_s)
            if tr.shape[1] > max(1, X_train.shape[1] * 4):
                raise ValueError("too many generated features")
            rec = candidate_from_code_for_selection(
                f"feature_candidate_{i + 1}",
                "feature_llm_eval",
                builtin_candidate_code(feature_eval_kind),
                tr,
                va,
                te,
                y_train,
                y_val,
                y_test,
                n_clusters,
                args.seed,
                args.model_code_timeout_s,
                args,
            )
            valid_count += 1
            consecutive_feature_failures = 0
            save_generated_code(output_dir, stage="feature", name=f"feature_candidate_{i + 1}", code=code, accepted=True)
            log_event(
                output_dir,
                {
                    "stage": "feature_provenance",
                    "candidate": f"feature_candidate_{i + 1}",
                    "records": build_feature_provenance(
                        X_train.columns,
                        tr.columns,
                        target_column_name,
                        generated_code=code,
                        round_num=i + 1,
                    ),
                },
            )
            history.append((f"feature_candidate_{i + 1}", rec.val_ari))
            log_event(output_dir, {"stage": "feature_eval", "candidate": i + 1, "val_ari": rec.val_ari})
            if rec.val_ari > best_score:
                best = (tr, va, te)
                best_score = rec.val_ari
        except Exception as exc:
            consecutive_feature_failures += 1
            if code:
                save_generated_code(output_dir, stage="feature", name=f"feature_candidate_{i + 1}", code=code, accepted=False, reason=repr(exc))
            history.append((f"feature_candidate_{i + 1} failed: {str(exc)[:120]}", best_score))
            log_event(output_dir, {"stage": "feature_failed", "candidate": i + 1, "error": repr(exc)})
    return best, {"best_feature_val_ari": best_score, "valid_feature_candidates": valid_count}


def generate_teacher_candidates(
    args,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    dataset: str,
    description: str,
    n_clusters: int,
    target_column_name: str,
    output_dir: Path,
) -> List[CandidateRecord]:
    candidates: List[CandidateRecord] = []
    if getattr(args, "large_dataset_active", False):
        builtin_kinds = [
            "mbkmeans_fast",
            "gmm_diag_fast",
            "gmm_tied_fast",
            "gmm_spherical_fast",
            "birch",
            "birch_tight",
            "birch_loose",
        ]
        log_event(
            output_dir,
            {
                "stage": "large_dataset_fast_candidates",
                "builtin_kinds": builtin_kinds,
                "selection_repeats": effective_selection_repeats(args),
                "selection_train_size": args.large_selection_train_size,
                "selection_val_size": args.large_selection_val_size,
            },
        )
    else:
        builtin_kinds = [
            "kmeans",
            "kmeans_robust",
            "kmeans_power",
            "kmeans_quantile",
            "kmeans_pca",
            "mbkmeans",
            "gmm_full",
            "gmm_diag",
            "gmm_tied",
            "gmm_spherical",
            "gmm_robust",
            "birch",
            "birch_tight",
            "birch_loose",
            "birch_power",
        ]
    for kind in builtin_kinds:
        try:
            code = builtin_candidate_code(kind)
            rec = candidate_from_code_for_selection(
                f"builtin_{kind}",
                "builtin",
                code,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                n_clusters,
                args.seed,
                args.model_code_timeout_s,
                args,
            )
            rec = apply_repeated_selection_score(
                rec,
                code,
                X_train,
                X_val,
                y_train,
                y_val,
                n_clusters,
                args.seed,
                args,
            )
            candidates.append(rec)
            log_event(output_dir, {"stage": "teacher_eval", "name": rec.name, "val_ari": rec.val_ari, "test_ari": rec.test_ari})
        except Exception as exc:
            log_event(output_dir, {"stage": "teacher_builtin_failed", "kind": kind, "error": repr(exc)})

    history: List[Tuple[str, float]] = [(c.name, c.val_ari) for c in candidates]
    for i in range(args.model_iterations):
        code = ""
        try:
            code = call_llm(
                make_model_prompt(X_train, dataset, description, n_clusters, history),
                model=args.llm,
                base_url=args.api_base,
                api_key=args.api_key,
                output_dir=output_dir,
                stage="model_generation",
                timeout_s=args.api_timeout_s,
                max_tokens=args.max_tokens,
            )
            reason = validate_generated_code(code, target_column_name, "model")
            if reason:
                raise ValueError(reason)
            best_code = code
            best_rec = candidate_from_code_for_selection(
                f"llm_model_{i + 1}",
                "llm_model",
                best_code,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                n_clusters,
                args.seed + i,
                args.model_code_timeout_s,
                args,
            )
            best_rec = apply_repeated_selection_score(
                best_rec,
                best_code,
                X_train,
                X_val,
                y_train,
                y_val,
                n_clusters,
                args.seed + i,
                args,
            )
            save_generated_code(output_dir, stage="model", name=f"llm_model_{i + 1}", code=code, accepted=True)
            history.append((best_rec.name, best_rec.val_ari))
            for j in range(args.param_iterations):
                param_code = ""
                try:
                    param_code = call_llm(
                        make_model_prompt(X_train, dataset, description, n_clusters, history, previous_code=best_code),
                        model=args.llm,
                        base_url=args.api_base,
                        api_key=args.api_key,
                        output_dir=output_dir,
                        stage="param_refinement",
                        timeout_s=args.api_timeout_s,
                        max_tokens=args.max_tokens,
                    )
                    reason = validate_generated_code(param_code, target_column_name, "model")
                    if reason:
                        raise ValueError(reason)
                    param_rec = candidate_from_code_for_selection(
                        f"llm_model_{i + 1}_param_{j + 1}",
                        "llm_param",
                        param_code,
                        X_train,
                        X_val,
                        X_test,
                        y_train,
                        y_val,
                        y_test,
                        n_clusters,
                        args.seed + i * 100 + j,
                        args.model_code_timeout_s,
                        args,
                    )
                    param_rec = apply_repeated_selection_score(
                        param_rec,
                        param_code,
                        X_train,
                        X_val,
                        y_train,
                        y_val,
                        n_clusters,
                        args.seed + i * 100 + j,
                        args,
                    )
                    save_generated_code(output_dir, stage="param", name=f"llm_model_{i + 1}_param_{j + 1}", code=param_code, accepted=True)
                    history.append((param_rec.name, param_rec.val_ari))
                    if param_rec.val_ari > best_rec.val_ari:
                        best_rec = param_rec
                        best_code = param_code
                except Exception as exc:
                    if param_code:
                        save_generated_code(output_dir, stage="param", name=f"llm_model_{i + 1}_param_{j + 1}", code=param_code, accepted=False, reason=repr(exc))
                    history.append((f"param_failed_{i + 1}_{j + 1}: {str(exc)[:120]}", max([h[1] for h in history] or [0.0])))
                    log_event(output_dir, {"stage": "param_failed", "model_iter": i + 1, "param_iter": j + 1, "error": repr(exc)})
            candidates.append(best_rec)
            log_event(output_dir, {"stage": "teacher_eval", "name": best_rec.name, "val_ari": best_rec.val_ari, "test_ari": best_rec.test_ari})
        except Exception as exc:
            if code:
                save_generated_code(output_dir, stage="model", name=f"llm_model_{i + 1}", code=code, accepted=False, reason=repr(exc))
            history.append((f"model_failed_{i + 1}: {str(exc)[:120]}", max([h[1] for h in history] or [0.0])))
            log_event(output_dir, {"stage": "model_failed", "model_iter": i + 1, "error": repr(exc)})
    candidates = sorted(candidates, key=lambda c: c.val_ari, reverse=True)
    return materialize_large_candidates(
        candidates,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        n_clusters,
        args.seed,
        args,
        output_dir,
    )


def learn_alignment(anchor: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    anchor = np.asarray(anchor).astype(int)
    labels = np.asarray(labels).astype(int)
    anchor_vals = np.unique(anchor)
    label_vals = np.unique(labels)
    mat = np.zeros((len(anchor_vals), len(label_vals)), dtype=np.int64)
    a_map = {v: i for i, v in enumerate(anchor_vals)}
    l_map = {v: i for i, v in enumerate(label_vals)}
    for a, l in zip(anchor, labels):
        mat[a_map[a], l_map[l]] += 1
    row_ind, col_ind = linear_sum_assignment(-mat)
    mapping = {label_vals[c]: anchor_vals[r] for r, c in zip(row_ind, col_ind)}
    return {int(k): int(v) for k, v in mapping.items()}


def apply_alignment(labels: np.ndarray, mapping: Dict[int, int], anchor: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    anchor = np.asarray(anchor).astype(int)
    local = dict(mapping)
    used = set(local.values())
    anchor_vals = np.unique(anchor)
    next_label = int(anchor_vals.max() + 1) if len(anchor_vals) else 0
    out = []
    for x in labels:
        x = int(x)
        if x not in local:
            while next_label in used:
                next_label += 1
            local[x] = next_label
            used.add(next_label)
            next_label += 1
        out.append(local[x])
    return np.asarray(out, dtype=int)


def align_labels_to_anchor(anchor: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return apply_alignment(labels, learn_alignment(anchor, labels), anchor)


def vote_labels(selected: List[CandidateRecord], split: str, weighted: bool) -> np.ndarray:
    if not selected:
        raise ValueError("no selected candidates")
    if len(selected) == 1:
        return np.asarray(getattr(selected[0], f"{split}_labels"), dtype=int)
    anchor_val = np.asarray(selected[0].val_labels, dtype=int)
    aligned = []
    weights = []
    for rec in selected:
        mapping = learn_alignment(anchor_val, np.asarray(rec.val_labels, dtype=int))
        val_aligned = apply_alignment(np.asarray(rec.val_labels, dtype=int), mapping, anchor_val)
        split_labels = np.asarray(getattr(rec, f"{split}_labels"), dtype=int)
        if split == "val":
            aligned_split = val_aligned
        else:
            aligned_split = apply_alignment(split_labels, mapping, anchor_val)
        aligned.append(aligned_split)
        weights.append(max(float(rec.val_ari), 1e-6) if weighted else 1.0)
    arr = np.vstack(aligned)
    labels_out = []
    for col in arr.T:
        scores: Dict[int, float] = {}
        for lab, w in zip(col, weights):
            scores[int(lab)] = scores.get(int(lab), 0.0) + float(w)
        labels_out.append(max(scores.items(), key=lambda kv: (kv[1], -kv[0]))[0])
    return np.asarray(labels_out, dtype=int)


def student_from_teacher(
    rec: CandidateRecord,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> CandidateRecord:
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)
    Xs = scaler.transform(X_test)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=350,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=20,
    )
    pseudo = np.asarray(rec.train_labels, dtype=int)
    clf.fit(Xt, pseudo)
    train_labels = clf.predict(Xt).astype(int)
    val_labels = clf.predict(Xv).astype(int)
    test_labels = clf.predict(Xs).astype(int)
    return CandidateRecord(
        name=f"student_{rec.name}",
        source="student_distilled",
        code="MLPClassifier distillation from cached teacher labels",
        val_ari=float(adjusted_rand_score(y_val, val_labels)),
        val_nmi=float(normalized_mutual_info_score(y_val, val_labels)),
        test_ari=float(adjusted_rand_score(y_test, test_labels)),
        test_nmi=float(normalized_mutual_info_score(y_test, test_labels)),
        train_labels=train_labels.tolist(),
        val_labels=val_labels.tolist(),
        test_labels=test_labels.tolist(),
    )


def internal_metrics(X: pd.DataFrame, labels: np.ndarray, max_samples: int = 0, seed: int = 0) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) <= 1 or len(np.unique(labels)) >= len(labels):
        return {"silhouette": math.nan, "db": math.nan, "ch": math.nan}
    if max_samples and max_samples > 0 and len(labels) > max_samples:
        idx = _stratified_take(labels, int(max_samples), seed)
        labels = labels[idx]
        X = X.iloc[idx].reset_index(drop=True)
        if len(np.unique(labels)) <= 1 or len(np.unique(labels)) >= len(labels):
            return {"silhouette": math.nan, "db": math.nan, "ch": math.nan}
    arr = X.to_numpy(dtype=float)
    try:
        sil = float(silhouette_score(arr, labels))
    except Exception:
        sil = math.nan
    try:
        db = float(davies_bouldin_score(arr, labels))
    except Exception:
        db = math.nan
    try:
        ch = float(calinski_harabasz_score(arr, labels))
    except Exception:
        ch = math.nan
    return {"silhouette": sil, "db": db, "ch": ch}


def evaluate_configs(
    teachers: List[CandidateRecord],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args,
    n_clusters: int,
) -> List[Dict[str, Any]]:
    valid_teachers = [c for c in teachers if not c.failed and np.isfinite(c.val_ari)]
    valid_teachers.sort(key=lambda c: c.val_ari, reverse=True)
    best_teacher_val = valid_teachers[0].val_ari if valid_teachers else -1.0
    eligible = [
        c
        for c in valid_teachers
        if c.val_ari >= args.min_val_ari and c.val_ari >= best_teacher_val - args.ensemble_val_gap
    ]
    if not eligible:
        eligible = valid_teachers[:1]
    selected = eligible[: max(1, min(args.top_k, len(eligible)))]

    students: List[CandidateRecord] = []
    for i, rec in enumerate(selected):
        try:
            students.append(student_from_teacher(rec, X_train, X_val, X_test, y_val, y_test, args.seed + 1000 + i))
        except Exception:
            pass
    if not students:
        students = selected
    else:
        students.sort(key=lambda c: c.val_ari, reverse=True)
        best_student_val = students[0].val_ari if students else -1.0
        eligible_students = [
            c
            for c in students
            if c.val_ari >= args.min_val_ari and c.val_ari >= best_student_val - args.ensemble_val_gap
        ]
        students = (eligible_students or students[:1])[: max(1, min(args.top_k, len(eligible_students or students)))]

    def choose_auto(recs: List[CandidateRecord]):
        plans = [("best_single", recs[:1], False)]
        if len(recs) > 1:
            plans.extend([("majority", recs, False), ("weighted", recs, True)])
        scored = []
        for mode, mode_recs, weighted in plans:
            val_labels = vote_labels(mode_recs, "val", weighted=weighted)
            score = float(adjusted_rand_score(y_val, val_labels))
            scored.append((score, -len(mode_recs), 1 if mode == "best_single" else 0, mode, mode_recs, weighted))
        scored.sort(reverse=True)
        _, _, _, mode, mode_recs, weighted = scored[0]
        return mode, mode_recs, weighted

    teacher_auto_mode, teacher_auto_recs, teacher_auto_weighted = choose_auto(selected)
    student_auto_mode, student_auto_recs, student_auto_weighted = choose_auto(students)

    configs = [
        ("teacher_no_distill_no_consensus", selected[:1], "test", False, "best_single"),
        ("teacher_no_distill_majority_consensus", selected, "test", False, "majority"),
        ("teacher_no_distill_weighted_consensus", selected, "test", True, "weighted"),
        ("teacher_no_distill_auto_consensus", teacher_auto_recs, "test", teacher_auto_weighted, teacher_auto_mode),
        ("student_distilled_no_consensus", students[:1], "test", False, "best_single"),
        ("student_distilled_majority_consensus", students, "test", False, "majority"),
        ("student_distilled_weighted_consensus", students, "test", True, "weighted"),
        ("student_distilled_auto_consensus", student_auto_recs, "test", student_auto_weighted, student_auto_mode),
    ]
    rows: List[Dict[str, Any]] = []
    for config, recs, split, weighted, selection_mode in configs:
        test_labels = vote_labels(recs, "test", weighted=weighted)
        val_labels = vote_labels(recs, "val", weighted=weighted)
        vals = internal_metrics(
            X_test,
            test_labels,
            max_samples=(args.internal_metric_max_samples if getattr(args, "large_dataset_active", False) else 0),
            seed=args.seed,
        )
        rows.append(
            {
                "config": config,
                "dataset": args.dataset,
                "seed": args.seed,
                "llm": args.llm,
                "feat_iterations": args.feat_iterations,
                "model_iterations": args.model_iterations,
                "param_iterations": args.param_iterations,
                "top_k": args.top_k,
                "min_val_ari": args.min_val_ari,
                "ensemble_val_gap": args.ensemble_val_gap,
                "stage_timeout_s": args.stage_timeout_s,
                "selection_mode": selection_mode,
                "internal_metric_max_samples": int(args.internal_metric_max_samples if getattr(args, "large_dataset_active", False) else 0),
                "selected_k": int(len(np.unique(test_labels))),
                "requested_k": int(n_clusters),
                "val_ari": float(adjusted_rand_score(y_val, val_labels)) * 100.0,
                "val_nmi": float(normalized_mutual_info_score(y_val, val_labels)) * 100.0,
                "ari": float(adjusted_rand_score(y_test, test_labels)) * 100.0,
                "nmi": float(normalized_mutual_info_score(y_test, test_labels)) * 100.0,
                "silhouette": vals["silhouette"],
                "db": vals["db"],
                "ch": vals["ch"],
                "failed_flag": 0,
                "base_model_count": len(valid_teachers),
                "eligible_model_count": len(eligible),
                "used_model_count": len(recs),
                "selected_model_names": json.dumps([r.name for r in recs], ensure_ascii=False),
                "selected_model_val_ari": json.dumps([round(float(r.val_ari) * 100.0, 4) for r in recs]),
                "selected_model_raw_val_ari": json.dumps([round(float(r.raw_val_ari if r.raw_val_ari is not None else r.val_ari) * 100.0, 4) for r in recs]),
                "selected_model_repeats": json.dumps([int(getattr(r, "selection_repeats", 1)) for r in recs]),
            }
        )
    return rows


def run_one(args) -> Dict[str, Any]:
    task_spec, plan_bundle = materialize_task_protocol(
        "clustering",
        args.dataset,
        args,
        task_spec_path=args.task_spec,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir).resolve() / f"{args.dataset}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = AgenticReasoningAgent(task_spec, plan_bundle)
    write_task_protocol(output_dir / "task_protocol.json", task_spec, plan_bundle, agent=agent)

    X, y, target_name, n_clusters, description = load_dataset(args.dataset)
    args.large_dataset_active = is_large_dataset_mode(args, args.dataset, len(X))
    X_train, X_val, X_test, y_train, y_val, y_test = split_hidden_labels(X, y, args.seed)
    log_event(
        output_dir,
        {
            "stage": "start",
            "dataset": args.dataset,
            "seed": args.seed,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_clusters": n_clusters,
            "target_hidden": True,
            "large_dataset_active": bool(args.large_dataset_active),
            "effective_selection_repeats": effective_selection_repeats(args),
            "task_spec": task_spec.to_dict(),
            "plans": plan_bundle.to_dict(),
            "agent": agent.describe(),
        },
    )
    (X_train, X_val, X_test), feature_info = select_features(
        args,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        args.dataset,
        description,
        n_clusters,
        target_name,
        output_dir,
    )
    agent.observe(
        {
            "stage": "feature_engineering",
            "round": 1,
            "status": "completed",
            "events": {},
        }
    )
    teachers = generate_teacher_candidates(
        args,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        args.dataset,
        description,
        n_clusters,
        target_name,
        output_dir,
    )
    agent.observe(
        {
            "stage": "model_generation",
            "round": 1,
            "status": "completed",
            "val_ari": max((candidate.val_ari for candidate in teachers), default=None),
            "events": {},
        }
    )
    rows = evaluate_configs(teachers, X_train, X_val, X_test, y_val, y_test, args, n_clusters)
    agent.observe(
        {
            "stage": "output_control",
            "round": 1,
            "status": "completed",
            "events": {},
        }
    )
    for row in rows:
        row.update(feature_info)
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "target_column_hidden": target_name,
        "n_clusters": n_clusters,
        "split": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "large_dataset_active": bool(getattr(args, "large_dataset_active", False)),
        "effective_selection_repeats": effective_selection_repeats(args),
        "task_spec": task_spec.to_dict(),
        "plans": plan_bundle.to_dict(),
        "agent": agent.describe(),
        "agent_transitions": agent.transitions(),
        "configs": rows,
        "teacher_candidates": [
            {
                "name": c.name,
                "source": c.source,
                "val_ari": c.val_ari * 100.0,
                "val_nmi": c.val_nmi * 100.0,
                "test_ari": c.test_ari * 100.0,
                "test_nmi": c.test_nmi * 100.0,
                "raw_val_ari": (c.raw_val_ari if c.raw_val_ari is not None else c.val_ari) * 100.0,
                "selection_repeats": c.selection_repeats,
            }
            for c in teachers
        ],
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (output_dir / "per_config.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": args.dataset, "seed": args.seed, "configs": rows}, ensure_ascii=False), flush=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--llm", default="gpt-3.5-turbo")
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "https://api.laozhang.ai/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--feat-iterations", type=int, default=10)
    parser.add_argument("--model-iterations", type=int, default=7)
    parser.add_argument("--param-iterations", type=int, default=5)
    parser.add_argument("--stage-timeout-s", type=int, default=900)
    parser.add_argument("--api-timeout-s", type=int, default=180)
    parser.add_argument("--feature-code-timeout-s", type=int, default=45)
    parser.add_argument("--model-code-timeout-s", type=int, default=120)
    parser.add_argument("--max-feature-failures", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-val-ari", type=float, default=0.30)
    parser.add_argument("--ensemble-val-gap", type=float, default=0.03)
    parser.add_argument("--selection-repeats", type=int, default=3)
    parser.add_argument("--selection-val-fraction", type=float, default=0.35)
    parser.add_argument("--large-dataset-strategy", choices=["auto", "off"], default="auto")
    parser.add_argument("--large-datasets", default="cd2,ld2,cc3")
    parser.add_argument("--large-sample-threshold", type=int, default=12000)
    parser.add_argument("--large-selection-train-size", type=int, default=3000)
    parser.add_argument("--large-selection-val-size", type=int, default=1500)
    parser.add_argument("--large-selection-test-size", type=int, default=1500)
    parser.add_argument("--large-selection-repeats", type=int, default=1)
    parser.add_argument("--internal-metric-max-samples", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--task-spec", default=None, help="Optional JSON task specification")
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "results_cluster_hidden_evaluator_compare_smoke_20260530" / "raw_runs"))
    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("OPENAI_API_KEY or --api-key is required")
    return args


if __name__ == "__main__":
    run_one(parse_args())
