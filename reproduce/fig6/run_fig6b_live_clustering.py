"""Live-LLM rerun of Fig. 6 panel b clustering using autoLOGIC_new logic.

The script executes live LLM feature generation and live LLM clustering-model
generation, then records per-seed source data. It follows the current
``autoLOGIC_new/ensemble/cluster_ensemble/cluster_auto_ensemble.py`` logic:

* generated features are selected by ARI under KMeans against the known labels;
* generated clustering models are scored by ARI against the known labels;
* per-config top-k models are selected by ARI;
* the selected models are combined by Consensus Matrix + SpectralClustering;
* distilled student models use an MLP with gradient clipping and no-clipping
  variants.

This script intentionally records that labels are used in model/feature
selection because that is the actual autoLOGIC_new comparison-experiment logic.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import pickle
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from openai import OpenAI
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEFAULT_DATASETS = ["breast", "glass", "students", "iris", "seeds"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
EPS = 1e-12
PROMPT_SAMPLE_ROWS = 5
PROMPT_MAX_COLUMNS = 40
PROMPT_DESCRIPTION_CHARS = 4000
PROMPT_CODE_CHARS = 12000
PROMPT_ERROR_CHARS = 2000


def truncate_text(text: Any, max_chars: int) -> str:
    value = "" if text is None else str(text)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n...[truncated {len(value) - max_chars} chars]"


def prompt_column_subset(df: pd.DataFrame, max_cols: int = PROMPT_MAX_COLUMNS) -> Tuple[List[str], List[str]]:
    cols = [str(c) for c in df.columns]
    if len(cols) <= max_cols:
        return cols, []
    return cols[:max_cols], cols[max_cols:]


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def load_dataset(repo_root: Path, dataset: str) -> Tuple[pd.DataFrame, str, str]:
    data_root = Path(os.environ.get("AUTOLOGIC_DATA_DIR", repo_root / "data" / "pkl"))
    path = data_root / f"{dataset}.pkl"
    with path.open("rb") as fh:
        ds = pickle.load(fh)
    target = ds[4][-1]
    df = ds[1].copy()
    description = ds[-1]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, target, description


def numeric_frame(x: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=x.index)
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            out[col] = pd.to_numeric(x[col], errors="coerce")
        else:
            out[col] = pd.factorize(x[col].astype(str), sort=True)[0]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_prompt_samples(df: pd.DataFrame, sample_size: int = PROMPT_SAMPLE_ROWS, max_cols: int = PROMPT_MAX_COLUMNS) -> str:
    samples = ""
    shown_cols, omitted_cols = prompt_column_subset(df, max_cols=max_cols)
    df_head = df.loc[:, shown_cols].head(sample_size)
    for col in shown_cols:
        values = df_head[col].tolist()
        if str(df[col].dtype).startswith("float"):
            values = [round(float(v), 3) if pd.notna(v) else v for v in values]
        values = [truncate_text(v, 80) if isinstance(v, str) else v for v in values]
        samples += f"{col} ({df[col].dtype}): Samples {values}\n"
    if omitted_cols:
        samples += f"... {len(omitted_cols)} additional columns omitted from prompt: {omitted_cols[:20]}\n"
    return samples


def summarize_dataset(df: pd.DataFrame, sample_size: int = PROMPT_SAMPLE_ROWS, max_cols: int = PROMPT_MAX_COLUMNS) -> Dict[str, Any]:
    shown_cols, omitted_cols = prompt_column_subset(df, max_cols=max_cols)
    summary = {
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1]),
        "columns_shown": shown_cols,
        "n_columns_omitted_from_prompt": len(omitted_cols),
        "columns_omitted_preview": omitted_cols[:20],
        "feature_types": {},
        "feature_ranges": {},
        "mean_std": {},
        "sample_head": df.loc[:, shown_cols].head(sample_size).to_dict(orient="records"),
    }
    for col in shown_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary["feature_types"][col] = "numerical"
            summary["feature_ranges"][col] = [to_jsonable(df[col].min()), to_jsonable(df[col].max())]
            summary["mean_std"][col] = [to_jsonable(df[col].mean()), to_jsonable(df[col].std())]
        else:
            summary["feature_types"][col] = "categorical"
            summary["feature_ranges"][col] = [str(v) for v in list(df[col].unique())[:10]]
    return summary


def clean_code_block(text: str) -> str:
    text = text.replace("<end>", "").strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    text = re.sub(r"^```python\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def clean_cluster_code(code: str) -> str:
    code = clean_code_block(code)
    lines = code.splitlines()
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("class mycluster") or stripped.startswith("import ") or stripped.startswith("from "):
            cleaned.append(line)
        elif cleaned:
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def call_llm(
    model: str,
    messages: List[Dict[str, str]],
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=90.0)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                stop=["```end"],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            usage = completion.usage
            return completion.choices[0].message.content or "", {
                "prompt_tokens": getattr(usage, "prompt_tokens", np.nan),
                "completion_tokens": getattr(usage, "completion_tokens", np.nan),
                "total_tokens": getattr(usage, "total_tokens", np.nan),
                "attempt": attempt,
            }
        except TypeError:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                stop=["```end"],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = completion.usage
            return completion.choices[0].message.content or "", {
                "prompt_tokens": getattr(usage, "prompt_tokens", np.nan),
                "completion_tokens": getattr(usage, "completion_tokens", np.nan),
                "total_tokens": getattr(usage, "total_tokens", np.nan),
                "attempt": attempt,
            }
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {type(last_error).__name__}: {last_error}")


def validate_feature_code(code: str) -> None:
    tree = ast.parse(code)
    forbidden_names = {"open", "eval", "exec", "__import__", "compile", "input"}
    forbidden_modules = {"os", "sys", "subprocess", "socket", "pathlib", "shutil", "requests", "urllib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
            raise ValueError(f"Forbidden call in feature code: {node.func.id}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base in forbidden_modules:
                    raise ValueError(f"Forbidden import in feature code: {alias.name}")


def execute_feature_code(code: str, df: pd.DataFrame, target: str) -> pd.DataFrame:
    code = clean_code_block(code)
    validate_feature_code(code)
    out = copy.deepcopy(df)
    scope = {"df": out, "pd": pd, "np": np, "KMeans": KMeans, "StandardScaler": StandardScaler}
    exec(compile(ast.parse(code), filename="<feature_code>", mode="exec"), scope, {})
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    if target not in out.columns:
        out[target] = df[target].values
    # Keep only numeric generated/feature columns plus target.
    for col in list(out.columns):
        if col != target and not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col].astype(str), sort=True)[0]
    return out


def build_feature_prompt(df: pd.DataFrame, target: str, description: str, dataset: str) -> str:
    samples = build_prompt_samples(df)
    description = truncate_text(description, PROMPT_DESCRIPTION_CHARS)
    return f"""
The dataframe `df` is loaded and in memory.

Dataset name: {dataset}
Dataset description:
{description}

Columns and samples:
{samples}

The downstream task is clustering. Generate pandas code that adds useful
numerical features to `df` for discovering the latent clusters associated with
the target column `{target}`. The generated features may use existing feature
columns but must not use `{target}` itself.

Formatting rules:
- Output only one Python code block.
- The code must modify `df` by adding numerical columns.
- Do not print anything.
- Do not read or write files.
- Do not use network calls.
- Keep the code self-contained and compatible with pandas/numpy/sklearn.
```python
"""


def live_generate_features(
    df: pd.DataFrame,
    target: str,
    y_true: np.ndarray,
    true_k: int,
    dataset: str,
    description: str,
    seed: int,
    llm_model: str,
    base_url: str,
    api_key: str,
    feature_iterations: int,
) -> Tuple[pd.DataFrame, str, List[Dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    current = df.copy()
    records: List[Dict[str, Any]] = []
    for iteration in range(feature_iterations):
        prompt = build_feature_prompt(current, target, description, dataset)
        messages = [
            {"role": "system", "content": "You are an expert data scientist. Output executable Python code only."},
            {"role": "user", "content": prompt},
        ]
        raw, usage = call_llm(llm_model, messages, base_url, api_key, temperature=0.5, max_tokens=500)
        code = clean_code_block(raw)
        status = "ok"
        error = ""
        try:
            current = execute_feature_code(code, current, target)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "dataset": dataset,
                "seed": seed,
                "stage": "feature_generation",
                "iteration": iteration + 1,
                "status": status,
                "error": error,
                "code": code,
                **usage,
            }
        )
    feature_cols = [c for c in current.columns if c != target]
    x_aug = numeric_frame(current[feature_cols])
    x_aug_arr = x_aug.to_numpy(dtype=float)
    x_scaled = StandardScaler().fit_transform(x_aug_arr)
    raw_labels = KMeans(n_clusters=true_k, n_init=10, random_state=seed).fit_predict(x_aug_arr)
    scaled_labels = KMeans(n_clusters=true_k, n_init=10, random_state=seed).fit_predict(x_scaled)
    raw_ari = adjusted_rand_score(y_true, raw_labels)
    scaled_ari = adjusted_rand_score(y_true, scaled_labels)
    if raw_ari > scaled_ari:
        selected = pd.DataFrame(x_aug_arr, columns=feature_cols)
        method = f"llm_features_unscaled_selected_by_kmeans_ari(raw={raw_ari:.6f},scaled={scaled_ari:.6f})"
    else:
        selected = pd.DataFrame(x_scaled, columns=feature_cols)
        method = f"llm_features_scaled_selected_by_kmeans_ari(raw={raw_ari:.6f},scaled={scaled_ari:.6f})"
    return selected, method, records


def clustering_model_prompt(samples: str, summary: Dict[str, Any], sklearn_version: str) -> str:
    return f"""
You are to act as an expert in machine learning, especially unsupervised learning and clustering.

You will be provided with a dataset preview and statistical summary. Based on this information, estimate a reasonable number of clusters and generate Python code for clustering accordingly.

Dataset Preview (first 10 rows):
{samples}

Dataset Summary:
- Number of samples: {summary.get("n_samples")}
- Number of features: {summary.get("n_features")}
- Columns shown in prompt: {summary.get("columns_shown")}
- Number of additional columns omitted from prompt: {summary.get("n_columns_omitted_from_prompt")}
- Omitted columns preview: {summary.get("columns_omitted_preview")}
- Feature types: {summary.get("feature_types")}
- Feature ranges: {summary.get("feature_ranges")}
- Mean and standard deviation (numerical features only): {summary.get("mean_std")}

Task:
- Analyze the dataset summary to determine the optimal number of clusters automatically.
- Generate a Python class named `mycluster` for clustering.
- Each new version must use a different type of clustering algorithm or pipeline.
- The objective is to maximize Adjusted Rand Index (ARI).

Requirements for the class:
- Class name: `mycluster`
- Do not accept `n_clusters` as a constructor parameter.
- The optimal number of clusters should be hard-coded within the class based on your estimation.
- Must implement `fit_predict(X)` where X is a pandas DataFrame, returning predicted labels for all samples.
- Class must include all required imports.
- Be ready-to-execute Python code only.
- Avoid repeating previously used algorithms.
- Avoid text, explanations, or comments. Output only the class definition code.
- Do not use VotingClassifier.
- The installed scikit-learn version is {sklearn_version}; only use parameters supported by this version.
"""


def param_prompt(best_code: str, best_ari: float, dataset: str, description: str, x: pd.DataFrame, sklearn_version: str) -> str:
    shown_cols, omitted_cols = prompt_column_subset(x)
    table = x.loc[:, shown_cols].head(PROMPT_SAMPLE_ROWS).to_string(index=False)
    best_code = truncate_text(best_code, PROMPT_CODE_CHARS)
    description = truncate_text(description, PROMPT_DESCRIPTION_CHARS)
    return f"""
Here is the best clustering model code so far, with its current ARI score:

Current best ARI: {best_ari:.4f}

Model code:
```python
{best_code}
```

Dataset name: {dataset}
Dataset description:
{description}

Feature names shown in prompt:
{', '.join([str(c) for c in shown_cols])}

Dataset shape: {x.shape}
First rows for columns shown in prompt:
{table}
Additional columns omitted from first-rows prompt: {len(omitted_cols)}

Please only optimize the hyperparameters in the given clustering model code to
further improve ARI. Do not change the algorithm type or model structure.
Output only a new Python code block defining class `mycluster`.
Use only parameters supported by scikit-learn {sklearn_version}.
"""


def code_exec_model(code: str, class_name: str) -> Tuple[Optional[type], str]:
    namespace: Dict[str, Any] = {}
    try:
        exec(compile(code, "<cluster_model_code>", "exec"), namespace, namespace)
        if class_name not in namespace:
            return None, f"class {class_name} not found"
        return namespace[class_name], ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def rename_cluster_class(code: str, class_name: str) -> str:
    code = clean_cluster_code(code)
    code = re.sub(r"class\s+mycluster[_\w]*\s*:", f"class {class_name}:", code, count=1)
    return code


@dataclass
class ModelRecord:
    name: str
    code: str
    labels: np.ndarray
    ari: float
    nmi: float
    status: str
    error: str = ""


def instantiate_and_predict(model_cls: type, x: pd.DataFrame) -> np.ndarray:
    model = model_cls()
    labels = model.fit_predict(x)
    return np.asarray(labels)


def valid_labels(labels: np.ndarray, n_samples: int) -> bool:
    return labels is not None and len(labels) == n_samples and 1 < len(np.unique(labels)) < n_samples


def live_generate_models(
    x: pd.DataFrame,
    y_true: np.ndarray,
    dataset: str,
    description: str,
    seed: int,
    llm_model: str,
    base_url: str,
    api_key: str,
    model_iterations: int,
    param_iterations: int,
    sklearn_version: str,
) -> Tuple[List[ModelRecord], List[Dict[str, Any]]]:
    samples = build_prompt_samples(x)
    summary = summarize_dataset(x)
    system_message = {
        "role": "system",
        "content": (
            "You are a top-level clustering algorithm expert. "
            "You help design clustering models optimized for ARI. "
            "Your output must contain only Python code."
        ),
    }
    base_prompt = clustering_model_prompt(samples, summary, sklearn_version)
    records: List[ModelRecord] = []
    code_records: List[Dict[str, Any]] = []
    best_ari = -np.inf
    current_retry = 0
    max_retries = 3
    i = 0

    def compact_history() -> str:
        if not records:
            return ""
        lines = ["\nPreviously accepted candidate models; generate a meaningfully different next model:"]
        for idx, rec in enumerate(records[-6:], start=max(1, len(records) - 5)):
            code_excerpt = truncate_text(rec.code, 1000).replace("\n", " ")
            lines.append(f"- candidate_{idx}: ARI={rec.ari:.4f}, NMI={rec.nmi:.4f}, code_excerpt={code_excerpt}")
        return "\n".join(lines)

    def make_messages(extra_instruction: str = "") -> List[Dict[str, str]]:
        content = base_prompt + compact_history()
        if extra_instruction:
            content += "\n" + extra_instruction
        return [system_message, {"role": "user", "content": content}]

    messages = make_messages()
    while i < model_iterations:
        raw, usage = call_llm(llm_model, messages, base_url, api_key, temperature=0.7, max_tokens=700)
        class_name = f"mycluster_live_{dataset}_{seed}_{i + 1}"
        code = rename_cluster_class(raw, class_name)
        if len(code) > PROMPT_CODE_CHARS * 2:
            current_retry += 1
            err = f"generated code too long: {len(code)} chars"
            code_records.append({"dataset": dataset, "seed": seed, "stage": "model_generation", "iteration": i + 1, "status": "compile_failed", "error": err, "code": truncate_text(code, PROMPT_CODE_CHARS), **usage})
            if current_retry >= max_retries:
                current_retry = 0
                i += 1
            messages = make_messages(f"The previous output was invalid because {err}. Return a compact executable class only.")
            continue
        model_cls, err = code_exec_model(code, class_name)
        if model_cls is None:
            current_retry += 1
            code_records.append({"dataset": dataset, "seed": seed, "stage": "model_generation", "iteration": i + 1, "status": "compile_failed", "error": err, "code": code, **usage})
            if current_retry >= max_retries:
                current_retry = 0
                i += 1
            messages = make_messages(
                "The previous candidate failed to compile. "
                f"Error: {truncate_text(err, PROMPT_ERROR_CHARS)}. "
                f"Code excerpt: {truncate_text(code, 3000)}. "
                "Fix it and output only Python code."
            )
            continue
        try:
            labels = instantiate_and_predict(model_cls, x)
            if not valid_labels(labels, len(y_true)):
                raise ValueError(f"invalid labels: length={len(labels)}, unique={len(np.unique(labels))}")
            ari = float(adjusted_rand_score(y_true, labels))
            nmi = float(normalized_mutual_info_score(y_true, labels))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            current_retry += 1
            code_records.append({"dataset": dataset, "seed": seed, "stage": "model_generation", "iteration": i + 1, "status": "runtime_failed", "error": err, "code": code, **usage})
            if current_retry >= max_retries:
                current_retry = 0
                i += 1
            messages = make_messages(
                "The previous candidate failed at runtime or returned invalid cluster labels. "
                f"Error: {truncate_text(err, PROMPT_ERROR_CHARS)}. "
                f"Code excerpt: {truncate_text(code, 3000)}. "
                "Fix it and output only Python code."
            )
            continue

        current_retry = 0
        best_param_code = code
        best_param_ari = ari
        best_param_nmi = nmi
        best_param_labels = labels
        best_param_status = "ok"
        if len(best_param_code) <= PROMPT_CODE_CHARS:
            for p_iter in range(param_iterations):
                p_messages = [
                    {"role": "system", "content": "You are a clustering hyperparameter optimization assistant. Output Python code only."},
                    {"role": "user", "content": param_prompt(best_param_code, best_param_ari, dataset, description, x, sklearn_version)},
                ]
                try:
                    p_raw, p_usage = call_llm(llm_model, p_messages, base_url, api_key, temperature=0.7, max_tokens=700)
                    p_class_name = f"{class_name}_param_{p_iter + 1}"
                    p_code = rename_cluster_class(p_raw, p_class_name)
                    if len(p_code) > PROMPT_CODE_CHARS * 2:
                        raise RuntimeError(f"generated optimized code too long: {len(p_code)} chars")
                    p_cls, p_err = code_exec_model(p_code, p_class_name)
                    if p_cls is None:
                        raise RuntimeError(p_err)
                    p_labels = instantiate_and_predict(p_cls, x)
                    if not valid_labels(p_labels, len(y_true)):
                        raise ValueError(f"invalid labels: length={len(p_labels)}, unique={len(np.unique(p_labels))}")
                    p_ari = float(adjusted_rand_score(y_true, p_labels))
                    p_nmi = float(normalized_mutual_info_score(y_true, p_labels))
                    code_records.append({"dataset": dataset, "seed": seed, "stage": "param_optimization", "iteration": i + 1, "param_iteration": p_iter + 1, "status": "ok", "error": "", "ari": p_ari, "nmi": p_nmi, "code": p_code, **p_usage})
                    if p_ari > best_param_ari:
                        best_param_code = p_code
                        best_param_ari = p_ari
                        best_param_nmi = p_nmi
                        best_param_labels = p_labels
                except Exception as exc:
                    code_records.append({"dataset": dataset, "seed": seed, "stage": "param_optimization", "iteration": i + 1, "param_iteration": p_iter + 1, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "code": "", "prompt_tokens": np.nan, "completion_tokens": np.nan, "total_tokens": np.nan})
        else:
            code_records.append({"dataset": dataset, "seed": seed, "stage": "param_optimization", "iteration": i + 1, "param_iteration": 0, "status": "skipped", "error": f"base code too long for bounded prompt: {len(best_param_code)} chars", "code": "", "prompt_tokens": np.nan, "completion_tokens": np.nan, "total_tokens": np.nan})

        record = ModelRecord(
            name=f"model_{i + 1}",
            code=best_param_code,
            labels=best_param_labels,
            ari=best_param_ari,
            nmi=best_param_nmi,
            status=best_param_status,
        )
        records.append(record)
        code_records.append({"dataset": dataset, "seed": seed, "stage": "model_generation", "iteration": i + 1, "status": "ok", "error": "", "ari": best_param_ari, "nmi": best_param_nmi, "code": best_param_code, **usage})
        if best_param_ari > best_ari:
            best_ari = best_param_ari
        messages = make_messages(f"Current candidate ARI: {best_param_ari * 100:.2f}; best ARI so far: {best_ari * 100:.2f}. Generate the next different model.")
        i += 1
    records.sort(key=lambda r: r.ari, reverse=True)
    return records, code_records


class StudentNet(nn.Module):
    def __init__(self, input_dim: int, n_clusters: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_clusters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def distill_labels_mlp(x: pd.DataFrame, teacher_labels: np.ndarray, y_true: np.ndarray, seed: int, gradient_clip: bool, epochs: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_arr = StandardScaler().fit_transform(np.asarray(x, dtype=float))
    enc = LabelEncoder()
    teacher_y = enc.fit_transform(teacher_labels)
    n_clusters = max(len(enc.classes_), len(np.unique(y_true)))
    indices = np.arange(len(x_arr))
    train_idx, val_idx = train_test_split(indices, test_size=0.25, random_state=seed)
    x_train = torch.tensor(x_arr[train_idx], dtype=torch.float32)
    t_train = torch.tensor(teacher_y[train_idx], dtype=torch.long)
    x_val = torch.tensor(x_arr[val_idx], dtype=torch.float32)
    y_val = y_true[val_idx]
    model = StudentNet(x_arr.shape[1], n_clusters)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    batch_size = min(32, len(train_idx))
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train_idx))
        for start in range(0, len(train_idx), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_train[batch]), t_train[batch])
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(x_val).argmax(dim=1).cpu().numpy()
        score = adjusted_rand_score(y_val, pred)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x_arr, dtype=torch.float32)).argmax(dim=1).cpu().numpy()


def top_k(records: Sequence[ModelRecord]) -> List[ModelRecord]:
    if len(records) < 2:
        return list(records)
    k = max(min(len(records) // 2, 5), 2)
    return list(sorted(records, key=lambda r: r.ari, reverse=True)[:k])


def consensus_labels(records: Sequence[ModelRecord], n_clusters: int) -> np.ndarray:
    if len(records) < 2:
        raise RuntimeError(f"Need at least two valid models; got {len(records)}")
    labels = np.array([r.labels for r in records])
    n_models, n_samples = labels.shape
    matrix = np.zeros((n_samples, n_samples), dtype=float)
    for row in labels:
        matrix += (row[:, None] == row[None, :]).astype(float)
    matrix /= float(n_models)
    return SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42).fit_predict(matrix)


def internal_metrics(x: pd.DataFrame, labels: np.ndarray) -> Tuple[float, float, float, int]:
    if not valid_labels(labels, len(labels)):
        return np.nan, np.nan, np.nan, 1
    arr = np.asarray(x, dtype=float)
    try:
        return (
            float(silhouette_score(arr, labels)),
            float(davies_bouldin_score(arr, labels)),
            float(calinski_harabasz_score(arr, labels)),
            0,
        )
    except Exception:
        return np.nan, np.nan, np.nan, 1


def knn_overlap(x: pd.DataFrame, labels_a: np.ndarray, labels_b: np.ndarray, k: int = 10) -> float:
    arr = np.asarray(x, dtype=float)
    n_neighbors = min(k + 1, len(labels_a))
    if n_neighbors <= 1:
        return np.nan
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(arr)
    idx = nn.kneighbors(arr, return_distance=False)[:, 1:]
    return float(np.mean((labels_a[idx] == labels_a[:, None]) == (labels_b[idx] == labels_b[:, None])))


def eval_row(
    dataset: str,
    seed: int,
    config: str,
    feature_setting: str,
    model_family: str,
    gradient_clip: Optional[bool],
    x: pd.DataFrame,
    y_true: np.ndarray,
    labels: np.ndarray,
    true_k: int,
    target: str,
    selected_models: Sequence[ModelRecord],
    teacher_labels: Optional[np.ndarray],
    feature_method: str,
) -> Dict[str, Any]:
    sil, db, ch, failed = internal_metrics(x, labels)
    ari = float(adjusted_rand_score(y_true, labels))
    nmi = float(normalized_mutual_info_score(y_true, labels))
    row = {
        "task": "clustering",
        "dataset": dataset,
        "seed": seed,
        "config": config,
        "feature_setting": feature_setting,
        "model_family": model_family,
        "postprocessing_method": "autoLOGIC_new_consensus_matrix_spectral",
        "gradient_clip": gradient_clip,
        "target_column": target,
        "n_total": int(len(y_true)),
        "true_k": int(true_k),
        "selected_k": int(len(np.unique(labels))),
        "ari": ari,
        "nmi": nmi,
        "ari_percent": ari * 100,
        "nmi_percent": nmi * 100,
        "silhouette": sil,
        "db": db,
        "ch": ch,
        "failed_flag": int(failed),
        "failure_reason": "invalid_internal_metric" if failed else "",
        "candidate_count": int(len(selected_models)),
        "used_model_count": int(len(selected_models)),
        "selected_model_names": ";".join([m.name for m in selected_models]),
        "selected_model_ari_percent": ";".join([f"{m.ari * 100:.6f}" for m in selected_models]),
        "model_selection_method": "ground_truth_ari_topk_matches_autoLOGIC_new",
        "feature_generation_method": feature_method,
        "teacher_student_ari": np.nan,
        "teacher_student_nmi": np.nan,
        "knn_overlap": np.nan,
        "retention_ratio": np.nan,
    }
    if teacher_labels is not None:
        row["teacher_student_ari"] = float(adjusted_rand_score(teacher_labels, labels))
        row["teacher_student_nmi"] = float(normalized_mutual_info_score(teacher_labels, labels))
        row["knn_overlap"] = knn_overlap(x, teacher_labels, labels)
        teacher_ari = adjusted_rand_score(y_true, teacher_labels)
        row["retention_ratio"] = ari / teacher_ari if abs(teacher_ari) > EPS else np.nan
    return row


def split_membership(y_true: np.ndarray, seed: int) -> np.ndarray:
    indices = np.arange(len(y_true))
    stratify = y_true if len(np.unique(y_true)) > 1 and min(np.bincount(y_true)) >= 2 else None
    try:
        _, test_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=stratify)
    except Exception:
        _, test_idx = train_test_split(indices, test_size=0.30, random_state=seed)
    split = np.full(len(y_true), "train_or_fit", dtype=object)
    split[np.asarray(test_idx, dtype=int)] = "panel_d_test"
    return split


def assignment_rows(
    dataset: str,
    seed: int,
    config: str,
    feature_setting: str,
    model_family: str,
    x: pd.DataFrame,
    y_true: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    teacher_labels: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    labels = np.asarray(labels).reshape(-1)
    teacher_arr = np.asarray(teacher_labels).reshape(-1) if teacher_labels is not None else None
    rows: List[Dict[str, Any]] = []
    for idx in range(len(labels)):
        row = {
            "task": "clustering",
            "dataset": dataset,
            "seed": seed,
            "sample_id": idx,
            "split": str(split[idx]),
            "config": config,
            "feature_setting": feature_setting,
            "model_family": model_family,
            "y_true": int(y_true[idx]),
            "cluster_label": int(labels[idx]),
            "n_features": int(x.shape[1]),
        }
        if teacher_arr is not None:
            row["teacher_cluster_label"] = int(teacher_arr[idx])
            row["student_cluster_label"] = int(labels[idx])
        else:
            row["teacher_cluster_label"] = int(labels[idx])
            row["student_cluster_label"] = np.nan
        rows.append(row)
    return rows


def assignment_fidelity_rows(
    dataset: str,
    seed: int,
    config: str,
    feature_setting: str,
    x: pd.DataFrame,
    teacher_labels: np.ndarray,
    student_labels: np.ndarray,
    split: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    teacher_arr = np.asarray(teacher_labels).reshape(-1)
    student_arr = np.asarray(student_labels).reshape(-1)
    for split_name in ["all", "panel_d_test"]:
        mask = np.ones(len(teacher_arr), dtype=bool) if split_name == "all" else np.asarray(split) == split_name
        if int(mask.sum()) < 2:
            continue
        t = teacher_arr[mask]
        s = student_arr[mask]
        x_sub = x.iloc[np.where(mask)[0], :]
        rows.append(
            {
                "task": "clustering",
                "dataset": dataset,
                "seed": seed,
                "split": split_name,
                "config": config,
                "feature_setting": feature_setting,
                "n_samples": int(mask.sum()),
                "assignment_ari": float(adjusted_rand_score(t, s)),
                "assignment_nmi": float(normalized_mutual_info_score(t, s)),
                "rand_index": float(np.mean((t[:, None] == t[None, :]) == (s[:, None] == s[None, :]))),
                "knn_overlap": knn_overlap(x_sub, t, s, k=10),
            }
        )
    return rows


def run_dataset_seed(args: argparse.Namespace, repo_root: Path, dataset: str, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    df, target, description = load_dataset(repo_root, dataset)
    y_true = LabelEncoder().fit_transform(df[target].astype(str))
    split = split_membership(y_true, seed)
    true_k = len(np.unique(y_true))
    x_orig = numeric_frame(df.drop(columns=[target]))
    x_no_feat = pd.DataFrame(StandardScaler().fit_transform(x_orig), columns=x_orig.columns)
    code_records: List[Dict[str, Any]] = []
    try:
        x_with_feat, feature_method, feature_records = live_generate_features(
            df=df,
            target=target,
            y_true=y_true,
            true_k=true_k,
            dataset=dataset,
            description=description,
            seed=seed,
            llm_model=args.llm_model,
            base_url=args.base_url,
            api_key=args.api_key,
            feature_iterations=args.feature_iterations,
        )
        code_records.extend(feature_records)
    except Exception as exc:
        x_with_feat = None
        feature_method = f"feature_generation_failed: {type(exc).__name__}: {exc}"
        code_records.append(
            {
                "dataset": dataset,
                "seed": seed,
                "stage": "feature_generation",
                "iteration": 0,
                "status": "failed",
                "error": feature_method,
                "code": "",
                "prompt_tokens": np.nan,
                "completion_tokens": np.nan,
                "total_tokens": np.nan,
            }
        )
    rows: List[Dict[str, Any]] = []
    assignment_records: List[Dict[str, Any]] = []
    assignment_fidelity: List[Dict[str, Any]] = []
    for feature_setting, x_current, feature_label in [
        ("with_features", x_with_feat, feature_method),
        ("no_features", x_no_feat, "raw_standardized_no_feature_generation"),
    ]:
        if x_current is None:
            rows.extend(failed_feature_rows(dataset, seed, feature_setting, feature_label))
            continue
        try:
            models, model_code_records = live_generate_models(
                x=x_current,
                y_true=y_true,
                dataset=dataset,
                description=description,
                seed=seed,
                llm_model=args.llm_model,
                base_url=args.base_url,
                api_key=args.api_key,
                model_iterations=args.model_iterations,
                param_iterations=args.param_iterations,
                sklearn_version=args.sklearn_version,
            )
            code_records.extend([{**r, "feature_setting": feature_setting} for r in model_code_records])
            teacher_top = top_k(models)
            if len(teacher_top) < 2:
                raise RuntimeError(f"{dataset} seed={seed} {feature_setting}: fewer than two valid live models")
            teacher_labels = consensus_labels(teacher_top, true_k)
            teacher_config = f"{feature_setting}_teacher_consensus"
            rows.append(
                eval_row(dataset, seed, teacher_config, feature_setting, "teacher_stack", None, x_current, y_true, teacher_labels, true_k, target, teacher_top, None, feature_label)
            )
            assignment_records.extend(
                assignment_rows(dataset, seed, teacher_config, feature_setting, "teacher_stack", x_current, y_true, teacher_labels, split, None)
            )
            for gradient_clip, suffix in [(True, "student_consensus_clip"), (False, "student_consensus_no_clip")]:
                try:
                    student_records: List[ModelRecord] = []
                    for i, model in enumerate(teacher_top):
                        labels = distill_labels_mlp(x_current, model.labels, y_true, seed + 1000 + i, gradient_clip, args.distill_epochs)
                        if not valid_labels(labels, len(y_true)):
                            continue
                        student_records.append(
                            ModelRecord(
                                name=f"student_{'clip' if gradient_clip else 'no_clip'}_{model.name}",
                                code="MLP_student_distillation",
                                labels=labels,
                                ari=float(adjusted_rand_score(y_true, labels)),
                                nmi=float(normalized_mutual_info_score(y_true, labels)),
                                status="ok",
                            )
                        )
                    student_top = top_k(student_records)
                    if len(student_top) < 2:
                        raise RuntimeError(f"{dataset} seed={seed} {feature_setting} {suffix}: fewer than two valid student models")
                    student_labels = consensus_labels(student_top, true_k)
                    student_config = f"{feature_setting}_{suffix}"
                    rows.append(
                        eval_row(dataset, seed, student_config, feature_setting, "distilled_student", gradient_clip, x_current, y_true, student_labels, true_k, target, student_top, teacher_labels, feature_label)
                    )
                    assignment_records.extend(
                        assignment_rows(dataset, seed, student_config, feature_setting, "distilled_student", x_current, y_true, student_labels, split, teacher_labels)
                    )
                    assignment_fidelity.extend(
                        assignment_fidelity_rows(dataset, seed, student_config, feature_setting, x_current, teacher_labels, student_labels, split)
                    )
                except Exception as exc:
                    rows.append(failed_single_config_row(dataset, seed, feature_setting, suffix, "distilled_student", gradient_clip, f"{type(exc).__name__}: {exc}"))
        except Exception as exc:
            rows.extend(failed_feature_rows(dataset, seed, feature_setting, f"{type(exc).__name__}: {exc}"))
    return rows, code_records, assignment_records, assignment_fidelity


def failed_single_config_row(dataset: str, seed: int, feature: str, suffix: str, family: str, clip: Optional[bool], reason: str) -> Dict[str, Any]:
    return {
        "task": "clustering",
        "dataset": dataset,
        "seed": seed,
        "config": f"{feature}_{suffix}",
        "feature_setting": feature,
        "model_family": family,
        "gradient_clip": clip,
        "postprocessing_method": "autoLOGIC_new_consensus_matrix_spectral",
        "failed_flag": 1,
        "failure_reason": reason,
    }


def failed_feature_rows(dataset: str, seed: int, feature: str, reason: str) -> List[Dict[str, Any]]:
    return [
        failed_single_config_row(dataset, seed, feature, "teacher_consensus", "teacher_stack", None, reason),
        failed_single_config_row(dataset, seed, feature, "student_consensus_clip", "distilled_student", True, reason),
        failed_single_config_row(dataset, seed, feature, "student_consensus_no_clip", "distilled_student", False, reason),
    ]


def failed_rows(dataset: str, seed: int, reason: str) -> List[Dict[str, Any]]:
    rows = []
    for feature in ["with_features", "no_features"]:
        for suffix, family, clip in [
            ("teacher_consensus", "teacher_stack", None),
            ("student_consensus_clip", "distilled_student", True),
            ("student_consensus_no_clip", "distilled_student", False),
        ]:
            rows.append(
                {
                    "task": "clustering",
                    "dataset": dataset,
                    "seed": seed,
                    "config": f"{feature}_{suffix}",
                    "feature_setting": feature,
                    "model_family": family,
                    "gradient_clip": clip,
                    "postprocessing_method": "autoLOGIC_new_consensus_matrix_spectral",
                    "failed_flag": 1,
                    "failure_reason": reason,
                }
            )
    return rows


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "config", "feature_setting", "model_family", "postprocessing_method", "gradient_clip"]
    metric_cols = [
        "ari",
        "nmi",
        "ari_percent",
        "nmi_percent",
        "selected_k",
        "silhouette",
        "db",
        "ch",
        "teacher_student_ari",
        "teacher_student_nmi",
        "knn_overlap",
        "retention_ratio",
    ]
    rows = []
    for key, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row["n_rows"] = int(len(group))
        row["n_seeds"] = int(group["seed"].nunique())
        row["failed_count"] = int(pd.to_numeric(group["failed_flag"], errors="coerce").fillna(0).sum())
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce") if col in group.columns else pd.Series(dtype=float)
            row[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{col}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
            row[f"{col}_n"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_assignment_fidelity(assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame()
    rows = []
    student = assignments[assignments["model_family"].eq("distilled_student")].copy()
    for key, group in student.groupby(["dataset", "seed", "split", "config", "feature_setting"], dropna=False, sort=True):
        dataset, seed, split, config, feature_setting = key
        teacher = pd.to_numeric(group.get("teacher_cluster_label"), errors="coerce")
        student_labels = pd.to_numeric(group.get("student_cluster_label"), errors="coerce")
        valid = teacher.notna() & student_labels.notna()
        if valid.sum() < 2:
            continue
        teacher_arr = teacher[valid].astype(int).to_numpy()
        student_arr = student_labels[valid].astype(int).to_numpy()
        x_proxy = group.loc[valid, ["sample_id"]].astype(float)
        rows.append(
            {
                "task": "clustering",
                "dataset": dataset,
                "seed": int(seed),
                "split": split,
                "config": config,
                "feature_setting": feature_setting,
                "n_samples": int(valid.sum()),
                "assignment_ari": float(adjusted_rand_score(teacher_arr, student_arr)),
                "assignment_nmi": float(normalized_mutual_info_score(teacher_arr, student_arr)),
                "rand_index": float(np.mean((teacher_arr[:, None] == teacher_arr[None, :]) == (student_arr[:, None] == student_arr[None, :]))),
                "label_match_fraction_raw": float(np.mean(teacher_arr == student_arr)),
            }
        )
    return pd.DataFrame(rows)


def parse_mean_std(value: Any) -> Tuple[float, float]:
    text = str(value)
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[±¡À]\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def load_comparison(repo_root: Path) -> pd.DataFrame:
    target = None
    for path in repo_root.parent.glob("*.xlsx"):
        if "对比" in path.name and "汇总" in path.name:
            target = path
            break
    if target is None:
        return pd.DataFrame()
    raw = pd.read_excel(target, sheet_name="Clustering", header=None)
    rows = []
    current_metric = None
    header_seen = False
    for _, row in raw.iterrows():
        first = row.iloc[0]
        if isinstance(first, str) and first.strip() in {"ARI (%)", "NMI (%)"}:
            current_metric = "ari" if first.strip().startswith("ARI") else "nmi"
            header_seen = False
            continue
        if isinstance(first, str) and first.strip() == "Dataset":
            header_seen = True
            continue
        if current_metric and header_seen and isinstance(first, str) and first.strip():
            mean, std = parse_mean_std(row.iloc[5])
            if np.isfinite(mean):
                rows.append({"dataset": first.strip(), "metric": current_metric, "comparison_auto_logic_mean_percent": mean, "comparison_auto_logic_std_percent": std})
    return pd.DataFrame(rows)


def build_comparison(summary: pd.DataFrame, comparison_raw: pd.DataFrame) -> pd.DataFrame:
    if comparison_raw.empty:
        return pd.DataFrame()
    best_rows = []
    for dataset, group in summary.groupby("dataset", sort=True):
        best_ari = group.sort_values("ari_percent_mean", ascending=False).iloc[0]
        best_nmi = group.sort_values("nmi_percent_mean", ascending=False).iloc[0]
        best_rows.append(
            {
                "dataset": dataset,
                "rerun_best_ari_config": best_ari["config"],
                "rerun_best_ari_percent_mean": best_ari["ari_percent_mean"],
                "rerun_best_ari_percent_std": best_ari["ari_percent_std"],
                "rerun_best_nmi_config": best_nmi["config"],
                "rerun_best_nmi_percent_mean": best_nmi["nmi_percent_mean"],
                "rerun_best_nmi_percent_std": best_nmi["nmi_percent_std"],
            }
        )
    best = pd.DataFrame(best_rows)
    ari = comparison_raw[comparison_raw["metric"] == "ari"].rename(
        columns={
            "comparison_auto_logic_mean_percent": "comparison_ari_percent_mean",
            "comparison_auto_logic_std_percent": "comparison_ari_percent_std",
        }
    )[["dataset", "comparison_ari_percent_mean", "comparison_ari_percent_std"]]
    nmi = comparison_raw[comparison_raw["metric"] == "nmi"].rename(
        columns={
            "comparison_auto_logic_mean_percent": "comparison_nmi_percent_mean",
            "comparison_auto_logic_std_percent": "comparison_nmi_percent_std",
        }
    )[["dataset", "comparison_nmi_percent_mean", "comparison_nmi_percent_std"]]
    out = best.merge(ari, on="dataset", how="left").merge(nmi, on="dataset", how="left")
    out["ari_gap_rerun_minus_comparison_pp"] = out["rerun_best_ari_percent_mean"] - out["comparison_ari_percent_mean"]
    out["nmi_gap_rerun_minus_comparison_pp"] = out["rerun_best_nmi_percent_mean"] - out["comparison_nmi_percent_mean"]
    out["large_ari_gap_abs_gt_5pp"] = out["ari_gap_rerun_minus_comparison_pp"].abs() > 5
    out["large_nmi_gap_abs_gt_5pp"] = out["nmi_gap_rerun_minus_comparison_pp"].abs() > 5
    return out


def write_outputs(output_dir: Path, tables: Mapping[str, pd.DataFrame], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    with pd.ExcelWriter(output_dir / "fig6_panel_b_source_data.xlsx", engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    manifest = {
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "datasets": args.datasets,
        "seeds": args.seeds,
        "llm_model": args.llm_model,
        "base_url": args.base_url,
        "api_key_recorded": False,
        "feature_iterations": args.feature_iterations,
        "model_iterations": args.model_iterations,
        "param_iterations": args.param_iterations,
        "distill_epochs": args.distill_epochs,
        "sklearn_version": args.sklearn_version,
        "autoLOGIC_new_alignment": {
            "live_llm_feature_generation_executed": True,
            "live_llm_model_generation_executed": True,
            "uses_ground_truth_ari_for_feature_selection": True,
            "uses_ground_truth_ari_for_model_selection": True,
            "uses_true_k_in_consensus": True,
            "uses_consensus_matrix_spectral": True,
            "uses_mlp_student_distillation_with_clip_and_no_clip": True,
        },
    }
    (output_dir / "configs_manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8")
    readme = f"""# Live LLM autoLOGIC_new clustering rerun

Command:

```bash
python reproduce/fig6/run_fig6b_live_clustering.py --output {output_dir}
```

This run executes live LLM feature generation and live LLM clustering model
generation. Summaries are aggregated from `clustering_per_seed.csv`.
The API key is not written to this directory.
"""
    (output_dir / "run_readme.md").write_text(readme, encoding="utf-8")


def verify_summary(per_seed: pd.DataFrame, summary: pd.DataFrame) -> int:
    group_cols = ["dataset", "config", "feature_setting", "model_family", "postprocessing_method", "gradient_clip"]
    metrics = ["ari", "nmi", "ari_percent", "nmi_percent", "selected_k", "silhouette", "db", "ch"]
    problems = 0
    for key, group in per_seed.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        mask = pd.Series([True] * len(summary))
        for col, val in zip(group_cols, key):
            if pd.isna(val):
                mask &= summary[col].isna()
            else:
                mask &= summary[col].astype(str).eq(str(val))
        if mask.sum() != 1:
            problems += 1
            continue
        row = summary[mask].iloc[0]
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            mean = values.mean()
            if pd.notna(mean) and abs(mean - row[f"{metric}_mean"]) > 1e-10:
                problems += 1
    return problems


def run_all(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    per_seed_rows: List[Dict[str, Any]] = []
    code_rows: List[Dict[str, Any]] = []
    assignment_rows_all: List[Dict[str, Any]] = []
    assignment_fidelity_rows_all: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"[live-clustering] dataset={dataset} seed={seed}", flush=True)
            try:
                rows, codes, assignments, assignment_fidelity_rows_one = run_dataset_seed(args, repo_root, dataset, int(seed))
                per_seed_rows.extend(rows)
                code_rows.extend(codes)
                assignment_rows_all.extend(assignments)
                assignment_fidelity_rows_all.extend(assignment_fidelity_rows_one)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                print(f"[failed] dataset={dataset} seed={seed}: {reason}", flush=True)
                traceback.print_exc()
                per_seed_rows.extend(failed_rows(dataset, int(seed), reason))
            # incremental checkpoint
            pd.DataFrame(per_seed_rows).to_csv(output_dir / "clustering_per_seed.csv", index=False)
            pd.DataFrame(code_rows).to_csv(output_dir / "llm_generation_records.csv", index=False)
            pd.DataFrame(assignment_rows_all).to_csv(output_dir / "clustering_assignment_records.csv", index=False)
            pd.DataFrame(assignment_fidelity_rows_all).to_csv(output_dir / "clustering_assignment_fidelity.csv", index=False)
    per_seed = pd.DataFrame(per_seed_rows)
    code_records = pd.DataFrame(code_rows)
    assignment_records = pd.DataFrame(assignment_rows_all)
    assignment_fidelity = pd.DataFrame(assignment_fidelity_rows_all)
    summary = aggregate_summary(per_seed)
    if assignment_fidelity.empty:
        assignment_fidelity = aggregate_assignment_fidelity(assignment_records)
    comparison = build_comparison(summary, load_comparison(repo_root))
    instability = per_seed.merge(
        summary[
            (pd.to_numeric(summary["ari_percent_std"], errors="coerce").abs() > 5)
            | (pd.to_numeric(summary["nmi_percent_std"], errors="coerce").abs() > 5)
            | (pd.to_numeric(summary["failed_count"], errors="coerce") > 0)
        ][["dataset", "config"]],
        on=["dataset", "config"],
        how="inner",
    )
    tables = {
        "clustering_per_seed": per_seed,
        "clustering_summary": summary,
        "clustering_vs_comparison": comparison,
        "clustering_instability": instability,
        "llm_generation_records": code_records,
        "clustering_assignment_records": assignment_records,
        "clustering_assignment_fidelity": assignment_fidelity,
    }
    write_outputs(output_dir, tables, args)
    problems = verify_summary(per_seed, summary)
    report = [
        "# Live LLM autoLOGIC_new clustering execution report",
        "",
        f"- clustering_per_seed.csv: {len(per_seed)} rows",
        f"- clustering_summary.csv: {len(summary)} rows",
        f"- clustering_vs_comparison.csv: {len(comparison)} rows",
        f"- llm_generation_records.csv: {len(code_records)} rows",
        f"- clustering_assignment_records.csv: {len(assignment_records)} rows",
        f"- clustering_assignment_fidelity.csv: {len(assignment_fidelity)} rows",
        f"- failed rows: {int(pd.to_numeric(per_seed['failed_flag'], errors='coerce').fillna(0).sum()) if 'failed_flag' in per_seed else 0}",
        f"- summary recomputation problems: {problems}",
        "",
        "Comparison against autoLOGIC comparison workbook:",
    ]
    for _, row in comparison.iterrows():
        report.append(
            f"- {row['dataset']}: ARI gap {row['ari_gap_rerun_minus_comparison_pp']:.2f} pp; "
            f"NMI gap {row['nmi_gap_rerun_minus_comparison_pp']:.2f} pp"
        )
    (output_dir / "execution_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results_fig6_panel_b_rerun_autologic_new_clustering")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--feature-iterations", type=int, default=1)
    parser.add_argument("--model-iterations", type=int, default=7)
    parser.add_argument("--param-iterations", type=int, default=1)
    parser.add_argument("--distill-epochs", type=int, default=30)
    parser.add_argument("--sklearn-version", default="")
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise ValueError("base URL and API key are required, via arguments or environment variables")
    if not args.sklearn_version:
        import sklearn

        args.sklearn_version = sklearn.__version__
    return args


def main() -> None:
    args = parse_args()
    print(f"[start] python={sys.executable}", flush=True)
    print(f"[start] output={args.output}", flush=True)
    print(f"[start] datasets={args.datasets}", flush=True)
    print(f"[start] seeds={args.seeds}", flush=True)
    tables = run_all(args)
    for name, df in tables.items():
        print(f"[done] {name}: rows={len(df)}", flush=True)


if __name__ == "__main__":
    main()
