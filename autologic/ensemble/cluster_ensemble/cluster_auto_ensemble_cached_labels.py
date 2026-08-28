import os
import sys

# Add project root directory
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   # Current file directory
                 "..", "..")                  # Go up two levels
)
sys.path.append(project_root)

from mystage1.run_llm_code import run_llm_code
from mystage1 import Stage1Classifier  # Automated Feature Engineering for tabular datasets
from sklearn.cluster import KMeans
from utils.model_generate import (
    build_prompt_samples,
    generate_model,
    get_clustering_model_prompt,
    get_clustering_model_prompt_v2,
)

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pickle
import re
import numpy as np
import random
import argparse
import warnings
import copy
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

from ensemble.cluster_ensemble_utils import (
    select_top_models_with_adaptive_threshold,
    cluster_ensemble_caps,
    get_param_prompt,
    distill_to_student_clustering,
    stacking_clustering_with_calibration,
    consensus_ensemble_topk,
)

from utils.utils import format_mean_std
from utils.quant_log import QuantLogger
from utils.csv_dataset_loader import load_csv_dataframe, resolve_description_path, read_txt_file
from pathlib import Path
from utils.time_limit import time_limit, TimeoutExceeded

STAGE_TIMEOUT_S = 300

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


def _cached_label_record(name, labels, ari, nmi, model=None, source="teacher"):
    labels = np.asarray(labels).astype(int, copy=False)
    return {
        "name": str(name),
        "labels": labels.copy(),
        "ari": float(ari),
        "nmi": float(nmi),
        "model": model,
        "source": source,
    }


def _evaluate_cached_record(model, X, y_true, name, source="teacher"):
    labels = model.fit_predict(X)
    labels = _validate_cluster_labels(labels, X.shape[0])
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    return _cached_label_record(name, labels, ari, nmi, model=model, source=source)


def _pairwise_ari_mean_cached(labels_list):
    labels_list = [np.asarray(x) for x in labels_list]
    if len(labels_list) < 2:
        return None
    vals = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            vals.append(adjusted_rand_score(labels_list[i], labels_list[j]))
    return float(np.mean(vals)) if vals else None


def consensus_ensemble_cached_labels(label_records, y_true, n_clusters=None, max_models=5, min_models=2, min_ari=0.30):
    """Consensus from already-validated labels; never refits base clusterers."""
    y_true = np.asarray(y_true)
    valid = []
    for rec in label_records:
        labels = np.asarray(rec.get("labels", []))
        if labels.shape[0] != y_true.shape[0]:
            continue
        if len(np.unique(labels)) <= 1:
            continue
        ari = float(rec.get("ari", float("nan")))
        if not np.isfinite(ari):
            continue
        valid.append(rec)

    if not valid:
        raise ValueError("No valid cached label records for ensemble")

    valid.sort(key=lambda r: float(r["ari"]), reverse=True)
    eligible = [r for r in valid if float(r["ari"]) >= float(min_ari)]
    fallback_used = False
    fallback_reason = None

    if len(eligible) >= min_models:
        k = len(eligible) // 2
        k = max(k, min_models)
        k = min(k, max_models, len(eligible))
        selected = eligible[:k]
    else:
        selected = valid[:1]
        fallback_used = True
        fallback_reason = "best_single_below_min_models_or_threshold"

    if len(selected) == 1:
        final_labels = np.asarray(selected[0]["labels"]).astype(int, copy=False)
    else:
        from sklearn.cluster import SpectralClustering

        base_labels = np.asarray([r["labels"] for r in selected]).astype(int, copy=False)
        S = np.zeros((y_true.shape[0], y_true.shape[0]), dtype=np.float32)
        for labels in base_labels:
            S += np.equal.outer(labels, labels).astype(np.float32)
        S /= float(len(selected))
        if n_clusters is None:
            n_clusters = len(np.unique(y_true))
        final_labels = SpectralClustering(
            n_clusters=int(n_clusters),
            affinity="precomputed",
            random_state=42,
        ).fit_predict(S)

    ari = adjusted_rand_score(y_true, final_labels)
    nmi = normalized_mutual_info_score(y_true, final_labels)
    selected_ari = [float(r["ari"]) for r in selected]
    return {
        "ensemble_metrics": {
            "ensemble_ari": float(ari),
            "ensemble_nmi": float(nmi),
            "base_pairwise_ari_mean": _pairwise_ari_mean_cached([r["labels"] for r in selected]),
            "used_model_count": int(len(selected)),
            "selected_model_ari": selected_ari,
            "eligible_model_count": int(len(eligible)),
            "cached_label_mode": True,
            "min_ari_threshold": float(min_ari),
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
        },
        "ensemble_labels": final_labels,
        "selected_model_count": int(len(selected)),
        "selected_model_ari": selected_ari,
        "eligible_model_count": int(len(eligible)),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
    }


def _encode_non_numeric_features(df: pd.DataFrame, target_column_name: str) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in df.columns if c != target_column_name]
    for c in feature_cols:
        col = df[c]
        if pd.api.types.is_bool_dtype(col):
            df[c] = col.astype(int)
            continue

        if pd.api.types.is_numeric_dtype(col):
            df[c] = pd.to_numeric(col, errors="coerce").fillna(0).astype("float32")
            continue

        dt = pd.to_datetime(col, errors="coerce")
        if dt.notna().any():
            df[c] = dt.view("int64").fillna(0).astype("float32")
            continue

        col_str = col.astype("string").fillna("__missing__").str.strip()
        cat = pd.Categorical(col_str)
        df[c] = pd.Series(cat.codes, index=df.index, dtype="float32")

    return df


# LoadLocal dataset
def load_origin_data_2(loc, seed=0):
    # Load dataset
    with open(loc, 'rb') as f:
        ds = pickle.load(f)

    df_train = ds[1]
    df_test = ds[2]
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train.fillna(0, inplace=True)
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.fillna(0, inplace=True)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    target_column_name = ds[4][-1]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Dataset description, including column descriptions
    dataset_description = ds[-1]
    # print(dataset_description)
    n_clusters = len(np.unique(df[target_column_name]))

    df = _encode_non_numeric_features(df, target_column_name)
    return df, n_clusters, target_column_name, dataset_description

# LoadLocal dataset
def load_origin_data(loc):
    if os.path.exists(loc):
        with open(loc, 'rb') as f:
            ds = pickle.load(f)
        target_column_name = ds[4][-1]
        df = ds[1]
        dataset_description = ds[-1]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        n_clusters = len(np.unique(df[target_column_name]))
        df = _encode_non_numeric_features(df, target_column_name)
        return df, n_clusters, target_column_name, dataset_description

    dataset_name = os.path.splitext(os.path.basename(loc))[0]
    csv_base = os.path.join(project_root, "data", "csv_data")
    df, target_column_name = load_csv_dataframe(Path(csv_base) / f"{dataset_name}.csv")
    desc_path = resolve_description_path(csv_base, dataset_name)
    dataset_description = read_txt_file(desc_path) if desc_path else ""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    n_clusters = len(np.unique(df[target_column_name]))
    df = _encode_non_numeric_features(df, target_column_name)
    return df, n_clusters, target_column_name, dataset_description

def base_model(n_clusters, seed):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return kmeans

def fallback_models(n_clusters, seed):
    from sklearn.cluster import MiniBatchKMeans, Birch
    models = [
        MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=2048, n_init=3),
        Birch(n_clusters=n_clusters),
    ]
    for idx, m in enumerate(models, start=1):
        try:
            m.name = f"fallback_{idx}"
        except Exception:
            pass
    return models


def to_pd(df, target_name):
    y = df[target_name]
    x = df.drop(target_name, axis=1)

    return x, y


# Execute generated code
def _reject_quadratic_generated_code(code: str):
    text = code or ""
    lowered = text.lower()
    banned_patterns = [
        "pairwise_distances",
        "pairwise_kernels",
        "euclidean_distances",
        "cosine_similarity",
        "sklearn.metrics.pairwise",
        "scipy.spatial.distance.pdist",
        "scipy.spatial.distance.cdist",
        "squareform",
        "agglomerativeclustering",
        "spectralclustering",
        "meanshift",
        "dbscan",
        "affinitypropagation",
        "optics",
        "pairwise",
        "all_pairs",
        "affinity='precomputed'",
        'affinity="precomputed"',
        "metric='precomputed'",
        'metric="precomputed"',
        "precomputed",
        "distance_matrix",
        "connectivity_matrix",
        "coassociation",
        "co-association",
    ]
    for pattern in banned_patterns:
        if pattern in lowered:
            return f"FORBIDDEN_O_N2_CODE: generated clustering code contains `{pattern}`"

    matrix_patterns = [
        r"np\.(zeros|ones|empty|full)\s*\(\s*\(\s*(?:n_samples|len\(\s*X\s*\)|X\.shape\s*\[\s*0\s*\])\s*,\s*(?:n_samples|len\(\s*X\s*\)|X\.shape\s*\[\s*0\s*\])",
        r"np\.(zeros|ones|empty|full)\s*\(\s*\[\s*(?:n_samples|len\(\s*X\s*\)|X\.shape\s*\[\s*0\s*\])\s*,\s*(?:n_samples|len\(\s*X\s*\)|X\.shape\s*\[\s*0\s*\])",
        r"np\.repeat\s*\(.*X\.shape\s*\[\s*0\s*\]",
        r"np\.tile\s*\(.*X\.shape\s*\[\s*0\s*\]",
        r"@\s*(?:X|x|X_[A-Za-z0-9_]+|[A-Za-z0-9_]+)\.T",
        r"np\.matmul\s*\(.*,\s*.*\.T\s*\)",
        r"np\.dot\s*\(.*,\s*.*\.T\s*\)",
    ]
    for pattern in matrix_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            return "FORBIDDEN_O_N2_CODE: generated clustering code constructs an n_samples by n_samples object"
    return None


def _append_retry_feedback(messages, code, error_text, retry_text, error_hint=""):
    forbidden = "FORBIDDEN_O_N2_CODE" in str(error_text)
    if forbidden:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Generated code was rejected before execution: {error_text}\n"
                    "Do not use pairwise/all-pairs/full distance matrices, MeanShift, DBSCAN, OPTICS, "
                    "AffinityPropagation, AgglomerativeClustering, SpectralClustering, precomputed affinities, "
                    "or any n_samples x n_samples construct. Generate a scalable model using KMeans, "
                    "MiniBatchKMeans, GaussianMixture, Birch, PCA/IncrementalPCA, and simple preprocessing only.\n"
                    f"{retry_text}"
                ),
            }
        )
    else:
        messages += [
            {"role": "assistant", "content": code},
            {
                "role": "user",
                "content": f"""
                Code execution failed with error: {error_text}
                Code: ```python{code}```{error_hint}
                
                {retry_text}
                """,
            },
        ]


def code_exec(code):
    try:
        blocked = _reject_quadratic_generated_code(code)
        if blocked is not None:
            raise ValueError(blocked)
        compiled_code = compile(code, "<string>", "exec")
        scope = {"__builtins__": __builtins__}
        exec(compiled_code, scope, scope)
        return None, scope
    except Exception as e:
        print("Code could not be executed:", e)
        return str(e), None


# Calculate statistical metrics
def print_stats(name, values):
    print(f"{name}: {np.mean(values):.2f} ± {np.std(values):.2f}")


def print_rmsle(name, values):
    print(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def clean_llm_code(code: str) -> str:
    import re
    # Remove ``` code block markers at beginning and end
    code = re.sub(r"^```python\s*", "", code.strip(), flags=re.IGNORECASE)
    code = re.sub(r"```$", "", code.strip())

    # Remove <end> and non-code text (possibly from LLM)
    code = re.sub(r"<end>", "", code)

    # Remove explanation sections or error prompts at text beginning from LLM output
    lines = code.strip().splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("class mycluster") or line.strip().startswith("import") or line.strip().startswith(
                "from"):
            cleaned_lines.append(line)
        elif cleaned_lines:  # If code block recording has started, continue adding subsequent code
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def _instantiate_cluster_model(model_class, n_clusters: int):
    try:
        return model_class(n_clusters=n_clusters)
    except TypeError:
        try:
            return model_class(n_clusters)
        except TypeError:
            return model_class()


def _validate_cluster_labels(y_pred, n_samples: int):
    if y_pred is None:
        raise ValueError("y_pred is None")
    arr = np.asarray(y_pred)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != int(n_samples):
        raise ValueError(f"y_pred length mismatch: expected {int(n_samples)}, got {int(arr.shape[0])}")
    return arr


def summarize_dataset(df, sample_size=10):
    summary = {
        "n_samples": df.shape[0],
        "n_features": df.shape[1],
        "feature_types": {},
        "feature_ranges": {},
        "mean_std": {},
        "sample_head": df.head(sample_size).to_dict(orient='records')
    }

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary["feature_types"][col] = "numerical"
            summary["feature_ranges"][col] = [df[col].min(), df[col].max()]
            summary["mean_std"][col] = [df[col].mean(), df[col].std()]
        else:
            summary["feature_types"][col] = "categorical"
            summary["feature_ranges"][col] = list(df[col].unique())[:10]  # limit unique values for readability

    return summary


# Generate特征
def generate_feat(
        base_model,
        df,
        dataset_name,
        round_num,
        llm_model='gpt-3.5-turbo',
        iterations=10,
        target_column_name='class',
        dataset_description=None,
        task_type="clustering",
        base_url:str=None,api_key:str=None,
        logger=None
):
    if base_url is None or api_key is None:
        raise ValueError("base_url and api_key must be provided.")
    stage1_clf = Stage1Classifier(base_classifier=base_model,
                                llm_model=llm_model,
                                iterations=iterations)

    stage1_clf.fit_pandas(df,
                         target_column_name=target_column_name,
                         dataset_description=dataset_description,
                         dataset_name=dataset_name,
                         round_num=round_num,
                         task_type=task_type,
                         base_url=base_url,
                         api_key=api_key,
                         logger=logger
                        )

    df_aug = run_llm_code(stage1_clf.code, df, target_column_name)
    df_aug.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_aug.fillna(0, inplace=True)

    final_columns = [x for x in stage1_clf.final_columns if x != target_column_name]

    # Generate features
    X_aug = df_aug[final_columns]
    # Labels
    y_true = df_aug[target_column_name]
    if not pd.api.types.is_numeric_dtype(y_true):
        y_true = y_true.astype("category").cat.codes.astype(int)

    # ARI metric without standardization under kmeans
    y_pred_aug = base_model.fit_predict(X_aug)
    test_ari_aug = adjusted_rand_score(y_true, y_pred_aug) * 100

    # Standardization results
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
    X_scaled_df = pd.DataFrame(X_scaled, columns=final_columns)
    # ARI metric with standardization under kmeans
    y_pred_scaled = base_model.fit_predict(X_scaled_df)
    test_ari_scaled = adjusted_rand_score(y_true, y_pred_scaled) * 100
    # print(f'test_ari_aug:{test_ari_aug}')
    # print(f'test_ari_scaled:{test_ari_scaled}')

    # Select appropriate data preprocessing method: without standardization / with standardization
    if test_ari_aug > test_ari_scaled:
        return X_aug, y_true
    else:
        return X_scaled_df, y_true

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default="0", type=str)
    parser.add_argument('-s', '--default_seed', default=42, type=int)
    parser.add_argument('-l', '--llm', default='gpt-3.5-turbo', type=str)
    parser.add_argument('-e', '--exam_iterations', default=2, type=int)
    parser.add_argument('-f', '--feat_iterations', default=1, type=int)
    parser.add_argument('-m', '--model_iterations', default=5, type=int)
    parser.add_argument('-p', '--param_iterations', default=1, type=int)
    parser.add_argument('-d', '--dataset',default="breast",help="Dataset") # 'breast','glass','iris','students','seeds'
    parser.add_argument('--stage_timeout_s', default=STAGE_TIMEOUT_S, type=int, help='Stage timeout seconds')
    parser.add_argument('--min_ensemble_ari', default=0.30, type=float, help='Minimum cached single-model ARI required for multi-model consensus')
    parser.add_argument('--top_k', default=0, type=int, help='Fixed cached-label top-k for consensus; 0 keeps the original adaptive rule')
    parser.add_argument('--top_k_values', nargs='*', type=int, default=None, help='Evaluate several fixed cached-label top-k values in one run')
    parser.add_argument("--enable_optimization", action='store_true', default=True, 
                        help="Whether to enable hyperparameter optimization logic")
    parser.add_argument("--enable_feedback", action='store_true', default=True, 
                        help="Whether to enable LLM model generation feedback logic")
    args = parser.parse_args()
    STAGE_TIMEOUT_S = int(args.stage_timeout_s)


    """
    OpenAI API Configuration
    """
    env_url = os.getenv("OPENAI_BASE_URL")
    env_key = os.getenv("OPENAI_API_KEY")
    # --- Manual configuration ---
    manual_url = ""
    manual_key = "sk-"

    if env_url and env_key:
        base_url = env_url
        api_key = env_key
    elif manual_url and manual_key:
        base_url = manual_url
        api_key = manual_key
    else:
        raise ValueError(
            "No valid OpenAI API configuration found. "
            "Please set environment variables or manual config."
        )

    
    ds_name=args.dataset
    print(f"=========== Dataset {ds_name} ===========")
    logger = QuantLogger(
        task="clustering",
        dataset=ds_name,
        out_dir=os.path.join(project_root, "result", "logs"),
    )
    # Print hyperparameter optimization switch status
    print(f"Hyperparameter Optimization Status: {'Enabled' if args.enable_optimization else 'Disabled'}")
    # Print model generation feedback switch status
    print(f"Model Generation Feedback Status: {'Enabled' if args.enable_feedback else 'Disabled'}")

    loc = f"{project_root}/data/" + ds_name + ".pkl"

    # ===== 4种配置的结果存储（聚类Stacking）=====
    # 配置1：不蒸馏 + 不校准
    config1_ari, config1_nmi = [], []
    # 配置2：不蒸馏 + 校准
    config2_ari, config2_nmi = [], []
    # 配置3：蒸馏 + 校准
    config3_ari, config3_nmi = [], []
    # 配置4：蒸馏 + 不校准
    config4_ari, config4_nmi = [], []

    # Number of base models successfully added to ensemble per round
    models_per_experiment = []

    exam_iter = args.exam_iterations

    # ---------------------- Time & Token Statistics Initialization ----------------------
    all_time_start = time.time()
    feat_time_list = []  # Store feature generation time per round

    for exp in range(exam_iter):
        print(f"=========== Experiment {exp + 1}/{exam_iter} ===========")
        test_ari_list = []
        test_nmi_list = []
        seed = args.default_seed + exp
        random.seed(seed)
        np.random.seed(seed)

        # Load data
        df, n_clusters, target_column_name, dataset_description = load_origin_data(loc)
        baseline_model = base_model(n_clusters, seed)

        # ========== Generate Features ==========
        X_aug, y_true = generate_feat(
            base_model=baseline_model,
            df=df,
            dataset_name=ds_name,
            round_num=exp + 1,
            llm_model=args.llm,
            iterations=args.feat_iterations,
            target_column_name=target_column_name,
            dataset_description=dataset_description,
            task_type='clustering',
            base_url=base_url,
            api_key=api_key,
            logger=logger
        )
        logger.log(
            stage="feature_summary",
            r=exp + 1,
            task_type="clustering",
            feature_count=int(X_aug.shape[1]),
            base_model_count=0,
            meta_config={
                "feat_iterations": int(args.feat_iterations),
                "llm": args.llm,
            },
        )

        # Build initial samples and model prompts
        s = build_prompt_samples(X_aug)
        summary = summarize_dataset(X_aug)
        model_prompt = get_clustering_model_prompt(
            samples=s,
            n_clusters=n_clusters,
        )
        model_messages = [
            {
                "role": "system",
                "content": (
                    "You are a top-level clustering algorithm expert. "
                    "You help design clustering models optimized for ARI. "
                    "Your output must contain ONLY python code.\n\n"
                    "**IMPORTANT: You MUST set hyperparameters to utilize all available CPU cores for training to speed up the process.**\n\n"
                    "**STRICT SCALABILITY RULES FOR GENERATED CLUSTERING CODE:**\n"
                    "- Do NOT use pairwise_distances, pairwise_kernels, euclidean_distances, cosine_similarity, pdist, cdist, squareform, full distance matrices, precomputed affinities, co-association matrices, AgglomerativeClustering, SpectralClustering, MeanShift, DBSCAN, OPTICS, or AffinityPropagation.\n"
                    "- Do NOT construct any n_samples x n_samples matrix or use meta-classifiers over pairwise features.\n"
                    "- Use scalable candidates such as KMeans, MiniBatchKMeans, GaussianMixture, Birch, PCA/IncrementalPCA, StandardScaler/RobustScaler, and simple pipelines.\n\n"
                    "**CRITICAL: The sklearn version is >= 1.4 with important API changes:**\n"
                    "- Many ensemble/meta-learning classes changed parameter names from 'base_estimator' to 'estimator'\n"
                    "- Some classes changed attribute names and other APIs\n"
                    "- When encountering 'unexpected keyword argument' errors, consult the official sklearn 1.4+ documentation\n"
                ),
            },
            {"role": "user", "content": model_prompt},
        ]

        best_model_code = None
        best_model_ari = -1
        best_model_nmi = -1
        base_models = []
        base_label_records = []
        # Error retry mechanism
        max_retries_per_model = 3  # Maximum 3 retries per model
        current_model_retry_count = 0

        # ========== model_iterations ==========
        for i in range(args.model_iterations):
            logger.start_round(
                stage="model_iter",
                r=i + 1,
                exp=exp + 1,
                task_type="clustering",
                feature_count=int(X_aug.shape[1]),
                base_model_count=int(len(base_models)),
                meta_config={
                    "enable_optimization": bool(args.enable_optimization),
                    "enable_feedback": bool(args.enable_feedback),
                    "llm": args.llm,
                },
            )
            try:
                try:
                    with time_limit(STAGE_TIMEOUT_S):
                        code = generate_model(args.llm, model_messages, base_url, api_key)
                except TimeoutExceeded as te:
                    logger.event("timeout", 1)
                    logger.end_round(status="skipped", skip_reason="timeout", timeout_stage="llm_model_generate", error=str(te))
                    current_model_retry_count = 0
                    continue
                code = clean_llm_code(code)
            except Exception as e:
                print("Error in LLM API: " + str(e))
                logger.event("llm_api_error", 1)
                logger.end_round(status="llm_api_error", error_type=type(e).__name__, error=str(e))
                continue

            new_class_name = f"mycluster_{i + 1}"
            # Use regex to match any mycluster class name (including numbers that LLM may modify)
            code = re.sub(
                r'class\s+mycluster[_\w]*\s*(\([^)]*\))?\s*:', 
                f'class {new_class_name}:', 
                code,
                count=1  # Only replace first match
            )
            
            print(f"\n========== Generated Model Code (Round {i+1}) ==========")
            print(code)
            print("=" * 50)

            err, exec_scope = code_exec(code)
            
            # Check compilation error
            if err is not None:
                current_model_retry_count += 1
                logger.event("compile_error", 1)
                logger.event("retry", 1)
                print(f"Compilation error (retry {current_model_retry_count}/{max_retries_per_model}): {err}")
                
                if current_model_retry_count >= max_retries_per_model:
                    print(f"Maximum retries exceeded, skipping model {i+1}")
                    logger.end_round(
                        status="skipped",
                        skip_reason="compile_error",
                        retry_count=int(current_model_retry_count),
                        code_len=int(len(code)) if code is not None else None,
                    )
                    current_model_retry_count = 0  # Reset counter
                    i += 1  # Skip to next model
                    continue
                
                # Build error feedback
                error_hint = ""
                if "base_estimator" in err:
                    error_hint = "\n**HINT**: The parameter 'base_estimator' is not recognized in sklearn 1.4+. Check the documentation - some classes changed this parameter name. Consider alternatives in the official documentation."
                elif "got an unexpected keyword argument" in err:
                    error_hint = "\n**HINT**: An unexpected keyword argument was encountered. This might be due to sklearn version differences. Verify all parameter names in the official sklearn 1.4+ documentation."
                
                _append_retry_feedback(
                    model_messages,
                    code,
                    err,
                    f"Retry attempt {current_model_retry_count}/{max_retries_per_model}. Please fix the code and generate the next version:",
                    error_hint,
                )
                logger.end_round(
                    status="compile_error",
                    retry_count=int(current_model_retry_count),
                    code_len=int(len(code)) if code is not None else None,
                )
                continue

            try:
                model_class = exec_scope[new_class_name]
                model = _instantiate_cluster_model(model_class, n_clusters)
            except Exception as e:
                current_model_retry_count += 1
                logger.event("runtime_error", 1)
                logger.event("retry", 1)
                error_type = type(e).__name__
                error_msg = str(e)
                if isinstance(e, KeyError) and len(getattr(e, "args", [])) == 1 and str(e.args[0]) == new_class_name:
                    available_classes = [k for k, v in exec_scope.items() if isinstance(v, type)]
                    error_msg = f"missing_class={new_class_name}; available_classes={available_classes[:20]}"
                print(f"Model init failed: {error_type}: {error_msg}")
                print(f"Runtime error (retry {current_model_retry_count}/{max_retries_per_model})")
                
                if current_model_retry_count >= max_retries_per_model:
                    print(f"Maximum retries exceeded, skipping model {i+1}")
                    logger.end_round(
                        status="skipped",
                        skip_reason="runtime_error",
                        retry_count=int(current_model_retry_count),
                        last_error_type=error_type,
                        last_error=error_msg,
                    )
                    current_model_retry_count = 0  # Reset counter
                    i += 1  # Skip to next model
                    continue
                
                # Build detailed error feedback
                error_hint = ""
                if "base_estimator" in error_msg:
                    error_hint = "\n**HINT**: The parameter 'base_estimator' is not recognized in sklearn 1.4+. Check the documentation - some classes changed this parameter name. Consider alternatives in the official documentation."
                elif "got an unexpected keyword argument" in error_msg:
                    error_hint = "\n**HINT**: An unexpected keyword argument was encountered. This might be due to sklearn version differences. Verify all parameter names in the official sklearn 1.4+ documentation."
                elif "All intermediate steps should be transformers" in error_msg:
                    error_hint = "\n**HINT**: If you use sklearn.pipeline.Pipeline, every intermediate step must implement fit+transform (a transformer). Do not place estimators like IsolationForest/KMeans as intermediate steps; keep the clustering algorithm as the final step or avoid Pipeline."
                elif error_type == "KeyError" and "missing_class=" in error_msg:
                    error_hint = "\n**HINT**: You must define a class named `mycluster` (it will be renamed automatically). Output python code only."
                
                _append_retry_feedback(
                    model_messages,
                    code,
                    f"{error_type}: {error_msg}",
                    f"Retry attempt {current_model_retry_count}/{max_retries_per_model}. Please fix the code and generate the next version:",
                    error_hint,
                )
                logger.end_round(
                    status="runtime_error",
                    retry_count=int(current_model_retry_count),
                    last_error_type=error_type,
                    last_error=error_msg,
                )
                continue

            model_copy = copy.deepcopy(model)
            model_copy.name = "myCluster" + str(i + 1)

            try:
                try:
                    with time_limit(STAGE_TIMEOUT_S):
                        y_pred = model.fit_predict(X_aug)
                except TimeoutExceeded as te:
                    logger.event("timeout", 1)
                    logger.end_round(status="skipped", skip_reason="timeout", timeout_stage="base_model_fit_predict", error=str(te))
                    continue
                y_pred = _validate_cluster_labels(y_pred, X_aug.shape[0])
            except Exception as e:
                current_model_retry_count += 1
                logger.event("runtime_error", 1)
                logger.event("retry", 1)
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"Model predict failed: {error_type}: {error_msg}")
                print(f"Runtime error (retry {current_model_retry_count}/{max_retries_per_model})")
                
                if current_model_retry_count >= max_retries_per_model:
                    print(f"Maximum retries exceeded, skipping model {i+1}")
                    logger.end_round(
                        status="skipped",
                        skip_reason="runtime_error",
                        retry_count=int(current_model_retry_count),
                        last_error_type=error_type,
                        last_error=error_msg,
                    )
                    current_model_retry_count = 0  # Reset counter
                    i += 1  # Skip to next model
                    continue
                
                # Build detailed error feedback
                error_hint = ""
                if "base_estimator" in error_msg:
                    error_hint = "\n**HINT**: The parameter 'base_estimator' is not recognized in sklearn 1.4+. Check the documentation - some classes changed this parameter name. Consider alternatives in the official documentation."
                elif "got an unexpected keyword argument" in error_msg:
                    error_hint = "\n**HINT**: An unexpected keyword argument was encountered. This might be due to sklearn version differences. Verify all parameter names in the official sklearn 1.4+ documentation."
                elif "All intermediate steps should be transformers" in error_msg:
                    error_hint = "\n**HINT**: If you use sklearn.pipeline.Pipeline, every intermediate step must implement fit+transform (a transformer). Do not place estimators like IsolationForest/KMeans as intermediate steps; keep the clustering algorithm as the final step or avoid Pipeline."
                elif "has no attribute 'values'" in error_msg:
                    error_hint = "\n**HINT**: Do not assume X is a pandas DataFrame. Avoid using `X.values`; use `np.asarray(X)` instead."
                elif "Negative values in data passed to NMF" in error_msg:
                    error_hint = "\n**HINT**: NMF requires non-negative inputs. Apply MinMaxScaler or shift data to be >= 0 before NMF, or avoid NMF."
                
                _append_retry_feedback(
                    model_messages,
                    code,
                    f"{error_type}: {error_msg}",
                    f"Retry attempt {current_model_retry_count}/{max_retries_per_model}. Please fix the code and generate the next version:",
                    error_hint,
                )
                logger.end_round(
                    status="runtime_error",
                    retry_count=int(current_model_retry_count),
                    last_error_type=error_type,
                    last_error=error_msg,
                )
                continue

            # Model executed successfully, reset retry counter
            current_model_retry_count = 0

            try:
                test_ari = adjusted_rand_score(y_true, y_pred) * 100
                test_nmi = normalized_mutual_info_score(y_true, y_pred) * 100
            except Exception as e:
                current_model_retry_count += 1
                logger.event("runtime_error", 1)
                logger.event("retry", 1)
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"Metric computation failed: {error_type}: {error_msg}")
                logger.end_round(
                    status="runtime_error",
                    retry_count=int(current_model_retry_count),
                    last_error_type=error_type,
                    last_error=error_msg,
                )
                continue

            # ===========================
            # Hyperparameter Optimization Phase (switchable)
            # ===========================
            param_best_code = code
            param_best_ari = test_ari
            param_best_nmi = test_nmi
            param_best_labels = np.asarray(y_pred).copy()

            if args.enable_optimization:
                # Construct parameter optimization prompt
                final_cols = [x for x in X_aug.columns]
                param_prompt = get_param_prompt(param_best_code, param_best_ari,
                                                dataset_description, X_aug, final_cols)

                param_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a clustering optimization assistant. "
                            "Only optimize hyperparameters. Output python code only.\n\n"
                            "**IMPORTANT: Preserve or set hyperparameters to utilize all available CPU cores for training.**\n"
                            "**STRICT: Do NOT introduce pairwise distances, precomputed affinities, n_samples x n_samples matrices, AgglomerativeClustering, SpectralClustering, or pairwise meta-learning. Keep the model scalable.**"
                        ),
                    },
                    {"role": "user", "content": param_prompt},
                ]

                param_iter = args.param_iterations
                for p_iter in range(param_iter):
                    print(f"----- Param Opt {p_iter + 1}/{param_iter} -----")
                    try:
                        try:
                            with time_limit(STAGE_TIMEOUT_S):
                                param_code = generate_model(args.llm, param_messages, base_url, api_key)
                        except TimeoutExceeded:
                            logger.event("timeout", 1)
                            break
                        param_code = clean_llm_code(param_code)
                    except Exception as e:
                        print("LLM param error:", e)
                        break

                    param_new_class = f"{new_class_name}_param_{p_iter + 1}"
                    # Use regex to match any mycluster class name
                    param_code = re.sub(
                        r'class\s+mycluster[_\w]*\s*(\([^)]*\))?\s*:', 
                        f'class {param_new_class}:', 
                        param_code,
                        count=1
                    )
                    
                    print(f"\n========== Hyperparameter Optimization Code (Round {i+1}-Param Opt {p_iter+1}) ==========")
                    print(param_code)
                    print("=" * 50)

                    err, param_scope = code_exec(param_code)

                    # Handle compilation error
                    if err is not None:
                        print(f"Hyperparameter optimization code compilation error: {err}")
                        # Build detailed error feedback
                        error_hint = ""
                        if "base_estimator" in err:
                            error_hint = "\n**HINT**: The parameter 'base_estimator' is not recognized in sklearn 1.4+. Check the documentation - some classes changed this parameter name. Consider alternatives in the official documentation."
                        elif "got an unexpected keyword argument" in err:
                            error_hint = "\n**HINT**: An unexpected keyword argument was encountered. This might be due to sklearn version differences. Verify all parameter names in the official sklearn 1.4+ documentation."
                        
                        _append_retry_feedback(
                            param_messages,
                            param_code,
                            f"The optimized code execution failed with compilation error: {err}",
                            "Please fix the code and generate the next version:",
                            error_hint,
                        )
                        continue

                    try:
                        pc = param_scope[param_new_class]
                        param_model = _instantiate_cluster_model(pc, n_clusters)
                        try:
                            with time_limit(STAGE_TIMEOUT_S):
                                y_pred_param = param_model.fit_predict(X_aug)
                        except TimeoutExceeded:
                            logger.event("timeout", 1)
                            break
                        y_pred_param = _validate_cluster_labels(y_pred_param, X_aug.shape[0])
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        print(f"Param model failed: {error_type}: {error_msg}")
                        
                        # Build detailed error feedback
                        error_hint = ""
                        if "base_estimator" in error_msg:
                            error_hint = "\n**HINT**: The parameter 'base_estimator' is not recognized in sklearn 1.4+. Check the documentation - some classes changed this parameter name. Consider alternatives in the official documentation."
                        elif "got an unexpected keyword argument" in error_msg:
                            error_hint = "\n**HINT**: An unexpected keyword argument was encountered. This might be due to sklearn version differences. Verify all parameter names in the official sklearn 1.4+ documentation."
                        
                        _append_retry_feedback(
                            param_messages,
                            param_code,
                            f"{error_type}: {error_msg}",
                            "Please fix the code and generate the next version:",
                            error_hint,
                        )
                        continue

                    param_ari = adjusted_rand_score(y_true, y_pred_param) * 100
                    param_nmi = normalized_mutual_info_score(y_true, y_pred_param) * 100

                    if param_ari > param_best_ari:
                        param_best_ari = param_ari
                        param_best_nmi = param_nmi
                        param_best_code = param_code
                        param_best_labels = np.asarray(y_pred_param).copy()
                        model_copy = copy.deepcopy(param_model)
                        model_copy.name = "myCluster" + str(i + 1)

                    param_messages += [
                        {"role": "assistant", "content": param_code},
                        {
                            "role": "user",
                            "content": f"Current ARI: {param_ari:.2f}, Best: {param_best_ari:.2f}. Please improve further.",
                        },
                    ]

            # ========== End of One Model Round ==========

            base_models.append(model_copy)
            base_label_records.append(
                _cached_label_record(
                    getattr(model_copy, "name", f"teacher_{len(base_label_records) + 1}"),
                    param_best_labels,
                    param_best_ari / 100.0,
                    param_best_nmi / 100.0,
                    model=model_copy,
                    source="teacher",
                )
            )

            if param_best_ari > best_model_ari:
                best_model_ari = param_best_ari
                best_model_nmi = param_best_nmi
                best_model_code = param_best_code
            logger.end_round(
                status="success",
                val_metric_main="ari",
                val_score=float(param_best_ari) / 100.0,
                val_nmi=float(param_best_nmi) / 100.0,
                best_score_so_far=float(best_model_ari) / 100.0 if best_model_ari is not None else None,
                retry_count=int(current_model_retry_count),
                code_len=int(len(param_best_code)) if param_best_code is not None else None,
            )

            test_ari_list.append(param_best_ari)
            test_nmi_list.append(param_best_nmi)

            # Model feedback logic
            if args.enable_feedback:
                model_messages += [
                    {"role": "assistant", "content": param_best_code},
                    {
                        "role": "user",
                        "content": f"""
                        ✅ The clustering model executed successfully.
                        
                        📊 Current model ARI: {param_best_ari:.4f}
                        🏆 Best historical ARI so far: {best_model_ari:.4f}
                        
                        Please now propose a new clustering model that is **more likely to improve the ARI** on the given data.
                        The model must differ from all previous ones **by algorithm type or internal structure**.
                        
                        ⚠️ Remember:
                        - You must only output valid Python code for a complete clustering model named `mycluster`.
                        - The class must include all imports and implement: `fit` and `fit_predict`.
                        - Do not repeat models you've already used.
                        - Prioritize models that provide stable and efficient clustering.
                        - Do not use all-pairs/pairwise/full-distance/precomputed-affinity methods or O(n^2) constructs.
                        
                        🎯 Next code block:
                        """,
                    },
                ]
            else:
                model_messages += [
                    {"role": "assistant", "content": param_best_code},
                    {
                        "role": "user",
                        "content": """
                        ✅ The clustering model executed successfully.
                        
                        Please now propose a new clustering model that is **more likely to improve the ARI** on the given data.
                        The model must differ from all previous ones **by algorithm type or internal structure**.
                        
                        ⚠️ Remember:
                        - You must only output valid Python code for a complete clustering model named `mycluster`.
                        - The class must include all imports and implement: `fit` and `fit_predict`.
                        - Do not repeat models you've already used.
                        - Prioritize models that provide stable and efficient clustering.
                        - Do not use all-pairs/pairwise/full-distance/precomputed-affinity methods or O(n^2) constructs.
                        
                        🎯 Next code block:
                        """,
                    },
                ]

        # ========== End of Single Experiment ==========

        # Record the number of valid base models generated in current experiment
        valid_model_count = len(base_models)
        if valid_model_count == 0:
            fb = fallback_models(n_clusters=n_clusters, seed=seed)
            base_models.extend(fb)
            for fb_idx, fb_model in enumerate(fb, start=1):
                try:
                    base_label_records.append(
                        _evaluate_cached_record(
                            fb_model,
                            X_aug,
                            y_true,
                            getattr(fb_model, "name", f"fallback_{fb_idx}"),
                            source="teacher_fallback",
                        )
                    )
                except Exception:
                    continue
            valid_model_count = len(base_models)
            logger.log(
                stage="fallback_models",
                r=exp + 1,
                task_type="clustering",
                feature_count=int(X_aug.shape[1]),
                base_model_count=int(valid_model_count),
                meta_config={"reason": "no_valid_llm_models"},
            )
        models_per_experiment.append(valid_model_count)
        print(f"Round {exp + 1} valid base model count: {valid_model_count}")

        print(f"\n=== Experiment {exp + 1} Result ===")
        print("ARI:", test_ari_list)
        print("NMI:", test_nmi_list)

        # 数据分割：用一部分数据进行蒸馏和校准的训练
        from sklearn.model_selection import train_test_split
        X_ensemble, X_calib, y_ensemble, y_calib = train_test_split(
            X_aug, y_true, test_size=0.25, random_state=args.default_seed + exp
        )

        """
        Ensemble Learning - 4种配置对比（聚类 Stacking）
        """
        # 基础模型分类
        base_models_without_distillation = base_models  # 原始优化后的模型
        base_models_with_distillation = []  # 蒸馏后的模型列表
        base_label_records_without_distillation = list(base_label_records)

        # ========== 执行知识蒸馏 ==========
        print(f"\n【第 {exp + 1} 轮】开始知识蒸馏...")
        for idx, model in enumerate(base_models):
            print(f"  模型 {idx + 1} - 开始知识蒸馏到MLP学生网络...")
            try:
                device = 'cpu'
                try:
                    with time_limit(STAGE_TIMEOUT_S):
                        student_model = distill_to_student_clustering(
                            teacher_model=model,
                            X_train=X_ensemble,
                            y_train=y_ensemble,
                            X_val=X_calib,
                            y_val=y_calib,
                            device=device,
                            epochs=30,
                            n_clusters=None
                        )
                except TimeoutExceeded as te:
                    logger.event("timeout", 1)
                    logger.log(
                        stage="distill",
                        r=exp + 1,
                        task_type="clustering",
                        model_index=int(idx + 1),
                        status="skipped",
                        skip_reason="timeout",
                        timeout_stage="distill",
                        error=str(te),
                        decision={"distill_success": False},
                    )
                    base_models_with_distillation.append(model)
                    continue
                base_models_with_distillation.append(student_model)
                logger.log(
                    stage="distill",
                    r=exp + 1,
                    task_type="clustering",
                    model_index=int(idx + 1),
                    status="success",
                    decision={"distill_success": True},
                    distill_val_ari=getattr(student_model, "distill_val_ari", None),
                )
                print(f"  模型 {idx + 1} - 蒸馏成功，使用学生网络作为集成基模型")
            except Exception as distill_error:
                logger.log(
                    stage="distill",
                    r=exp + 1,
                    task_type="clustering",
                    model_index=int(idx + 1),
                    status="failed",
                    error_type=type(distill_error).__name__,
                    error=str(distill_error),
                    decision={"distill_success": False},
                )
                print(f"  模型 {idx + 1} - 蒸馏失败: {distill_error}")
                print(f"  模型 {idx + 1} - 回退使用原始优化模型")
                base_models_with_distillation.append(model)

        base_label_records_with_distillation = []
        for idx, model in enumerate(base_models_with_distillation):
            try:
                base_label_records_with_distillation.append(
                    _evaluate_cached_record(
                        model,
                        X_aug,
                        y_true,
                        getattr(model, "name", f"student_{idx + 1}"),
                        source="student",
                    )
                )
            except Exception:
                if idx < len(base_label_records_without_distillation):
                    rec = dict(base_label_records_without_distillation[idx])
                    rec["source"] = "student_eval_failed_teacher_fallback"
                    base_label_records_with_distillation.append(rec)

        # ========== 四种配置集成 ==========
        configurations = [
            {
                'name': '配置1: basemodel不蒸馏 + 不校准',
                'models': base_models_without_distillation,
                'label_records': base_label_records_without_distillation,
                'calibrate': False,
                'results': (config1_ari, config1_nmi)
            },
            {
                'name': '配置2: basemodel不蒸馏 + 校准',
                'models': base_models_without_distillation,
                'label_records': base_label_records_without_distillation,
                'calibrate': True,
                'results': (config2_ari, config2_nmi)
            },
            {
                'name': '配置3: basemodel蒸馏 + 校准',
                'models': base_models_with_distillation,
                'label_records': base_label_records_with_distillation,
                'calibrate': True,
                'results': (config3_ari, config3_nmi)
            },
            {
                'name': '配置4: basemodel蒸馏 + 不校准',
                'models': base_models_with_distillation,
                'label_records': base_label_records_with_distillation,
                'calibrate': False,
                'results': (config4_ari, config4_nmi)
            }
        ]

        if args.top_k_values:
            top_k_values = [int(v) for v in args.top_k_values if int(v) > 0]
        elif int(args.top_k) > 0:
            top_k_values = [int(args.top_k)]
        else:
            top_k_values = [0]

        for config in configurations:
            for requested_top_k in top_k_values:
                top_k_label = requested_top_k if requested_top_k > 0 else "adaptive"
                print(f"\n【第 {exp + 1} 轮】正在执行 {config['name']} top_k={top_k_label} ...")
                try:
                    result = None
                    try:
                        with time_limit(STAGE_TIMEOUT_S):
                            consensus_top_k = int(requested_top_k) if int(requested_top_k) > 0 else None
                            result = consensus_ensemble_cached_labels(
                                label_records=config["label_records"],
                                y_true=y_true,
                                n_clusters=len(np.unique(y_true)),
                                max_models=consensus_top_k if consensus_top_k is not None else 5,
                                min_models=consensus_top_k if consensus_top_k is not None else 2,
                                min_ari=args.min_ensemble_ari,
                            )
                            result.setdefault("ensemble_metrics", {})["requested_top_k"] = consensus_top_k
                            if config["calibrate"]:
                                result.setdefault("ensemble_metrics", {}).setdefault(
                                    "calibration",
                                    {
                                        "calibration_method": "cached_label_caps_topk",
                                        "selected_model_count": result.get("selected_model_count", None),
                                        "selected_model_ari": result.get("selected_model_ari", None),
                                        "eligible_model_count": result.get("eligible_model_count", None),
                                        "fallback_used": result.get("fallback_used", None),
                                    },
                                )
                    except TimeoutExceeded:
                        logger.event("timeout", 1)
                        continue
                    ensemble_metrics = result['ensemble_metrics']
                    calibration_info = ensemble_metrics.get("calibration", None) or {}
                    trust = None
                    trust_name = None
                    if config["calibrate"]:
                        trust_name = "confidence_mean"
                        trust = calibration_info.get("confidence_mean", None)
                        if trust is None and isinstance(result, dict) and "selected_model_count" in result:
                            trust_name = "selected_model_count"
                            trust = result.get("selected_model_count", None)
                    else:
                        trust_name = "base_pairwise_ari_mean"
                        trust = ensemble_metrics.get("base_pairwise_ari_mean", None)
                        if trust is None and isinstance(result, dict) and "selected_model_count" in result:
                            trust_name = "selected_model_count"
                            trust = result.get("selected_model_count", None)
                    logger.log(
                        stage="ensemble_eval",
                        r=exp + 1,
                        task_type="clustering",
                        feature_count=int(X_aug.shape[1]),
                        base_model_count=int(len(config["label_records"])),
                        meta_config={
                            "config_name": config["name"],
                            "calibrate": bool(config["calibrate"]),
                            "cached_label_mode": True,
                            "min_ensemble_ari": float(args.min_ensemble_ari),
                            "requested_top_k": consensus_top_k,
                            "selected_model_ari": result.get("selected_model_ari", None),
                            "eligible_model_count": result.get("eligible_model_count", None),
                            "fallback_used": result.get("fallback_used", None),
                        },
                        test_metric_main="ari",
                        test_score=float(ensemble_metrics.get("ensemble_ari", float("nan"))),
                        test_trust_metric=trust_name,
                        test_trust=trust,
                        nmi=float(ensemble_metrics.get("ensemble_nmi", float("nan"))),
                        calibration_check_triggered=1 if config["calibrate"] else 0,
                    )

                    print(f"\n=========== 第{exp + 1}次聚类Stacking集成结果--{config['name']} top_k={top_k_label} ===========")
                    print(f"ARI  : {ensemble_metrics['ensemble_ari']:.4f}")
                    print(f"NMI  : {ensemble_metrics['ensemble_nmi']:.4f}")
                    if config['calibrate']:
                        print(f"Calibration: {ensemble_metrics['calibration']}")

                    # Save results to corresponding lists
                    ari_list, nmi_list = config['results']
                    ari_list.append(round(ensemble_metrics['ensemble_ari'] * 100, 4))
                    nmi_list.append(round(ensemble_metrics['ensemble_nmi'] * 100, 4))

                except Exception as e:
                    print(f"集成失败: {e}")
                    continue

    print("\n========== Summary ==========")
    print(f"\n【配置1: 不蒸馏 + 不校准】")
    print('  ARI : ' + format_mean_std([v/100 for v in config1_ari]))
    print('  NMI : ' + format_mean_std([v/100 for v in config1_nmi]))

    print(f"\n【配置2: 不蒸馏 + 校准】")
    print('  ARI : ' + format_mean_std([v/100 for v in config2_ari]))
    print('  NMI : ' + format_mean_std([v/100 for v in config2_nmi]))

    print(f"\n【配置3: 蒸馏 + 校准】")
    print('  ARI : ' + format_mean_std([v/100 for v in config3_ari]))
    print('  NMI : ' + format_mean_std([v/100 for v in config3_nmi]))

    print(f"\n【配置4: 蒸馏 + 不校准】")
    print('  ARI : ' + format_mean_std([v/100 for v in config4_ari]))
    print('  NMI : ' + format_mean_std([v/100 for v in config4_nmi]))

    print(f"\n{'=' * 80}")

    print("\nValid base model count per round:")
    for idx, cnt in enumerate(models_per_experiment):
        print(f"Round {idx + 1}: {cnt} ")
    if models_per_experiment:
        avg_count = sum(models_per_experiment) / len(models_per_experiment)
        print(f"Average count per round: {avg_count:.2f}")
    else:
        avg_count = 0.0

    summary_content = f"""
=========== 数据集 {ds_name} ===========
超参数优化状态: {'已启用' if args.enable_optimization else '已禁用'}
模型生成反馈状态: {'已启用' if args.enable_feedback else '已禁用'}
综合性能统计报告 ({exam_iter} 轮实验)
=========================================
[聚类集成性能]
【配置1: 不蒸馏 + 不校准】
  ARI : {format_mean_std([v/100 for v in config1_ari])}
  NMI : {format_mean_std([v/100 for v in config1_nmi])}
  轮数: {len(config1_ari)}

【配置2: 不蒸馏 + 校准】
  ARI : {format_mean_std([v/100 for v in config2_ari])}
  NMI : {format_mean_std([v/100 for v in config2_nmi])}
  轮数: {len(config2_ari)}

【配置3: 蒸馏 + 校准】
  ARI : {format_mean_std([v/100 for v in config3_ari])}
  NMI : {format_mean_std([v/100 for v in config3_nmi])}
  轮数: {len(config3_ari)}

【配置4: 蒸馏 + 不校准】
  ARI : {format_mean_std([v/100 for v in config4_ari])}
  NMI : {format_mean_std([v/100 for v in config4_nmi])}
  轮数: {len(config4_ari)}
-----------------------------------------
[模型数量统计]
每轮有效基模型数量: {models_per_experiment}
每轮平均数量: {avg_count:.2f}
    """

    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = os.path.join(result_dir, f"{ds_name}.txt")
    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("=" * 80 + "\n")
            f.write(f"聚类集成性能对比统计 - {ds_name}\n")
            f.write("=" * 80 + "\n")
            f.write("=" * 80 + "\n\n")
            f.write(summary_content.strip() + "\n")
        print(f"\n结果已保存到: {result_file_path}")
    except Exception as e:
        print(f"\n保存结果文件失败: {e}")
    logger.close()
