from __future__ import annotations

import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(project_root)

import re
import time
import copy
import json
import random
import pickle
import argparse
import statistics
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from agent import AgenticReasoningAgent
from mystage1 import Stage1Classifier
from mystage1.run_llm_code import run_llm_code

from utils.model_generate import (
    build_prompt_samples,
    get_model_prompt_multi,
    generate_model_2,
)
from utils.CL_ensemble_utils import get_classification_param_prompt
from utils.ensembleUtils import getMetaModel_list
from utils.quant_log import QuantLogger
from utils.task_protocol import (
    log_feature_provenance,
    log_task_protocol,
    materialize_task_protocol,
    write_task_protocol,
)
from utils.time_limit import time_limit, TimeoutExceeded


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

STAGE_TIMEOUT_S = 300


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encode_non_numeric_features(df: pd.DataFrame, target_column_name: str) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in df.columns if c != target_column_name]
    for c in feature_cols:
        col = df[c]
        if pd.api.types.is_bool_dtype(col):
            df[c] = col.astype(int)
            continue
        if pd.api.types.is_numeric_dtype(col):
            continue
        col_str = col.astype("string").fillna("__missing__").str.strip()
        cat = pd.Categorical(col_str)
        df[c] = pd.Series(cat.codes, index=df.index, dtype="int32")
    return df


def _encode_non_numeric_features_train_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_column_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df_train.copy()
    df_test = df_test.copy()

    feature_cols = [c for c in df_train.columns if c != target_column_name]
    n_train = len(df_train)

    for c in feature_cols:
        train_col = df_train[c]

        if pd.api.types.is_bool_dtype(train_col):
            df_train[c] = train_col.fillna(False).astype(int)
            df_test[c] = df_test[c].fillna(False).astype(int)
            continue

        if pd.api.types.is_numeric_dtype(train_col):
            continue

        all_col = pd.concat([df_train[c], df_test[c]], ignore_index=True)
        all_str = all_col.astype("string").fillna("__missing__").str.strip()
        cat = pd.Categorical(all_str)
        codes = pd.Series(cat.codes, dtype="int32")
        df_train[c] = codes.iloc[:n_train].to_numpy()
        df_test[c] = codes.iloc[n_train:].to_numpy()

    return df_train, df_test


def load_origin_data(dataset_name: str, seed: int = 0):
    loc = os.path.join(project_root, "data", f"{dataset_name}.pkl")
    if os.path.exists(loc):
        with open(loc, "rb") as f:
            ds = pickle.load(f)
        data_df = ds[1]
        target_column_name = ds[4][-1]
        dataset_description = ds[-1]
    else:
        from utils.csv_dataset_loader import read_txt_file, resolve_description_path

        csv_base = os.path.join(project_root, "data", "csv_data")
        csv_loc = os.path.join(csv_base, f"{dataset_name}.csv")
        if not os.path.exists(csv_loc):
            raise FileNotFoundError(f"未找到数据集文件: {loc} 或 {csv_loc}")
        data_df = pd.read_csv(csv_loc).convert_dtypes()
        target_column_name = data_df.columns[-1]
        desc_path = resolve_description_path(csv_base, dataset_name)
        dataset_description = read_txt_file(desc_path, encoding="utf-8") if desc_path else ""

    data_df = data_df.dropna(subset=[target_column_name]).copy()
    try:
        df_train, df_test = train_test_split(
            data_df,
            test_size=0.25,
            random_state=seed,
            stratify=data_df[target_column_name],
        )
    except ValueError:
        df_train, df_test = train_test_split(
            data_df,
            test_size=0.25,
            random_state=seed,
            shuffle=True,
        )
    df_train.replace([float("inf"), float("-inf")], 0, inplace=True)
    df_test.replace([float("inf"), float("-inf")], 0, inplace=True)

    y_all = pd.concat([df_train[target_column_name], df_test[target_column_name]], ignore_index=True)
    le = LabelEncoder()
    le.fit(y_all.astype("string").fillna("__missing__"))
    df_train[target_column_name] = le.transform(df_train[target_column_name].astype("string").fillna("__missing__")).astype(int)
    df_test[target_column_name] = le.transform(df_test[target_column_name].astype("string").fillna("__missing__")).astype(int)
    df_train, df_test = _encode_non_numeric_features_train_test(df_train, df_test, target_column_name)

    feature_cols = [c for c in df_train.columns if c != target_column_name]
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df_train[c]):
            continue
        tr_num = pd.to_numeric(df_train[c], errors="coerce")
        med = float(tr_num.median()) if tr_num.size else 0.0
        if not np.isfinite(med):
            med = 0.0
        df_train[c] = pd.to_numeric(df_train[c], errors="coerce").fillna(med)
        df_test[c] = pd.to_numeric(df_test[c], errors="coerce").fillna(med)

    num_classes = int(len(le.classes_))
    return df_train, df_test, target_column_name, dataset_description, le, num_classes


def base_model(seed, n_jobs=-1):
    rforest = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        class_weight="balanced",
        n_jobs=n_jobs,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    param_grid = {"min_samples_leaf": [0.001, 0.01, 0.05], "max_depth": [5, 10, None]}
    gsmodel = GridSearchCV(rforest, param_grid, cv=cv, scoring="f1_weighted", n_jobs=n_jobs)
    return gsmodel


def generate_feat(
    base_classifier,
    df_train,
    df_test,
    dataset_name,
    round_num=0,
    llm_model="gpt-3.5-turbo",
    iterations=10,
    target_column_name="class",
    dataset_description=None,
    task_type="classification",
    base_url=None,
    api_key=None,
    logger=None,
):
    if base_url is None or api_key is None:
        raise ValueError("base_url and api_key must be provided.")
    input_columns = list(df_train.columns)
    stage1_clf = Stage1Classifier(
        base_classifier=base_classifier,
        llm_model=llm_model,
        iterations=iterations,
    )
    stage1_clf.fit_pandas(
        df_train,
        target_column_name=target_column_name,
        dataset_description=dataset_description,
        dataset_name=dataset_name,
        round_num=round_num,
        task_type=task_type,
        base_url=base_url,
        api_key=api_key,
        logger=logger,
    )
    df_train_aug = run_llm_code(stage1_clf.code, df_train, target_column_name)
    df_test_aug = run_llm_code(stage1_clf.code, df_test, target_column_name)
    df_train_aug = df_train_aug[stage1_clf.final_columns]
    df_test_aug = df_test_aug[stage1_clf.final_columns]
    if logger is not None:
        log_feature_provenance(
            logger,
            round_num=round_num,
            input_columns=input_columns,
            output_columns=df_train_aug.columns,
            target_column=target_column_name,
            generated_code=stage1_clf.code,
        )
    return df_train_aug, df_test_aug


def to_pd(df, target_name):
    y = df[target_name].astype(int)
    x = df.drop(target_name, axis=1)
    return x, y


def format_mean_std(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f}±{std_dev:.2f}"


def clean_llm_code(code: str) -> str:
    code = re.sub(r"^```python\s*", "", code.strip(), flags=re.IGNORECASE)
    code = re.sub(r"```$", "", code.strip())
    code = re.sub(r"<end>", "", code)
    lines = code.strip().splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("class myclassifier") or line.strip().startswith("import") or line.strip().startswith("from"):
            cleaned_lines.append(line)
        elif cleaned_lines:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def code_exec(code: str):
    try:
        compiled_code = compile(code, "<string>", "exec")
        scope = {"__builtins__": __builtins__}
        exec(compiled_code, scope, scope)
        return None, scope
    except Exception as e:
        return str(e), None


def extract_json_from_text(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except json.JSONDecodeError:
        return None


def safe_multiclass_auc(y_true, proba):
    try:
        y_true = np.asarray(y_true)
        proba = np.asarray(proba)
        if len(np.unique(y_true)) <= 1:
            return 0.0
        if proba.ndim != 2 or proba.shape[1] <= 1:
            return 0.0
        return roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
    except Exception:
        return 0.0


def calculate_multiclass_metrics(y_true, proba):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = np.argmax(proba, axis=1) if proba.ndim == 2 else np.asarray(proba).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "roc_auc": safe_multiclass_auc(y_true, proba),
    }


def multiclass_brier(y_true, proba):
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    n = len(y_true)
    if n == 0 or proba.ndim != 2:
        return None
    k = proba.shape[1]
    oh = np.zeros((n, k), dtype=float)
    oh[np.arange(n), np.clip(y_true, 0, k - 1)] = 1.0
    return float(np.mean(np.sum((proba - oh) ** 2, axis=1)))


def _align_proba_to_classes(model, proba, classes):
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes)
    if proba.ndim == 1:
        if len(classes) == 2:
            proba = np.stack([1 - proba, proba], axis=1)
        else:
            raise ValueError("predict_proba returned 1D but n_classes>2")
    if proba.shape[1] == len(classes):
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums
    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        raise ValueError("Model does not expose classes_ for alignment.")
    model_classes = np.asarray(model_classes)
    aligned = np.full((proba.shape[0], len(classes)), 1.0 / len(classes), dtype=float)
    for j, c in enumerate(classes):
        idx = np.where(model_classes == c)[0]
        if idx.size > 0 and int(idx[0]) < proba.shape[1]:
            aligned[:, j] = proba[:, int(idx[0])]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return aligned / row_sums


class StudentNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(d, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

        self._distill_columns = None
        self._distill_scaler = None
        self._distill_num_classes = int(num_classes)
        self.oof_refit_capable = False
        self.route_role = "prefit_student"
        self.fit_protocol = "distilled_once_then_reused"

    def forward(self, x):
        return self.net(x)

    def fit(self, X, y):
        return self

    def _transform_X(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        if isinstance(X_df, pd.DataFrame):
            X_df = X_df.copy()
        base = pd.get_dummies(X_df, dummy_na=True).fillna(0)
        if self._distill_columns is None:
            self._distill_columns = base.columns
        X_proc = pd.get_dummies(X_df, dummy_na=True).reindex(columns=self._distill_columns, fill_value=0)
        if self._distill_scaler is None:
            self._distill_scaler = StandardScaler()
            X_scaled = self._distill_scaler.fit_transform(X_proc)
        else:
            X_scaled = self._distill_scaler.transform(X_proc)
        return X_scaled.astype(np.float32)

    def predict_proba(self, X):
        self.eval()
        device = next(self.parameters()).device
        X_scaled = self._transform_X(X)
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            logits = self(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def kd_loss(student_logits, teacher_logits, T=3.0):
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def _preprocess_for_distillation_X(train_X: pd.DataFrame, val_X: pd.DataFrame):
    base = pd.concat([train_X, val_X], axis=0, ignore_index=False)
    base = pd.get_dummies(base, dummy_na=True).fillna(0)
    columns = base.columns
    tr = pd.get_dummies(train_X, dummy_na=True).reindex(columns=columns, fill_value=0)
    va = pd.get_dummies(val_X, dummy_na=True).reindex(columns=columns, fill_value=0)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(tr)
    X_val_scaled = scaler.transform(va)
    return X_tr_scaled.astype(np.float32), X_val_scaled.astype(np.float32), columns, scaler


def _teacher_logits_multiclass(teacher_model, X_df: pd.DataFrame, classes):
    proba = teacher_model.predict_proba(X_df)
    proba = _align_proba_to_classes(teacher_model, proba, classes)
    proba = np.clip(proba, 1e-6, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return np.log(proba).astype(np.float32)


def distill_to_student_multiclass(
    teacher_model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_classes: int,
    device: str = "cpu",
    epochs: int = 30,
):
    X_tr_scaled, X_val_scaled, columns, scaler = _preprocess_for_distillation_X(X_train, X_val)
    classes = np.arange(num_classes, dtype=int)
    train_logits = _teacher_logits_multiclass(teacher_model, X_train, classes)
    val_logits = _teacher_logits_multiclass(teacher_model, X_val, classes)

    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(np.asarray(y_train, dtype=np.int64), dtype=torch.long),
        torch.tensor(train_logits, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(np.asarray(y_val, dtype=np.int64), dtype=torch.long),
        torch.tensor(val_logits, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    student = StudentNet(input_dim=X_tr_scaled.shape[1], num_classes=num_classes, hidden_dims=(256, 128), dropout=0.3).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
    T = 3.0
    lambda_kd = 0.5
    best_auc = -1.0
    best_state = None

    for _ in range(epochs):
        student.train()
        for xb, yb, tb in train_loader:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            optimizer.zero_grad()
            logits = student(xb)
            loss_ce = F.cross_entropy(logits, yb)
            loss_kd = kd_loss(logits, tb, T=T)
            loss = (1 - lambda_kd) * loss_ce + lambda_kd * loss_kd
            loss.backward()
            optimizer.step()

        student.eval()
        probs_list, targets_list = [], []
        with torch.no_grad():
            for xb, yb, _tb in val_loader:
                xb = xb.to(device)
                logits = student(xb)
                probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
                targets_list.append(yb.numpy())
        probs = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0, num_classes))
        targets = np.concatenate(targets_list, axis=0) if targets_list else np.zeros((0,), dtype=int)
        auc = safe_multiclass_auc(targets, probs)
        if auc > best_auc:
            best_auc = auc
            best_state = copy.deepcopy(student.state_dict())

    if best_state is not None:
        student.load_state_dict(best_state)

    student._distill_columns = columns
    student._distill_scaler = scaler
    student.distill_val_auc = float(best_auc)
    student.distill_method = "kl"
    return student


class MulticlassTemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, y_true):
        logits = np.asarray(logits, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.int64)

        def nll(logT):
            T = float(np.exp(np.asarray(logT).reshape(-1)[0]))
            scaled = logits / T
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            expv = np.exp(scaled)
            prob = expv / expv.sum(axis=1, keepdims=True)
            prob = np.clip(prob, 1e-12, 1.0)
            return -np.log(prob[np.arange(prob.shape[0]), y_true]).mean()

        res = minimize_nll(nll)
        self.temperature = float(np.exp(res))
        return self

    def transform_proba(self, proba):
        proba = np.asarray(proba, dtype=np.float64)
        proba = np.clip(proba, 1e-12, 1.0)
        logits = np.log(proba)
        scaled = logits / float(self.temperature)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        expv = np.exp(scaled)
        out = expv / expv.sum(axis=1, keepdims=True)
        return out.astype(np.float64)


def minimize_nll(nll_fn):
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    def wrapper(logT):
        return nll_fn(logT)

    res = minimize(wrapper, x0=np.array([0.0], dtype=np.float64), method="L-BFGS-B")
    if not res.success:
        return 0.0
    return float(res.x.reshape(-1)[0])


def _fresh_model(base, seed=42):
    import torch.nn as tnn
    if isinstance(base, tnn.Module):
        return copy.deepcopy(base)
    m = copy.deepcopy(base)
    if hasattr(m, "set_params"):
        try:
            m.set_params(random_state=seed)
        except Exception:
            pass
    return m


def stacking_ensemble_multiclass_v2(
    base_models,
    meta_model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    n_folds: int = 5,
    calibrate_proba: bool = False,
    random_state: int = 42,
):
    y_train_np = np.asarray(y_train, dtype=int)
    classes = np.unique(y_train_np)
    n_classes = len(classes)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    meta_train_parts = []
    for m_idx, base in enumerate(base_models):
        oof = np.zeros((len(y_train_np), n_classes), dtype=float)
        skip = False
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train_np)):
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            seed = (random_state + 1009 * (m_idx + 1) + 31 * (fold_idx + 1)) & 0x7fffffff
            model_f = _fresh_model(base, seed=seed)
            model_f.fit(X_tr, y_tr)
            try:
                proba_va = model_f.predict_proba(X_va)
            except Exception:
                skip = True
                break
            oof[va_idx, :] = _align_proba_to_classes(model_f, proba_va, classes)
        if skip:
            continue
        meta_train_parts.append(oof)

    if not meta_train_parts:
        raise ValueError("No usable base models for multiclass stacking.")

    meta_X_train = np.concatenate(meta_train_parts, axis=1)
    meta_model_f = copy.deepcopy(meta_model)
    meta_model_f.fit(meta_X_train, y_train_np)

    meta_test_parts = []
    meta_val_parts = []
    for m_idx, base in enumerate(base_models):
        model_full = _fresh_model(base, seed=(random_state + 2027 * (m_idx + 1)) & 0x7fffffff)
        model_full.fit(X_train, y_train_np)
        proba_te = model_full.predict_proba(X_test)
        meta_test_parts.append(_align_proba_to_classes(model_full, proba_te, classes))
        if X_val is not None:
            proba_va = model_full.predict_proba(X_val)
            meta_val_parts.append(_align_proba_to_classes(model_full, proba_va, classes))

    meta_X_test = np.concatenate(meta_test_parts, axis=1)
    proba_test = meta_model_f.predict_proba(meta_X_test)
    proba_test = _align_proba_to_classes(meta_model_f, proba_test, classes)

    proba_val = None
    if X_val is not None:
        meta_X_val = np.concatenate(meta_val_parts, axis=1)
        proba_val = meta_model_f.predict_proba(meta_X_val)
        proba_val = _align_proba_to_classes(meta_model_f, proba_val, classes)

    calibration_applied = False
    if calibrate_proba and proba_val is not None and y_val is not None:
        y_val_np = np.asarray(y_val, dtype=int)
        in_train_mask = np.isin(y_val_np, classes)
        if np.any(in_train_mask):
            y_val_mapped = np.searchsorted(classes, y_val_np[in_train_mask])
            scaler = MulticlassTemperatureScaler()
            scaler.fit(np.log(np.clip(proba_val[in_train_mask], 1e-12, 1.0)), y_val_mapped)
            proba_test = scaler.transform_proba(proba_test)
            calibration_applied = True

    metrics = calculate_multiclass_metrics(y_test, proba_test)
    metrics["brier"] = multiclass_brier(y_test, proba_test)

    return {
        "meta_model": meta_model_f,
        "probas": proba_test,
        "metrics": metrics,
        "calibration_applied": calibration_applied,
        "calibration_fit_source": "validation_temperature_scaling" if calibration_applied else None,
    }


def voting_ensemble_multiclass_fallback(
    base_models,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    calibrate_proba: bool = False,
    random_state: int = 42,
    max_models: int = 3,
):
    y_train_np = np.asarray(y_train, dtype=int)
    classes = np.unique(y_train_np)
    n_classes = len(classes)

    kept = []
    for m_idx, base in enumerate(base_models[: max_models if max_models and max_models > 0 else len(base_models)]):
        model_full = _fresh_model(base, seed=(random_state + 2027 * (m_idx + 1)) & 0x7fffffff)
        try:
            proba_te = model_full.predict_proba(X_test)
        except Exception:
            try:
                model_full.fit(X_train, y_train_np)
                proba_te = model_full.predict_proba(X_test)
            except Exception:
                continue
        kept.append((model_full, proba_te))

    if not kept:
        raise ValueError("No usable base models for multiclass voting fallback.")

    probas_test = []
    probas_val = []
    for model_full, proba_te in kept:
        probas_test.append(_align_proba_to_classes(model_full, proba_te, classes))
        if X_val is not None:
            try:
                proba_va = model_full.predict_proba(X_val)
                probas_val.append(_align_proba_to_classes(model_full, proba_va, classes))
            except Exception:
                pass

    proba_test = np.mean(np.stack(probas_test, axis=0), axis=0)

    calibration_applied = False
    if calibrate_proba and X_val is not None and y_val is not None and probas_val:
        proba_val = np.mean(np.stack(probas_val, axis=0), axis=0)
        y_val_np = np.asarray(y_val, dtype=int)
        in_train_mask = np.isin(y_val_np, classes)
        if np.any(in_train_mask):
            y_val_mapped = np.searchsorted(classes, y_val_np[in_train_mask])
            scaler = MulticlassTemperatureScaler()
            scaler.fit(np.log(np.clip(proba_val[in_train_mask], 1e-12, 1.0)), y_val_mapped)
            proba_test = scaler.transform_proba(proba_test)
            calibration_applied = True

    metrics = calculate_multiclass_metrics(y_test, proba_test)
    metrics["brier"] = multiclass_brier(y_test, proba_test)
    metrics["fallback_used"] = True
    metrics["fallback_kept_models"] = len(kept)
    metrics["fallback_max_models"] = int(max_models)

    return {
        "meta_model": None,
        "probas": proba_test,
        "metrics": metrics,
        "calibration_applied": calibration_applied,
        "calibration_fit_source": "validation_temperature_scaling" if calibration_applied else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", default="0", type=str, help="GPU settings")
    parser.add_argument("-s", "--default_seed", default=42, type=int, help="Random seed")
    parser.add_argument("-l", "--llm", default="gpt-3.5-turbo", type=str, help="Large language model")
    parser.add_argument("-e", "--exam_iterations", default=3, type=int, help="Number of experiments")
    parser.add_argument("-f", "--feat_iterations", default=10, type=int, help="Feature iteration count")
    parser.add_argument("-m", "--model_iterations", default=5, type=int, help="Model iteration count")
    parser.add_argument("-p", "--param_iterations", default=5, type=int, help="Parameter tuning iterations")
    parser.add_argument("-d", "--dataset", default="iris", help="Dataset")
    parser.add_argument("--enable_optimization", action="store_true", default=True)
    parser.add_argument("--enable_feedback", action="store_true", default=True)
    parser.add_argument("--stage_timeout_s", default=STAGE_TIMEOUT_S, type=int, help="Stage timeout seconds")
    parser.add_argument("--task-spec", default=None, help="Optional JSON task specification")
    args = parser.parse_args()

    task_spec, plan_bundle = materialize_task_protocol(
        "multiclassification",
        args.dataset,
        args,
        task_spec_path=args.task_spec,
    )

    STAGE_TIMEOUT_S = int(args.stage_timeout_s)

    need_llm = (args.feat_iterations > 0) or (args.model_iterations > 0)
    base_url = os.getenv("OPENAI_BASE_URL") if need_llm else None
    api_key = os.getenv("OPENAI_API_KEY") if need_llm else None
    if need_llm and not (base_url and api_key):
        raise ValueError("未检测到 OPENAI_BASE_URL / OPENAI_API_KEY 环境变量，但当前参数会触发 LLM 调用。")

    ds_name = args.dataset
    print(f"=========== Dataset {ds_name} (multiclass) ===========")
    logger = QuantLogger(
        task="multiclassification",
        dataset=ds_name,
        out_dir=os.path.join(project_root, "result", "logs"),
    )
    agent = AgenticReasoningAgent(task_spec, plan_bundle)
    logger.bind_agent(agent)
    write_task_protocol(
        os.path.splitext(logger.path)[0] + ".protocol.json",
        task_spec,
        plan_bundle,
        agent=agent,
    )
    log_task_protocol(logger, task_spec, plan_bundle, agent=agent)

    config1_acc, config1_f1, config1_auc, config1_pre, config1_rec = [], [], [], [], []
    config2_acc, config2_f1, config2_auc, config2_pre, config2_rec = [], [], [], [], []
    config3_acc, config3_f1, config3_auc, config3_pre, config3_rec = [], [], [], [], []
    config4_acc, config4_f1, config4_auc, config4_pre, config4_rec = [], [], [], [], []

    all_time_start = time.time()
    feat_time_list = []
    token_model_list = []
    models_per_experiment = []
    total_llm_time = 0.0

    for exp in range(args.exam_iterations):
        seed = args.default_seed + exp * 10
        seed_everything(seed)
        print(f"=========== Experiment {exp + 1}/{args.exam_iterations} ===========")

        df_train_aug, df_test_aug, target_column_name, dataset_description, label_encoder, num_classes = load_origin_data(ds_name, seed=seed)
        baseline_model = base_model(seed)

        if args.feat_iterations > 0:
            feat_start = time.time()
            df_train_aug, df_test_aug = generate_feat(
                base_classifier=baseline_model,
                df_train=df_train_aug,
                df_test=df_test_aug,
                dataset_name=ds_name,
                round_num=exp + 1,
                llm_model=args.llm,
                iterations=args.feat_iterations,
                target_column_name=target_column_name,
                dataset_description=dataset_description,
                base_url=base_url,
                api_key=api_key,
                logger=logger,
            )
            feat_time_list.append(time.time() - feat_start)
        else:
            feat_time_list.append(0.0)

        try:
            df_train_aug, df_valid_aug = train_test_split(
                df_train_aug,
                test_size=0.25,
                random_state=seed,
                stratify=df_train_aug[target_column_name],
            )
        except ValueError:
            df_train_aug, df_valid_aug = train_test_split(
                df_train_aug,
                test_size=0.25,
                random_state=seed,
                shuffle=True,
            )

        train_aug_x, train_aug_y = to_pd(df_train_aug, target_column_name)
        val_aug_x, val_aug_y = to_pd(df_valid_aug, target_column_name)
        test_aug_x, test_aug_y = to_pd(df_test_aug, target_column_name)

        s = build_prompt_samples(df_train_aug)
        model_prompt = get_model_prompt_multi(
            target_column_name=target_column_name,
            samples=s,
        )
        model_messages = [
            {
                "role": "system",
                "content": (
                    "You are a top-level machine learning multi-class classification expert.\n"
                    "Your primary goal is to maximize the multi-class AUC (OVR, weighted) on the validation set.\n"
                    "Your answer should only generate valid Python code.\n"
                    "You MUST set hyperparameters to utilize all available CPU cores (e.g., n_jobs=-1).\n"
                ),
            },
            {"role": "user", "content": model_prompt},
        ]

        base_models_without_distillation = []
        base_models_with_distillation = []
        total_token_model_exp = 0

        if args.model_iterations <= 0:
            bm = copy.deepcopy(baseline_model)
            bm.fit(train_aug_x, train_aug_y)
            base_models_without_distillation.append(bm)
            base_models_with_distillation.append(bm)
        else:
            best_auc = 0.0
            i = 0
            max_retries_per_model = 3
            current_model_retry_count = 0

            while i < args.model_iterations:
                if not logger.in_round():
                    logger.start_round(
                        stage="model_iter",
                        r=i + 1,
                        exp=exp + 1,
                        task_type="multiclassification",
                        feature_count=int(train_aug_x.shape[1]),
                        base_model_count=int(len(base_models_without_distillation)),
                        meta_config={
                            "enable_optimization": bool(args.enable_optimization),
                            "enable_feedback": bool(args.enable_feedback),
                            "llm": args.llm,
                            "num_classes": int(num_classes),
                        },
                    )

                try:
                    model_llm_start = time.time()
                    try:
                        with time_limit(STAGE_TIMEOUT_S):
                            res = generate_model_2(args.llm, model_messages, base_url, api_key)
                    except TimeoutExceeded as te:
                        logger.event("timeout", 1)
                        logger.end_round(status="skipped", skip_reason="timeout", timeout_stage="llm_model_generate", error=str(te))
                        current_model_retry_count = 0
                        i += 1
                        continue
                    code = clean_llm_code(res["code"])
                    total_token_model_exp += int(res.get("total_tokens", 0) or 0)
                    total_llm_time += (time.time() - model_llm_start)

                    new_class_name = f"myclassifier_{i + 1}"
                    code = re.sub(r"class\s+myclassifier[_\w]*\s*(\([^)]*\))?\s*:", f"class {new_class_name}:", code, count=1)
                except Exception as e:
                    logger.event("llm_api_error", 1)
                    print("Error in LLM API." + str(e))
                    continue

                e, exec_scope = code_exec(code)
                if e is not None:
                    current_model_retry_count += 1
                    logger.event("compile_error", 1)
                    logger.event("retry", 1)
                    if current_model_retry_count >= max_retries_per_model:
                        logger.end_round(status="skipped", skip_reason="compile_error", retry_count=int(current_model_retry_count))
                        current_model_retry_count = 0
                        i += 1
                        continue
                    model_messages += [
                        {"role": "assistant", "content": code},
                        {"role": "user", "content": f"The classifier code failed with error: {e}\nCode: ```python{code}```"},
                    ]
                    continue

                try:
                    model_class = exec_scope[new_class_name]
                    model = model_class()
                    try:
                        with time_limit(STAGE_TIMEOUT_S):
                            model.fit(train_aug_x, train_aug_y)
                            model_copy = copy.deepcopy(model)
                            proba = model.predict_proba(val_aug_x)
                    except TimeoutExceeded as te:
                        logger.event("timeout", 1)
                        logger.end_round(status="skipped", skip_reason="timeout", timeout_stage="base_model_fit_predict", error=str(te))
                        current_model_retry_count = 0
                        i += 1
                        continue
                    model_list_append = model_copy
                except Exception as e2:
                    current_model_retry_count += 1
                    logger.event("runtime_error", 1)
                    logger.event("retry", 1)
                    if current_model_retry_count >= max_retries_per_model:
                        logger.end_round(status="skipped", skip_reason="runtime_error", retry_count=int(current_model_retry_count))
                        current_model_retry_count = 0
                        i += 1
                        continue
                    model_messages += [
                        {"role": "assistant", "content": code},
                        {"role": "user", "content": f"Code execution failed: {type(e2).__name__}: {e2}\nCode: ```python{code}```"},
                    ]
                    continue

                proba = _align_proba_to_classes(model_list_append, proba, np.arange(num_classes, dtype=int))
                val_auc = safe_multiclass_auc(val_aug_y, proba) * 100.0
                val_brier = multiclass_brier(val_aug_y, proba)

                current_model_retry_count = 0

                param_best_code = code
                param_best_auc = val_auc

                if args.enable_optimization and args.param_iterations > 0:
                    param_prompt = get_classification_param_prompt(
                        best_code=param_best_code,
                        best_auc=param_best_auc,
                        dataset_description=dataset_description,
                        X_test=val_aug_x,
                        feature_columns=train_aug_x.columns.tolist(),
                        dataset_name=ds_name,
                        max_rows=10,
                    )
                    param_messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a classification optimization assistant.\n"
                                "Tune hyperparameters only. Output executable Python code only.\n"
                                "Preserve or add n_jobs=-1 (or thread_count=-1 for CatBoost).\n"
                            ),
                        },
                        {"role": "user", "content": param_prompt},
                    ]

                    for p_iter in range(args.param_iterations):
                        try:
                            param_llm_start = time.time()
                            try:
                                with time_limit(STAGE_TIMEOUT_S):
                                    res2 = generate_model_2(args.llm, param_messages, base_url, api_key)
                            except TimeoutExceeded:
                                logger.event("timeout", 1)
                                break
                            param_code = clean_llm_code(res2["code"])
                            total_token_model_exp += int(res2.get("total_tokens", 0) or 0)
                            total_llm_time += (time.time() - param_llm_start)

                            param_new_class_name = f"myclassifier_{i+1}_param_{p_iter + 1}"
                            param_code = re.sub(
                                r"class\s+myclassifier[_\w]*\s*(\([^)]*\))?\s*:",
                                f"class {param_new_class_name}:",
                                param_code,
                                count=1,
                            )
                            param_err, param_scope = code_exec(param_code)
                            if param_err is not None:
                                param_messages += [
                                    {"role": "assistant", "content": param_code},
                                    {"role": "user", "content": f"Compilation error: {param_err}\nCode: ```python{param_code}```"},
                                ]
                                continue

                            model_class2 = param_scope[param_new_class_name]
                            tuned = model_class2()
                            try:
                                _t0 = time.time()
                                with time_limit(STAGE_TIMEOUT_S):
                                    tuned.fit(train_aug_x, train_aug_y)
                                    tuned_copy = copy.deepcopy(tuned)
                                    proba_tuned = tuned.predict_proba(val_aug_x)
                            except TimeoutExceeded:
                                logger.event("timeout", 1)
                                break
                            except KeyboardInterrupt:
                                if (time.time() - _t0) >= (STAGE_TIMEOUT_S - 1.0):
                                    logger.event("timeout", 1)
                                    break
                                raise

                            proba_tuned = _align_proba_to_classes(tuned_copy, proba_tuned, np.arange(num_classes, dtype=int))
                            auc_tuned = safe_multiclass_auc(val_aug_y, proba_tuned) * 100.0
                            if auc_tuned > param_best_auc:
                                param_best_auc = auc_tuned
                                param_best_code = param_code
                                model_list_append = tuned_copy

                            param_messages += [
                                {"role": "assistant", "content": param_code},
                                {"role": "user", "content": f"Current AUC: {auc_tuned:.2f}, Best AUC: {param_best_auc:.2f}. Improve further."},
                            ]
                        except Exception:
                            continue

                if best_auc < param_best_auc:
                    best_auc = param_best_auc

                base_models_without_distillation.append(model_list_append)

                distill_success = False
                distill_val_auc = None
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    try:
                        with time_limit(STAGE_TIMEOUT_S):
                            student_model = distill_to_student_multiclass(
                                teacher_model=model_list_append,
                                X_train=train_aug_x,
                                y_train=train_aug_y,
                                X_val=val_aug_x,
                                y_val=val_aug_y,
                                num_classes=num_classes,
                                device=device,
                                epochs=30,
                            )
                    except TimeoutExceeded as te:
                        logger.event("timeout", 1)
                        logger.end_round(status="skipped", skip_reason="timeout", timeout_stage="distill", error=str(te))
                        i += 1
                        continue
                    base_models_with_distillation.append(student_model)
                    distill_success = True
                    distill_val_auc = getattr(student_model, "distill_val_auc", None)
                except Exception:
                    logger.event("distill_error", 1)
                    base_models_with_distillation.append(model_list_append)

                logger.end_round(
                    status="success",
                    val_metric_main="auc",
                    val_score=float(param_best_auc) / 100.0,
                    val_score_before_opt=float(val_auc) / 100.0 if val_auc is not None else None,
                    val_trust_metric="brier" if val_brier is not None else None,
                    val_trust=val_brier,
                    best_score_so_far=float(best_auc) / 100.0 if best_auc is not None else None,
                    retry_count=int(current_model_retry_count),
                    llm_tokens=int(res.get("total_tokens", 0) or 0),
                    code_len=int(len(param_best_code)) if param_best_code is not None else None,
                    decision={"distill_used": True, "distill_success": bool(distill_success)},
                    distill_val_auc=distill_val_auc,
                )

                if args.enable_feedback and len(code) > 10:
                    model_messages += [
                        {"role": "assistant", "content": param_best_code},
                        {
                            "role": "user",
                            "content": (
                                f"Current model AUC: {param_best_auc:.2f}, Best AUC: {best_auc:.2f}. "
                                "Propose a new classifier more likely to improve AUC."
                            ),
                        },
                    ]
                i += 1

        token_model_list.append(total_token_model_exp)
        models_per_experiment.append(int(len(base_models_without_distillation)))

        metaModelName_list = getMetaModel_list(["LogisticRegression"])
        configurations = [
            {"name": "配置1: basemodel不蒸馏 + 不校准", "models": base_models_without_distillation, "calibrate": False, "results": (config1_acc, config1_f1, config1_auc, config1_pre, config1_rec)},
            {"name": "配置2: basemodel不蒸馏 + 校准", "models": base_models_without_distillation, "calibrate": True, "results": (config2_acc, config2_f1, config2_auc, config2_pre, config2_rec)},
            {"name": "配置3: basemodel蒸馏 + 校准", "models": base_models_with_distillation, "calibrate": True, "results": (config3_acc, config3_f1, config3_auc, config3_pre, config3_rec)},
            {"name": "配置4: basemodel蒸馏 + 不校准", "models": base_models_with_distillation, "calibrate": False, "results": (config4_acc, config4_f1, config4_auc, config4_pre, config4_rec)},
        ]

        for config in configurations:
            print(f"\n【第 {exp + 1} 轮】正在执行 {config['name']} ...")
            for meta_model in metaModelName_list:
                try:
                    _t0 = time.time()
                    with time_limit(STAGE_TIMEOUT_S):
                        result = stacking_ensemble_multiclass_v2(
                            base_models=config["models"],
                            meta_model=meta_model,
                            X_train=train_aug_x,
                            y_train=train_aug_y,
                            X_test=test_aug_x,
                            y_test=test_aug_y,
                            X_val=val_aug_x,
                            y_val=val_aug_y,
                            n_folds=5,
                            calibrate_proba=config["calibrate"],
                            random_state=42 + exp,
                        )
                except TimeoutExceeded:
                    logger.event("timeout", 1)
                    try:
                        result = voting_ensemble_multiclass_fallback(
                            base_models=config["models"],
                            X_train=train_aug_x,
                            y_train=train_aug_y,
                            X_test=test_aug_x,
                            y_test=test_aug_y,
                            X_val=val_aug_x,
                            y_val=val_aug_y,
                            calibrate_proba=config["calibrate"],
                            random_state=42 + exp,
                            max_models=3,
                        )
                    except Exception:
                        continue
                except KeyboardInterrupt:
                    if (time.time() - _t0) >= (STAGE_TIMEOUT_S - 1.0):
                        logger.event("timeout", 1)
                        try:
                            result = voting_ensemble_multiclass_fallback(
                                base_models=config["models"],
                                X_train=train_aug_x,
                                y_train=train_aug_y,
                                X_test=test_aug_x,
                                y_test=test_aug_y,
                                X_val=val_aug_x,
                                y_val=val_aug_y,
                                calibrate_proba=config["calibrate"],
                                random_state=42 + exp,
                                max_models=3,
                            )
                        except Exception:
                            continue
                    else:
                        raise
                except Exception:
                    logger.event("ensemble_eval_error", 1)
                    try:
                        result = voting_ensemble_multiclass_fallback(
                            base_models=config["models"],
                            X_train=train_aug_x,
                            y_train=train_aug_y,
                            X_test=test_aug_x,
                            y_test=test_aug_y,
                            X_val=val_aug_x,
                            y_val=val_aug_y,
                            calibrate_proba=config["calibrate"],
                            random_state=42 + exp,
                            max_models=3,
                        )
                    except Exception:
                        continue

                m = result["metrics"]
                acc_list, f1_list, auc_list, pre_list, rec_list = config["results"]
                acc_list.append(float(m["accuracy"]) * 100.0)
                f1_list.append(float(m["f1"]) * 100.0)
                auc_list.append(float(m["roc_auc"]) * 100.0)
                pre_list.append(float(m["precision"]) * 100.0)
                rec_list.append(float(m["recall"]) * 100.0)

                logger.log(
                    stage="ensemble_eval",
                    r=exp + 1,
                    task_type="multiclassification",
                    feature_count=int(train_aug_x.shape[1]),
                    base_model_count=int(len(config["models"])),
                    meta_config={
                        "config_name": config["name"],
                        "calibrate_proba": bool(config["calibrate"]),
                        "meta_model": type(meta_model).__name__,
                        "num_classes": int(num_classes),
                    },
                    test_metric_main="auc",
                    test_score=float(m.get("roc_auc", float("nan"))),
                    brier=m.get("brier", None),
                    calibration_check_triggered=1 if config["calibrate"] else 0,
                )

                print(f"\n========== 第 {exp + 1} 轮 {config['name']} 结果 ({type(meta_model).__name__}) ==========")
                if result.get("calibration_applied"):
                    print("【信息】已应用 Temperature Scaling 概率校准")
                print(f"准确率   : {m['accuracy']:.4f}")
                print(f"F1分数   : {m['f1']:.4f}")
                print(f"AUC      : {m['roc_auc']:.4f}")
                print(f"精确率   : {m['precision']:.4f}")
                print(f"召回率   : {m['recall']:.4f}")

    print(f"\n\n{'='*80}")
    print(f"完成 {args.exam_iterations} 轮实验，多分类集成学习性能对比统计")

    print("\n【配置1: 不蒸馏 + 不校准】")
    print("  准确率: " + format_mean_std(config1_acc))
    print("  F1分数: " + format_mean_std(config1_f1))
    print("  AUC  : " + format_mean_std(config1_auc))
    print("  精确率: " + format_mean_std(config1_pre))
    print("  召回率: " + format_mean_std(config1_rec))

    print("\n【配置2: 不蒸馏 + 校准】")
    print("  准确率: " + format_mean_std(config2_acc))
    print("  F1分数: " + format_mean_std(config2_f1))
    print("  AUC  : " + format_mean_std(config2_auc))
    print("  精确率: " + format_mean_std(config2_pre))
    print("  召回率: " + format_mean_std(config2_rec))

    print("\n【配置3: 蒸馏 + 校准】")
    print("  准确率: " + format_mean_std(config3_acc))
    print("  F1分数: " + format_mean_std(config3_f1))
    print("  AUC  : " + format_mean_std(config3_auc))
    print("  精确率: " + format_mean_std(config3_pre))
    print("  召回率: " + format_mean_std(config3_rec))

    print("\n【配置4: 蒸馏 + 不校准】")
    print("  准确率: " + format_mean_std(config4_acc))
    print("  F1分数: " + format_mean_std(config4_f1))
    print("  AUC  : " + format_mean_std(config4_auc))
    print("  精确率: " + format_mean_std(config4_pre))
    print("  召回率: " + format_mean_std(config4_rec))

    all_time = time.time() - all_time_start
    print(f"\nTotal experiment time: {all_time:.2f} seconds")

    avg_feat_time = sum(feat_time_list) / len(feat_time_list) if feat_time_list else 0.0
    avg_model_count = sum(models_per_experiment) / len(models_per_experiment) if models_per_experiment else 0.0
    time_stats_content = f"""
=========== 数据集 {ds_name} ===========
超参数优化状态: {'已启用' if args.enable_optimization else '已禁用'}
模型生成反馈状态: {'已启用' if args.enable_feedback else '已禁用'}
综合性能统计报告 ({args.exam_iterations} 轮实验)
=========================================
[时间统计]
1. 实验总运行时间: {all_time:.2f} 秒
2. 每轮平均特征生成模块时间: {avg_feat_time:.2f} 秒
3. 总LLM调用和解析时间（模型生成 + 超参数优化）: {total_llm_time:.2f} 秒
-----------------------------------------
[模型生成和优化Token统计]
4. 每轮模型生成和优化的token数列表: {token_model_list}
 --所有实验的总token数: {sum(token_model_list)}
 --每轮平均token数: {sum(token_model_list) / len(token_model_list):.2f} (若有实验轮次)
-----------------------------------------
[模型数量统计]
5. 每轮有效基模型数量（不蒸馏列表长度）: {models_per_experiment}
 --每轮平均数量: {avg_model_count:.2f}
-----------------------------------------
每轮特征生成时间详情:
{[f"第 {i+1} 轮: {t:.2f} 秒" for i, t in enumerate(feat_time_list)]}
    """
    print(time_stats_content)

    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = os.path.join(result_dir, f"{ds_name}.txt")
    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("=" * 80 + "\n")
            f.write(f"多分类集成学习性能对比统计 - {ds_name}\n")
            f.write("=" * 80 + "\n")
            f.write("=" * 80 + "\n\n")

            f.write("【配置1: 不蒸馏 + 不校准】\n")
            f.write("  准确率: " + format_mean_std(config1_acc) + "\n")
            f.write("  F1分数: " + format_mean_std(config1_f1) + "\n")
            f.write("  AUC  : " + format_mean_std(config1_auc) + "\n")
            f.write("  精确率: " + format_mean_std(config1_pre) + "\n")
            f.write("  召回率: " + format_mean_std(config1_rec) + "\n\n")

            f.write("【配置2: 不蒸馏 + 校准】\n")
            f.write("  准确率: " + format_mean_std(config2_acc) + "\n")
            f.write("  F1分数: " + format_mean_std(config2_f1) + "\n")
            f.write("  AUC  : " + format_mean_std(config2_auc) + "\n")
            f.write("  精确率: " + format_mean_std(config2_pre) + "\n")
            f.write("  召回率: " + format_mean_std(config2_rec) + "\n\n")

            f.write("【配置3: 蒸馏 + 校准】\n")
            f.write("  准确率: " + format_mean_std(config3_acc) + "\n")
            f.write("  F1分数: " + format_mean_std(config3_f1) + "\n")
            f.write("  AUC  : " + format_mean_std(config3_auc) + "\n")
            f.write("  精确率: " + format_mean_std(config3_pre) + "\n")
            f.write("  召回率: " + format_mean_std(config3_rec) + "\n\n")

            f.write("【配置4: 蒸馏 + 不校准】\n")
            f.write("  准确率: " + format_mean_std(config4_acc) + "\n")
            f.write("  F1分数: " + format_mean_std(config4_f1) + "\n")
            f.write("  AUC  : " + format_mean_std(config4_auc) + "\n")
            f.write("  精确率: " + format_mean_std(config4_pre) + "\n")
            f.write("  召回率: " + format_mean_std(config4_rec) + "\n\n")

            f.write("=" * 80 + "\n\n")
            f.write(time_stats_content)
        print(f"\n结果已保存到: {result_file_path}")
    except Exception as e:
        print(f"\n保存结果文件失败: {e}")

    try:
        logger.close()
    except Exception:
        pass
