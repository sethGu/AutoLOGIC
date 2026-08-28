import os
import sys

# Add project root directory
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   # Current file directory
                 "..", "..")                  # Go up two levels
)
sys.path.append(project_root)

# Internal project imports
from agent import AgenticReasoningAgent
from mystage1 import Stage1Classifier  # Automated Feature Engineering for tabular datasets
from mystage1 import data
from mystage1.run_llm_code import run_llm_code

import re
import time
import psutil  

from utils.ensembleUtils import getMetaModel_list
from utils.CL_ensemble_utils import get_classification_param_prompt
from utils.model_generate import (
    build_prompt_samples,
    get_model_prompt,
    generate_model_2,
)
from utils.ensemble_utils2 import stacking_ensemble_v2
from utils.quant_log import QuantLogger
from utils.task_protocol import (
    log_feature_provenance,
    log_task_protocol,
    materialize_task_protocol,
    write_task_protocol,
)
from utils.time_limit import time_limit, TimeoutExceeded

import copy
import pandas as pd
import torch
import statistics

# Third-party library imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy.special import logit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pickle
import random
import argparse
import warnings
from utils.csv_dataset_loader import read_txt_file, resolve_description_path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

STAGE_TIMEOUT_S = 300


def get_data_split(ds, seed):
    def get_df(X, y):
        df = pd.DataFrame(
            data=np.concatenate([X, np.expand_dims(y, -1)], -1), columns=ds[4]
        )
        cat_features = ds[3]
        for c in cat_features:
            if len(np.unique(df.iloc[:, c])) > 50:
                cat_features.remove(c)
                continue
            df[df.columns[c]] = df[df.columns[c]].astype("int32")
        return df.infer_objects()

    ds = copy.deepcopy(ds)

    X = ds[1].numpy() if type(ds[1]) == torch.Tensor else ds[1]
    y = ds[2].numpy() if type(ds[2]) == torch.Tensor else ds[2]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    df_train = get_df(X_train, y_train)
    df_test = get_df(X_test, y_test)
    df_train.iloc[:, -1] = df_train.iloc[:, -1].astype("category")
    df_test.iloc[:, -1] = df_test.iloc[:, -1].astype("category")

    return ds, df_train, df_test


def _normalize_target_to_int(s: pd.Series) -> pd.Series:
    s_str = s.astype("string").str.strip().str.lower()
    map01 = {
        "yes": 1, "y": 1, "true": 1, "t": 1, "positive": 1, "pos": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "f": 0, "negative": 0, "neg": 0, "0": 0
    }

    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        uniq = set(pd.unique(s.dropna()))
        if uniq <= {0, 1, 0.0, 1.0}:
            return s.astype(int)
        return s.round().astype(int)

    non_na = s_str.dropna()
    if not non_na.empty and non_na.isin(map01).all():
        return s_str.map(map01).astype(int)

    le = LabelEncoder()
    enc = le.fit_transform(s_str.fillna("__missing__"))
    return pd.Series(enc, index=s.index, dtype=int)


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


def load_new_dataset(dataset_name: str, base_loc: str, seed: int = 42, test_size: float = 0.2):
    loc = os.path.join(base_loc, dataset_name + ".csv")
    df = pd.read_csv(loc).convert_dtypes()
    df = df.replace({pd.NA: np.nan})

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64", "Int64"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")

    target_column_name = df.columns[-1]
    df = df[~df[target_column_name].isna()].copy()
    df[target_column_name] = _normalize_target_to_int(df[target_column_name])

    if int(pd.Series(df[target_column_name]).nunique()) > 2:
        raise ValueError(f"{dataset_name} 不是二分类数据集（类别数>2），请使用多分类流程。")

    if dataset_name == "nyc_taxi_yellow_2023_01_1m":
        try:
            df_train_raw, df_test_raw = train_test_split(
                df,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
                stratify=df[target_column_name],
            )
        except ValueError:
            df_train_raw, df_test_raw = train_test_split(
                df,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
            )
    else:
        df_train_raw, df_test_raw = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )

    feature_cols = [c for c in df.columns if c != target_column_name]
    X_train = df_train_raw[feature_cols]
    X_test = df_test_raw[feature_cols]

    num_cols = X_train.select_dtypes(include=["number", "boolean"]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    X_train = X_train.copy()
    X_test = X_test.copy()
    if num_cols:
        X_train[num_cols] = X_train[num_cols].apply(pd.to_numeric, errors="coerce")
        X_test[num_cols] = X_test[num_cols].apply(pd.to_numeric, errors="coerce")
    if cat_cols:
        for c in cat_cols:
            X_train[c] = X_train[c].astype("object").where(X_train[c].notna(), None)
            X_test[c] = X_test[c].astype("object").where(X_test[c].notna(), None)

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0))])
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=None, strategy="constant", fill_value="__MISSING__")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_fit = X_train
    if len(X_fit) > 50000:
        X_fit = X_fit.sample(n=50000, random_state=seed)
    preproc.fit(X_fit)

    X_train_enc = preproc.transform(X_train)
    X_test_enc = preproc.transform(X_test)
    if hasattr(X_train_enc, "toarray"):
        X_train_enc = X_train_enc.toarray()
    if hasattr(X_test_enc, "toarray"):
        X_test_enc = X_test_enc.toarray()
    X_train_enc = np.asarray(X_train_enc, dtype=np.float32)
    X_test_enc = np.asarray(X_test_enc, dtype=np.float32)

    feature_order = num_cols + cat_cols
    df_train_feat = pd.DataFrame(X_train_enc, columns=feature_order, index=X_train.index)
    df_test_feat = pd.DataFrame(X_test_enc, columns=feature_order, index=X_test.index)

    df_train = pd.concat(
        [
            df_train_feat.reset_index(drop=True),
            df_train_raw[target_column_name].reset_index(drop=True).astype(int),
        ],
        axis=1,
    )
    df_test = pd.concat(
        [
            df_test_feat.reset_index(drop=True),
            df_test_raw[target_column_name].reset_index(drop=True).astype(int),
        ],
        axis=1,
    )

    desc_path = resolve_description_path(base_loc, dataset_name)
    dataset_description = read_txt_file(desc_path, encoding="utf-8") if desc_path else ""
    return df_train, df_test, target_column_name, dataset_description

def load_origin_data(dataset_name, seed=0):
    # Old dataset keywords that use .pkl files (substring matching)
    old_keys = ('credit','cd1','cc1','ld1','cc2','cd2','cf1','balance-scale')
    name_l = dataset_name.lower()
    is_old = any(k in name_l for k in old_keys)

    if is_old:
        loc = f"{project_root}/data/{dataset_name}.pkl"
        if os.path.exists(loc):
            with open(loc, 'rb') as f:
                ds = pickle.load(f)

            if 'credit' in name_l and not isinstance(ds[1], pd.DataFrame):
                ds, df_train, df_test = get_data_split(ds, seed=seed)
            else:
                df_train, df_test = ds[1].copy(), ds[2].copy()
            target_column_name = ds[4][-1]
            dataset_description = ds[-1]

            y_all = pd.concat(
                [df_train[target_column_name], df_test[target_column_name]],
                ignore_index=True
            )
            y_all_norm = _normalize_target_to_int(y_all)
            n_train = len(df_train)
            df_train[target_column_name] = y_all_norm.iloc[:n_train].to_numpy()
            df_test[target_column_name] = y_all_norm.iloc[n_train:].to_numpy()

            if int(pd.Series(df_train[target_column_name]).nunique()) > 2:
                raise ValueError(f"{dataset_name} 不是二分类数据集（类别数>2），请使用多分类流程。")
            df_train[target_column_name] = df_train[target_column_name].astype(int)
            df_test[target_column_name] = df_test[target_column_name].astype(int)
            df_train = _encode_non_numeric_features(df_train, target_column_name)
            df_test = _encode_non_numeric_features(df_test, target_column_name)
            return df_train, df_test, target_column_name, dataset_description

    csv_base = os.path.join(project_root, "data", "csv_data")
    return load_new_dataset(dataset_name, base_loc=csv_base, seed=seed, test_size=0.2)

def base_model(seed, n_jobs=-1):
    rforest = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        class_weight='balanced',
        n_jobs=n_jobs,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # Repeatable random data split
    param_grid = {
        "min_samples_leaf": [0.001, 0.01, 0.05],  # Adjustment range
        "max_depth": [5, 10, None]  # Added depth control
    }
    gsmodel = GridSearchCV(rforest, param_grid, cv=cv, scoring='f1', n_jobs=n_jobs)

    return gsmodel

def code_exec(code):
    try:
        compiled_code = compile(code, "<string>", "exec")
        scope = {"__builtins__": __builtins__}
        exec(compiled_code, scope, scope)
        return None, scope
    except Exception as e:
        print("Code could not be executed:", e)
        return str(e), None

def print_stats(name, values):
    print(f"{name}: {np.mean(values):.2f} ± {np.std(values):.2f}")

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
        if line.strip().startswith("class myclassifier") or line.strip().startswith("import") or line.strip().startswith(
                "from"):
            cleaned_lines.append(line)
        elif cleaned_lines:  # If code block recording has started, continue adding subsequent code
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def to_pd(df_train, target_name):
    y = df_train[target_name].astype(int)
    x = df_train.drop(target_name, axis=1)

    return x, y

def format_mean_std(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f}±{std_dev:.2f}"

def generate_feat(
        base_classifier,
        df_train,
        df_test,
        dataset_name,
        round_num = 0,
        llm_model='gpt-3.5-turbo',
        iterations=10,
        target_column_name='class',  # Target column name
        dataset_description=None,
        task_type="classification",
        base_url:str=None,api_key:str=None,
        logger=None
):
    if base_url is None or api_key is None:
        raise ValueError("base_url and api_key must be provided.")
    input_columns = list(df_train.columns)
    stage1_clf = Stage1Classifier(base_classifier=base_classifier,
                                llm_model=llm_model,
                                iterations=iterations,
                               )

    stage1_clf.fit_pandas(df_train,
                         target_column_name=target_column_name,
                         dataset_description=dataset_description,
                         dataset_name=dataset_name,
                         round_num=round_num,
                         task_type=task_type,
                         base_url=base_url,
                         api_key=api_key,
                         logger=logger
                         )

    df_train_aug = run_llm_code(stage1_clf.code, df_train, target_column_name)  # Training set after feature engineering
    df_test_aug = run_llm_code(stage1_clf.code, df_test, target_column_name)  # Test set after feature engineering

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

def format_mean_std(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f}±{std_dev:.2f}"

# =========================
# 学生网络定义
# =========================
class StudentNet(nn.Module):
    """用于知识蒸馏的学生MLP网络"""
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.oof_refit_capable = False
        self.route_role = "prefit_student"
        self.fit_protocol = "distilled_once_then_reused"
        
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def fit(self, X, y):
        """占位方法，用于兼容sklearn接口"""
        pass
    
    def predict_proba(self, X):
        """预测概率，用于兼容sklearn接口"""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            logits = self(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs


# =========================
# 知识蒸馏损失
# =========================
def kd_loss(student_logits, teacher_logits, T=3.0):
    """知识蒸馏损失 (KL散度)"""
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


# =========================
# 学生网络蒸馏训练
# =========================
def distill_to_student(teacher_model, X_train, y_train, X_val, y_val, device='cpu', epochs=30):
    """
    将教师模型蒸馏到学生MLP网络
    
    Args:
        teacher_model: 已训练的教师模型 (sklearn接口)
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        device: 'cpu' or 'cuda'
        epochs: 训练轮数
    
    Returns:
        student_model: 训练好的学生网络
    """
    print("    开始知识蒸馏...")
    
    # 数据预处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 获取教师模型的logits
    def get_logits(model, X):
        probs = model.predict_proba(X)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        # 确保为二维 [p0, p1]
        if probs.ndim == 1:
            probs = np.stack([1 - probs, probs], axis=1)
        elif probs.shape[1] == 1:
            probs = np.concatenate([1 - probs, probs], axis=1)
        return logit(probs)
    
    train_logits = get_logits(teacher_model, X_train)
    val_logits = get_logits(teacher_model, X_val)
    
    # 创建DataLoader
    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.long),
        torch.tensor(train_logits, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val, dtype=torch.long),
        torch.tensor(val_logits, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    # 初始化学生网络
    input_dim = X_train_scaled.shape[1]
    student = StudentNet(input_dim=input_dim, hidden_dims=(256, 128), dropout=0.3).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 蒸馏超参数
    T = 3.0  # 温度
    lambda_kd = 0.5  # 蒸馏损失权重
    
    best_val_auc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        student.train()
        for xb, yb, tb in train_loader:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            optimizer.zero_grad()
            logits = student(xb)
            loss_ce = F.cross_entropy(logits, yb)
            loss_kd = kd_loss(logits, tb, T)
            loss = (1 - lambda_kd) * loss_ce + lambda_kd * loss_kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        
        # 验证阶段
        student.eval()
        probs, targets = [], []
        with torch.no_grad():
            for xb, yb, tb in val_loader:
                xb = xb.to(device)
                logits = student(xb)
                p = F.softmax(logits, dim=1)[:, 1]
                probs.extend(p.cpu().numpy())
                targets.extend(yb.numpy())
        
        try:
            if len(np.unique(targets)) > 1:
                auc = roc_auc_score(targets, probs)
            else:
                auc = 0.5
        except ValueError:
            auc = 0.5
        
        if auc > best_val_auc:
            best_val_auc = auc
            best_state = copy.deepcopy(student.state_dict())
    
    # 加载最佳权重
    if best_state:
        student.load_state_dict(best_state)
    
    # 保存scaler到student对象中
    student.scaler = scaler
    
    # 重写predict_proba方法以使用scaler
    def predict_proba_with_scaler(self, X):
        self.eval()
        device = next(self.parameters()).device
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            logits = self(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs
    
    student.predict_proba = lambda X: predict_proba_with_scaler(student, X)
    
    print(f"    蒸馏完成，学生网络验证集 AUC: {best_val_auc:.4f}")
    student.distill_val_auc = float(best_val_auc)
    student.distill_method = "kl"
    return student

if __name__ == '__main__':
    # Parse command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default="0", type=str, help='GPU settings')
    parser.add_argument('-s', '--default_seed', default=42, type=int, help='Random seed')
    parser.add_argument('-l', '--llm', default='gpt-3.5-turbo', type=str, help='Large language model')
    # parser.add_argument('-l', '--llm', default='gpt-4o', type=str, help='Large language model')
    parser.add_argument('-e', '--exam_iterations', default=3, type=int, help='Number of experiments')  
    parser.add_argument('-f', '--feat_iterations', default=10, type=int, help='Feature iteration count')
    parser.add_argument('-m', '--model_iterations', default=5, type=int, help='Model iteration count')
    parser.add_argument('-p', '--param_iterations', default=5, type=int,help='Parameter tuning iterations')
    parser.add_argument('-d', '--dataset', default = "cc1",help = "Dataset")
    parser.add_argument('--stage_timeout_s', default=STAGE_TIMEOUT_S, type=int, help='Stage timeout seconds')
    parser.add_argument('--enable_optimization', action='store_true', default=True, 
                        help='Whether to enable model hyperparameter optimization. True = enable, False = disable')
    parser.add_argument('--enable_feedback', action='store_true', default=True,
                        help='Whether to enable model generation feedback. True = enable, False = disable')
    parser.add_argument('--task-spec', default=None, help='Optional JSON task specification')
    args = parser.parse_args()

    task_spec, plan_bundle = materialize_task_protocol(
        "classification",
        args.dataset,
        args,
        task_spec_path=args.task_spec,
    )

    STAGE_TIMEOUT_S = int(args.stage_timeout_s)

    """
    OpenAI API Configuration
    """
    # TODO     OpenAI API Configuration
    # --- Get environment variables ---nment variables ---
    env_url = os.getenv("OPENAI_BASE_URL")
    env_key = os.getenv("OPENAI_API_KEY")
    # --- Manual configuration ---
    manual_url = ""
    manual_key = ""

    # --- Selection priority: environment variables > manual configuration ---
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


    # Model label, globally distinguish LLM-generated models
    model_tab = 1

    ds_name = args.dataset
    # cd1 cc1 ld1  cc2 cd2 cf1 balance-scale ds_credit
    print(f"=========== Dataset {ds_name} ===========")
    logger = QuantLogger(
        task="classification",
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
    # Print hyperparameter optimization switch status
    print(f"Hyperparameter Optimization Status: {'Enabled' if args.enable_optimization else 'Disabled'}")
    # Print model generation feedback switch status
    print(f"Model Generation Feedback Status: {'Enabled' if args.enable_feedback else 'Disabled'}")

    # ===== 4种配置的结果存储 =====
    # 配置1: 不蒸馏 + 不校准
    config1_acc, config1_f1, config1_auc, config1_pre, config1_rec = [], [], [], [], []
    # 配置2: 不蒸馏 + 校准
    config2_acc, config2_f1, config2_auc, config2_pre, config2_rec = [], [], [], [], []
    # 配置3: 蒸馏 + 校准
    config3_acc, config3_f1, config3_auc, config3_pre, config3_rec = [], [], [], [], []
    # 配置4: 蒸馏 + 不校准
    config4_acc, config4_f1, config4_auc, config4_pre, config4_rec = [], [], [], [], []

    # Number of base models successfully added to ensemble per round
    models_per_experiment = []

    # New: Used to store Voting metrics results
    test_acc_list_ensemble_voting = []
    test_f1_list_ensemble_voting = []
    test_auc_list_ensemble_voting = []
    test_pre_list_ensemble_voting = []
    test_rec_list_ensemble_voting = []

    # ---------------------- Time & Memory Statistics Initialization ----------------------
    all_time_start = time.time()
    feat_time_list = []  # Store feature generation time per round
    token_model_list = [] # Store model generation and optimization token count per round
    token_feat_list = [] # Store feature generation token count per round
    total_llm_time = 0.0  # Store total LLM call + parsing time
    
    for exp in range(args.exam_iterations):
        print(f"=========== Experiment {exp + 1}/{args.exam_iterations} ===========")
        
        total_token_model_exp = 0 # Store total token count for each experiment

        # List to store results for each experiment
        test_auc_list = []
        seed = args.default_seed + exp*10
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        # Load dataset and model
        df_train_aug, df_test_aug, target_column_name, dataset_description = load_origin_data(ds_name)
        baseline_model = base_model(seed)  # Random Forest model

        
        # Check if feature engineering should be skipped
        if args.feat_iterations > 0:
            feat_start = time.time()
            df_train_aug, df_test_aug = generate_feat(
                base_classifier=baseline_model,  # Model for evaluating generated features is Random Forest
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
                logger=logger

            )
            print("Feature generation completed")
            feat_end = time.time()
            feat_elapsed = feat_end - feat_start
            feat_time_list.append(feat_elapsed)
            print(f"Feature generation completed, time elapsed: {feat_elapsed:.2f} seconds")
        else:
            print("Feature engineering skipped (feat_iterations=0)")
            feat_time_list.append(0.0)

        
        

        df_train_aug,df_valid_aug = train_test_split(df_train_aug,test_size=0.25,random_state=seed,stratify=df_train_aug[target_column_name])
        

        # Data conversion to get feature matrix and label (target) vector
        train_aug_x, train_aug_y = to_pd(df_train_aug, target_column_name)
        val_aug_x, val_aug_y = to_pd(df_valid_aug, target_column_name)
        test_aug_x, test_aug_y = to_pd(df_test_aug, target_column_name)
        logger.log(
            stage="feature_summary",
            r=exp + 1,
            task_type="classification",
            feature_count=int(train_aug_x.shape[1]),
            base_model_count=0,
            meta_config={
                "feat_iterations": int(args.feat_iterations),
                "llm": args.llm,
            },
        )

        # Data format needed for constructing prompts
        s = build_prompt_samples(df_train_aug)

        # Prompt for LLM to generate classification model code
        model_prompt = get_model_prompt(
            target_column_name=target_column_name,
            samples=s,
        )


        model_messages = [
            {
                "role": "system",
                "content": (
                    "You are a top-level machine learning classification expert.\n"
                    "Your task is to help me iteratively search for the most suitable classifier model.\n"
                    "Your primary goal is to maximize the AUC (Area Under the ROC Curve) on the test set.\n"
                    "You must focus on improving AUC more than any other metric.\n"
                    "Your answer should only generate valid Python code.\n\n"
                    "**IMPORTANT: You MUST set hyperparameters to utilize all available CPU cores for training to speed up the process.**\n\n"
                    "**CRITICAL: The sklearn version is >= 1.4 with important API changes:**\n"
                    "- Many ensemble/meta-learning classes changed parameter names from 'base_estimator' to 'estimator'\n"
                    "- Some classes changed attribute names and other APIs\n"
                    "- When encountering 'unexpected keyword argument' errors, consult the official sklearn 1.4+ documentation\n"
                ),
            },
            {
                "role": "user",
                "content": model_prompt,
            },
        ]


        model_iter = args.model_iterations
        best_auc = 0
        best_code = None
        i = 0

        # Base model list for ensemble learning
        base_models_without_distillation = []  # 不蒸馏的模型列表
        base_models_with_distillation = []     # 蒸馏后的模型列表
        base_models = base_models_without_distillation  # 默认指向不蒸馏列表
        # Error retry mechanism
        max_retries_per_model = 3  # Maximum 3 retries per model
        current_model_retry_count = 0
        
        # Model generation iteration
        while i < model_iter:
            if not logger.in_round():
                logger.start_round(
                    stage="model_iter",
                    r=i + 1,
                    exp=exp + 1,
                    task_type="classification",
                    feature_count=int(train_aug_x.shape[1]),
                    base_model_count=int(len(base_models_without_distillation)),
                    meta_config={
                        "enable_optimization": bool(args.enable_optimization),
                        "enable_feedback": bool(args.enable_feedback),
                        "llm": args.llm,
                    },
                )
            try:
                # ---------------------- LLM Model Generation Time Statistics ----------------------
                model_llm_start = time.time()
                # Generate downstream model code
                try:
                    with time_limit(STAGE_TIMEOUT_S):
                        res = generate_model_2(args.llm, model_messages,base_url,api_key)
                except TimeoutExceeded as te:
                    logger.event("timeout", 1)
                    logger.end_round(status="skipped", skip_reason="timeout", timeout_stage="llm_model_generate", error=str(te))
                    current_model_retry_count = 0
                    i += 1
                    continue
                code = res['code']
                total_tokens_model = res['total_tokens']
                total_token_model_exp += total_tokens_model
                # todo Add code_clean code
                code = clean_llm_code(code)
                model_llm_end = time.time()
                total_llm_time += (model_llm_end - model_llm_start)
                # -----------------------------------------------------------

                # Dynamically modify class name
                new_class_name = f"myclassifier_{i + 1}"
                code = re.sub(r'class\s+myclassifier[_\w]*\s*(\([^)]*\))?\s*:', f'class {new_class_name}:', code, count=1)
                print(f"----------------------------Original Code-----------------------")
                print(code)
            except Exception as e:
                print("Error in LLM API." + str(e))
                logger.event("llm_api_error", 1)
                continue

            e, exec_scope = code_exec(code)
            # Check compilation error
            if e is not None:  # If generated code execution fails, feed error info back to LLM to generate fixed code
                current_model_retry_count += 1
                logger.event("compile_error", 1)
                logger.event("retry", 1)
                print(f"Compilation error (retry {current_model_retry_count}/{max_retries_per_model}): {e}")
                
                if current_model_retry_count >= max_retries_per_model:
                    print(f"Maximum retries exceeded, skipping model {i+1}")
                    logger.end_round(
                        status="skipped",
                        skip_reason="compile_error",
                        retry_count=int(current_model_retry_count),
                        code_len=int(len(code)) if code is not None else None,
                        llm_tokens=int(total_tokens_model) if 'total_tokens_model' in locals() else None,
                    )
                    current_model_retry_count = 0  # 重置计数器
                    i += 1  # 跳过the模型
                    continue
                
                model_messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"""
                            The classifier code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```
                            Remember, your answer should only generate code.
                            Do not include explanations or comments outside the code block.
                            Generate next code block(fixing error?):
                            """,
                    },
                ]
                continue


            try:
                # Model instance
                model_class = exec_scope[new_class_name]
                model = model_class()
                try:
                    with time_limit(STAGE_TIMEOUT_S):
                        model.fit(train_aug_x, train_aug_y)
                        model_copy = copy.deepcopy(model)  # 在fit之后深拷贝已拟合的模型
                        pred = model.predict(val_aug_x)
                        proba = model.predict_proba(val_aug_x)
                except TimeoutExceeded as te:
                    logger.event("timeout", 1)
                    logger.end_round(status="skipped", skip_reason="timeout", timeout_stage="base_model_fit_predict", error=str(te))
                    current_model_retry_count = 0
                    i += 1
                    continue
                model_list_append = model_copy
                
            except Exception as e:
                current_model_retry_count += 1
                logger.event("runtime_error", 1)
                logger.event("retry", 1)
                error_type = type(e).__name__
                error_msg = str(e)
                if isinstance(e, KeyError) and len(getattr(e, "args", [])) == 1 and str(e.args[0]) == new_class_name:
                    available_classes = [k for k, v in exec_scope.items() if isinstance(v, type)]
                    error_msg = f"missing_class={new_class_name}; available_classes={available_classes[:20]}"
                print(f"Model code execution failed with error: {error_type}: {error_msg}")
                print(f"Runtime error (retry {current_model_retry_count}/{max_retries_per_model})")
                
                if current_model_retry_count >= max_retries_per_model:
                    print(f"Maximum retries exceeded, skipping model {i+1}")
                    logger.end_round(
                        status="skipped",
                        skip_reason="runtime_error",
                        retry_count=int(current_model_retry_count),
                        last_error_type=error_type,
                        last_error=error_msg,
                        llm_tokens=int(total_tokens_model) if 'total_tokens_model' in locals() else None,
                    )
                    current_model_retry_count = 0  # Reset counter
                    i += 1  # Skip this model
                    continue
                
                # Build detailed error feedback
                error_hint = ""
                if "base_estimator" in error_msg:
                    error_hint = "\n**HINT**: The parameter 'base_estimator' is not recognized in sklearn 1.4+. Check the documentation - some classes changed this parameter name. Consider alternatives in the official documentation."
                elif "got an unexpected keyword argument" in error_msg:
                    error_hint = "\n**HINT**: An unexpected keyword argument was encountered. This might be due to sklearn version differences. Verify all parameter names in the official sklearn 1.4+ documentation."
                elif error_type == "KeyError" and "missing_class=" in error_msg:
                    error_hint = "\n**HINT**: You must define a class named `myclassifier` (it will be renamed automatically). Output python code only."
                
                model_messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"""
                        Code execution failed with error: {error_type}: {error_msg}
                        Code: ```python{code}```{error_hint}
                        
                        Retry attempt {current_model_retry_count}/{max_retries_per_model}. Please fix the code and generate the next version:
                        """,
                    },
                ]
                # Don't auto-skip, give LLM a chance to fix error
                continue


            test_auc = roc_auc_score(val_aug_y, proba) * 100
            val_auc = float(test_auc) / 100.0
            try:
                proba_pos = proba[:, 1] if getattr(proba, "ndim", 1) > 1 else proba
                val_brier = float(brier_score_loss(val_aug_y, proba_pos))
            except Exception:
                val_brier = None

            # Model executed successfully, reset retry counter
            current_model_retry_count = 0

            # todo Add parameter optimization after this
            param_best_code = code
            param_best_auc = test_auc
            
            # Initialize parameter optimization prompt only if optimization is enabled
            if args.enable_optimization:
                param_prompt = get_classification_param_prompt(
                    best_code=param_best_code,
                    best_auc=param_best_auc,
                    dataset_description=dataset_description,
                    X_test=val_aug_x,
                    feature_columns=train_aug_x.columns.tolist(),
                    dataset_name=ds_name,
                    max_rows=10
                )

                param_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a classification optimization assistant.\n"
                            "Your task is to help me improve the test AUC of the given classifier\n"
                            "by tuning hyperparameters only. Your answer must contain only executable Python code.\n\n"
                            "**PRIMARY GOAL:** Focus on finding better *combinations* of hyperparameters, not just isolated changes to single parameters.\n\n"
                            "**CRITICAL CONSTRAINT 1: Parallelism**\n"
                            "- **You MUST preserve or add `n_jobs=-1` (or `thread_count=-1` for CatBoost) to ensure multi-core training.**\n"
                            "- Do NOT remove this parameter during optimization.\n\n"
                            "**CRITICAL CONSTRAINT 2: Efficient Search Space**\n"
                            "For ensemble/tree-based models (e.g., RandomForest, XGBoost, LightGBM, CatBoost):\n"
                            "- **n_estimators (or equivalent):** MUST be $\\le 300$ (e.g., 50-300).\n"
                            "- **max_depth (or equivalent):** MUST be $\\le 10$ (e.g., 3-10).\n"
                            "- **num_leaves (for LightGBM):** MUST be $\\le 64$.\n"
                            "For other models (e.g., SVM, Neural Networks, kNN):\n"
                            "- **C/alpha (regularization):** Use standard ranges (e.g., 0.001 to 10.0).\n"
                            "- **max_iter (or equivalent):** MUST be $\\le 500$.\n"
                            "Prioritize small, efficient parameter values."
                        )
                    },
                    {
                        "role": "user",
                        "content": param_prompt
                    },
                ]
                # -----------------------------------------------------------------------------

            # Start parameter optimization iteration, only execute if optimization is enabled
            param_iter_range = range(args.param_iterations) if args.enable_optimization else range(0)
            for p_iter in param_iter_range:
                print(f"++++++ Optimization {p_iter + 1} +++++++++")
                try:
                    # ---------------------- LLM Parameter Optimization Time Statistics ----------------------
                    param_llm_start = time.time()
                    try:
                        with time_limit(STAGE_TIMEOUT_S):
                            res = generate_model_2(args.llm, param_messages,base_url,api_key)
                    except TimeoutExceeded:
                        logger.event("timeout", 1)
                        break
                    param_code = res['code']
                    token_parm = res['total_tokens']
                    total_token_model_exp += token_parm

                    param_code = clean_llm_code(param_code)
                    param_llm_end = time.time()
                    total_llm_time += (param_llm_end - param_llm_start)
                    # -----------------------------------------------------------

                    param_new_class_name = f"myclassifier_{i+1}_param_{p_iter + 1}"
                    # Use regex to match any myclassifier class name (including numbers that LLM may modify)
                    param_code = re.sub(
                        r'class\s+myclassifier[_\w]*\s*(\([^)]*\))?\s*:', 
                        f'class {param_new_class_name}:', 
                        param_code,
                        count=1  # Only replace first match
                    )

                    param_err, param_scope = code_exec(param_code)

                    if param_err is not None:  # Compilation error handling
                        print(f"Hyperparameter optimization code compilation error: {param_err}")
                        param_messages += [
                            {"role": "assistant", "content": param_code},
                            {
                                "role": "user", 
                                "content": f"""
                                The optimized classifier code execution failed with compilation error: {param_err}
                                Code: ```python{param_code}```
                                Remember, your answer should only generate code.
                                Do not include explanations or comments outside the code block.
                                Generate next code block (fixing error?):
                                """
                            },
                        ]
                        continue

                    print('---------------Code after optimization\n' + param_code)
                    # Create model using new class name
                    try:
                        model_class = param_scope[param_new_class_name]
                        myclassifier_tuned = model_class()
                        try:
                            with time_limit(STAGE_TIMEOUT_S):
                                myclassifier_tuned.fit(train_aug_x, train_aug_y)
                                model_copy = copy.deepcopy(myclassifier_tuned)  # 在fit之后深拷贝已拟合的模型
                                proba_tuned = myclassifier_tuned.predict_proba(val_aug_x)
                                auc_tuned = roc_auc_score(val_aug_y, proba_tuned) * 100
                        except TimeoutExceeded:
                            logger.event("timeout", 1)
                            break

                        if auc_tuned > param_best_auc:
                            print(f"Parameter optimization improvement: {param_best_auc} --> {auc_tuned}")
                            param_best_auc = auc_tuned
                            param_best_code = param_code
                            model_list_append = model_copy
                        

                        param_messages += [
                            {"role": "assistant", "content": param_code},
                            {"role": "user",
                                "content": f"Current AUC: {auc_tuned:.2f}, Best AUC: {param_best_auc:.2f}. Please improve further."},
                        ]
                    
                    except Exception as exec_error:
                        # Runtime error handling: Model training execution failed
                        print(f"Hyperparameter optimization model training execution error: {type(exec_error).__name__}: {exec_error}")
                        param_messages += [
                            {"role": "assistant", "content": param_code},
                            {
                                "role": "user",
                                "content": f"""
                                Code execution failed with error: {type(exec_error).__name__}: {exec_error}
                                Code: ```python{param_code}```
                                Generate next code block (fixing error?):
                                """
                            },
                        ]
                        continue

                except Exception as e:
                    print("Tuning failed:", str(e))
                    continue

            # Update global best model and global best parameters
            if best_auc < param_best_auc:
                best_auc = param_best_auc
                best_code = param_best_code

            # ========== 保存不蒸馏的模型 ==========
            base_models_without_distillation.append(model_list_append)
            
            # ========== 知识蒸馏：将优化后的模型蒸馏到MLP学生网络 ==========
            print(f"  模型 {i+1} - 开始知识蒸馏到MLP学生网络...")
            distill_success = False
            distill_val_auc = None
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                try:
                    with time_limit(STAGE_TIMEOUT_S):
                        student_model = distill_to_student(
                            teacher_model=model_list_append,
                            X_train=train_aug_x,
                            y_train=train_aug_y,
                            X_val=val_aug_x,
                            y_val=val_aug_y,
                            device=device,
                            epochs=30  # 蒸馏训练轮数
                        )
                except TimeoutExceeded as te:
                    logger.event("timeout", 1)
                    logger.end_round(status="skipped", skip_reason="timeout", timeout_stage="distill", error=str(te))
                    i += 1
                    continue
                
                # 保存蒸馏后的学生模型
                base_models_with_distillation.append(student_model)
                distill_success = True
                distill_val_auc = getattr(student_model, "distill_val_auc", None)
                print(f"  模型 {i+1} - 蒸馏成功，使用学生网络作为集成基模型")
                
            except Exception as distill_error:
                logger.event("distill_error", 1)
                print(f"  模型 {i+1} - 蒸馏失败: {distill_error}")
                print(f"  模型 {i+1} - 回退使用原始优化模型")
                # 蒸馏失败，使用原始模型
                base_models_with_distillation.append(model_list_append)

            logger.end_round(
                status="success",
                val_metric_main="auc",
                val_score=float(param_best_auc) / 100.0,
                val_score_before_opt=val_auc if 'val_auc' in locals() else None,
                val_trust_metric="brier" if val_brier is not None else None,
                val_trust=val_brier,
                best_score_so_far=float(best_auc) / 100.0 if best_auc is not None else None,
                retry_count=int(current_model_retry_count),
                llm_tokens=int(total_tokens_model) if 'total_tokens_model' in locals() else None,
                code_len=int(len(param_best_code)) if param_best_code is not None else None,
                decision={
                    "distill_used": True,
                    "distill_success": bool(distill_success),
                },
                distill_val_auc=distill_val_auc,
            )

            # Store results
            test_auc_list.append(param_best_auc)

            # Print current experiment detailed results
            print(f"Current experiment result {i+1}/{args.model_iterations}")
            print(f"Test  AUC: {param_best_auc:.2f}")
            # Continue while loop
            i = i + 1

            # Append prompt for next round of model generation - only add if feedback is enabled
            if args.enable_feedback and len(code) > 10:
                model_messages += [
                    {"role": "assistant", "content": param_best_code},
                    {
                        "role": "user",
                        "content": f"""
                        ✅ The classifier code executed successfully.

                        📈 Current model AUC: {param_best_auc:.4f}
                        🏆 Best historical AUC so far: {best_auc:.4f}

                        Please now propose a new classifier that is **more likely to improve the AUC** on the given test data.
                        The model must differ from all previous ones **by model type or internal structure**.

                        ⚠️ Remember:
                        - You must only output valid Python code for a complete classifier named `myclassifier`.
                        - The class must include all imports and implement: `fit`, `predict`, and `predict_proba`.
                        - Do not repeat models you've already used.
                        - Prioritize models that provide reliable probabilistic outputs to help improve AUC.

                        🎯 Next code block:
                        """,
                    },
                ]
            else:
                model_messages += [
                    {"role": "assistant", "content": param_best_code},
                    {
                        "role": "user",
                        "content": f"""
                        ✅ The classifier code executed successfully.

                        Please now propose a new classifier that is **more likely to improve the AUC** on the given test data.
                        The model must differ from all previous ones **by model type or internal structure**.

                        ⚠️ Remember:
                        - You must only output valid Python code for a complete classifier named `myclassifier`.
                        - The class must include all imports and implement: `fit`, `predict`, and `predict_proba`.
                        - Do not repeat models you've already used.
                        - Prioritize models that provide reliable probabilistic outputs to help improve AUC.

                        🎯 Next code block:
                        """,
                    },
                ]
        token_model_list.append(total_token_model_exp)

        if len(base_models_without_distillation) == 0:
            bm = copy.deepcopy(baseline_model)
            bm.fit(train_aug_x, train_aug_y)
            base_models_without_distillation.append(bm)
        if len(base_models_with_distillation) == 0:
            base_models_with_distillation.extend(base_models_without_distillation)

        # Record the number of valid base models generated in current experiment
        valid_model_count_no_dist = len(base_models_without_distillation)
        valid_model_count_with_dist = len(base_models_with_distillation)
        models_per_experiment.append(valid_model_count_no_dist)
        print(f"Round {exp + 1} valid base model count (no distillation): {valid_model_count_no_dist}, (with distillation): {valid_model_count_with_dist}")

        """
        集成学习 - 4种配置对比
        """
        # List of meta-model names to test
        metaModelName_list = [
            # 'RandomForestClassifier',
            # 'XGBClassifier',
            # 'LGBMClassifier',
            # 'CatBoostClassifier',
            # 'SVC',
            # 'DecisionTreeClassifier',
            'LogisticRegression',
            # 'BaggingClassifier',
        ]
        # List of meta-models to test
        metaModelName_list = getMetaModel_list(metaModelName_list)

        # ===== 4种配置的集成学习 =====
        configurations = [
            {
                'name': '配置1:  basemodel不蒸馏 + 不校准',
                'models': base_models_without_distillation,
                'calibrate': False,
                'results': (config1_acc, config1_f1, config1_auc, config1_pre, config1_rec)
            },
            {
                'name': '配置2: basemodel不蒸馏 + 校准',
                'models': base_models_without_distillation,
                'calibrate': True,
                'results': (config2_acc, config2_f1, config2_auc, config2_pre, config2_rec)
            },
            {
                'name': '配置3: basemodel蒸馏 + 校准',
                'models': base_models_with_distillation,
                'calibrate': True,
                'results': (config3_acc, config3_f1, config3_auc, config3_pre, config3_rec)
            },
            {
                'name': '配置4: basemodel蒸馏 + 不校准',
                'models': base_models_with_distillation,
                'calibrate': False,
                'results': (config4_acc, config4_f1, config4_auc, config4_pre, config4_rec)
            }
        ]

        for config in configurations:
            print(f"\n【第 {exp + 1} 轮】正在执行 {config['name']} ...")
            
            for meta_model in metaModelName_list:
                try:
                    result = stacking_ensemble_v2(
                        config['models'],
                        train_aug_x, train_aug_y,
                        test_aug_x, test_aug_y,
                        X_val=val_aug_x,
                        y_val=val_aug_y,
                        weight_list=None,
                        weight_handling="ignore",
                        n_folds=5,
                        meta_cv_repeats=3,
                        calibrate_proba=config['calibrate'],
                        calibration_method='sigmoid' if config['calibrate'] else None,
                        calibration_cv=5 if config['calibrate'] else None,
                        verbose=False,
                        random_state=42 + exp
                    )
                except Exception as e:
                    logger.event("ensemble_eval_error", 1)
                    logger.log(
                        stage="ensemble_eval",
                        r=exp + 1,
                        task_type="classification",
                        feature_count=int(train_aug_x.shape[1]),
                        base_model_count=int(len(config['models'])),
                        meta_config={
                            "config_name": config["name"],
                            "calibrate_proba": bool(config["calibrate"]),
                            "meta_model": type(meta_model).__name__,
                        },
                        status="error",
                        error=str(e),
                        test_metric_main="auc",
                        test_score=float("nan"),
                        calibration_check_triggered=1 if config["calibrate"] else 0,
                    )
                    continue
                stacking_metrics = result['stacking_metrics']
                logger.log(
                    stage="ensemble_eval",
                    r=exp + 1,
                    task_type="classification",
                    feature_count=int(train_aug_x.shape[1]),
                    base_model_count=int(len(config['models'])),
                    meta_config={
                        "config_name": config["name"],
                        "calibrate_proba": bool(config["calibrate"]),
                        "meta_model": type(meta_model).__name__,
                    },
                    test_metric_main="auc",
                    test_score=float(stacking_metrics.get("roc_auc", float("nan"))),
                    test_trust_metric="ece" if stacking_metrics.get("ece", None) is not None else None,
                    test_trust=stacking_metrics.get("ece", None),
                    brier=stacking_metrics.get("brier", None),
                    calibration_check_triggered=1 if config["calibrate"] else 0,
                )

                print(f"\n========== 第 {exp + 1} 轮 {config['name']} 结果 ({type(meta_model).__name__}) ==========")
                if result.get('calibration_applied'):
                    print(f"【信息】已应用 {result.get('calibration_method', 'sigmoid')} 概率校准")
                print(f"准确率   : {stacking_metrics['accuracy']:.4f}")
                print(f"F1分数   : {stacking_metrics['f1']:.4f}")
                print(f"AUC      : {stacking_metrics['roc_auc']:.4f}")
                print(f"精确率   : {stacking_metrics['precision']:.4f}")
                print(f"召回率   : {stacking_metrics['recall']:.4f}")
                
                # Save results to corresponding lists
                acc_list, f1_list, auc_list, pre_list, rec_list = config['results']
                acc_list.append(round(stacking_metrics['accuracy'], 5)*100)
                f1_list.append(round(stacking_metrics['f1'], 5)*100)
                auc_list.append(round(stacking_metrics['roc_auc'], 5)*100)
                pre_list.append(round(stacking_metrics['precision'], 5)*100)
                rec_list.append(round(stacking_metrics['recall'], 5)*100)

    # Experiment iteration ended, statistics of ensemble learning results
    print(f"\n\n{'='*80}")
    print(f"{'='*80}")
    print(f"完成 {args.exam_iterations} 轮实验，集成学习性能对比统计")
    print(f"{'='*80}")
    print(f"{'='*80}")
    
    print("\n【配置1: 不蒸馏 + 不校准】")
    print('  准确率: ' + format_mean_std(config1_acc))
    print('  F1分数: ' + format_mean_std(config1_f1))
    print('  AUC  : ' + format_mean_std(config1_auc))
    print('  精确率: ' + format_mean_std(config1_pre))
    print('  召回率: ' + format_mean_std(config1_rec))
    
    print("\n【配置2: 不蒸馏 + 校准】")
    print('  准确率: ' + format_mean_std(config2_acc))
    print('  F1分数: ' + format_mean_std(config2_f1))
    print('  AUC  : ' + format_mean_std(config2_auc))
    print('  精确率: ' + format_mean_std(config2_pre))
    print('  召回率: ' + format_mean_std(config2_rec))
    
    print("\n【配置3: 蒸馏 + 校准】")
    print('  准确率: ' + format_mean_std(config3_acc))
    print('  F1分数: ' + format_mean_std(config3_f1))
    print('  AUC  : ' + format_mean_std(config3_auc))
    print('  精确率: ' + format_mean_std(config3_pre))
    print('  召回率: ' + format_mean_std(config3_rec))
    
    print("\n【配置4: 蒸馏 + 不校准】")
    print('  准确率: ' + format_mean_std(config4_acc))
    print('  F1分数: ' + format_mean_std(config4_f1))
    print('  AUC  : ' + format_mean_std(config4_auc))
    print('  精确率: ' + format_mean_std(config4_pre))
    print('  召回率: ' + format_mean_std(config4_rec))
    
    print(f"\n{'='*80}")

    print("\nValid base model count per round:")
    for idx, cnt in enumerate(models_per_experiment):
        print(f"Round {idx + 1}: {cnt} ")

    all_time_end = time.time()
    all_time = all_time_end - all_time_start
    print(f"Total experiment time: {all_time:.2f} seconds")
    total_experiment_time = all_time
    # 2. Average feature generation time per round
    avg_feat_time = sum(feat_time_list) / len(feat_time_list) if feat_time_list else 0.0
    # 3. Total time excluding feature generation
    total_non_feat_time = total_experiment_time - sum(feat_time_list)
    # 4. Total LLM call + parsing time
    llm_total_time = total_llm_time
    # 5. Remaining steps time (excluding feature generation and LLM time)
    remaining_time = total_non_feat_time - llm_total_time
    
    # Calculate average model count
    avg_model_count = sum(models_per_experiment) / len(models_per_experiment) if models_per_experiment else 0

    # Construct statistics content
    time_stats_content = f"""
=========== 数据集 {ds_name} ===========
超参数优化状态: {'已启用' if args.enable_optimization else '已禁用'}
模型生成反馈状态: {'已启用' if args.enable_feedback else '已禁用'}
综合性能统计报告 ({args.exam_iterations} 轮实验)
=========================================
[时间统计]
1. 实验总运行时间: {total_experiment_time:.2f} 秒
2. 每轮平均特征生成模块时间: {avg_feat_time:.2f} 秒
3. 总LLM调用和解析时间（模型生成 + 超参数优化）: {llm_total_time:.2f} 秒
4. 排除特征生成和LLM时间后的总剩余步骤时间: {remaining_time:.2f} 秒
-----------------------------------------
[模型生成和优化Token统计]
5. 每轮模型生成和优化的token数列表: {token_model_list}
 --所有实验的总token数: {sum(token_model_list)}
 --每轮平均token数: {sum(token_model_list) / len(token_model_list):.2f}
-----------------------------------------
[模型数量统计]
6. 每轮有效基模型数量: {models_per_experiment}
 --每轮平均数量: {avg_model_count:.2f}
-----------------------------------------
每轮特征生成时间详情:
{[f"第 {i+1} 轮: {t:.2f} 秒" for i, t in enumerate(feat_time_list)]}
    """
    print(time_stats_content)
    
    # 保存结果到txt文件
    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = os.path.join(result_dir, f"{ds_name}.txt")
    
    try:
        with open(result_file_path, 'w', encoding='utf-8') as f:
            # 写入集成学习性能对比结果
            f.write("="*80 + "\n")
            f.write("="*80 + "\n")
            f.write(f"集成学习性能对比统计 - {ds_name}\n")
            f.write("="*80 + "\n")
            f.write("="*80 + "\n\n")
            
            f.write("【配置1: 不蒸馏 + 不校准】\n")
            f.write('  准确率: ' + format_mean_std(config1_acc) + '\n')
            f.write('  F1分数: ' + format_mean_std(config1_f1) + '\n')
            f.write('  AUC  : ' + format_mean_std(config1_auc) + '\n')
            f.write('  精确率: ' + format_mean_std(config1_pre) + '\n')
            f.write('  召回率: ' + format_mean_std(config1_rec) + '\n\n')
            
            f.write("【配置2: 不蒸馏 + 校准】\n")
            f.write('  准确率: ' + format_mean_std(config2_acc) + '\n')
            f.write('  F1分数: ' + format_mean_std(config2_f1) + '\n')
            f.write('  AUC  : ' + format_mean_std(config2_auc) + '\n')
            f.write('  精确率: ' + format_mean_std(config2_pre) + '\n')
            f.write('  召回率: ' + format_mean_std(config2_rec) + '\n\n')
            
            f.write("【配置3: 蒸馏 + 校准】\n")
            f.write('  准确率: ' + format_mean_std(config3_acc) + '\n')
            f.write('  F1分数: ' + format_mean_std(config3_f1) + '\n')
            f.write('  AUC  : ' + format_mean_std(config3_auc) + '\n')
            f.write('  精确率: ' + format_mean_std(config3_pre) + '\n')
            f.write('  召回率: ' + format_mean_std(config3_rec) + '\n\n')
            
            f.write("【配置4: 蒸馏 + 不校准】\n")
            f.write('  准确率: ' + format_mean_std(config4_acc) + '\n')
            f.write('  F1分数: ' + format_mean_std(config4_f1) + '\n')
            f.write('  AUC  : ' + format_mean_std(config4_auc) + '\n')
            f.write('  精确率: ' + format_mean_std(config4_pre) + '\n')
            f.write('  召回率: ' + format_mean_std(config4_rec) + '\n\n')
            
            f.write("="*80 + "\n\n")
            f.write(time_stats_content)
        
        print(f"\n结果已保存到: {result_file_path}")
    except Exception as e:
        print(f"\n保存结果文件失败: {e}")
    logger.close()
