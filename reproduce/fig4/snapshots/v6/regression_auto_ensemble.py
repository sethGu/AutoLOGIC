import os
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[4] / "autologic")
sys.path.append(project_root)


from mystage1 import Stage1Classifier  # Automated Feature Engineering for tabular datasets
from mystage1.run_llm_code import run_llm_code
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utils.model_generate import build_prompt_samples, generate_model_2, get_regression_model_prompt
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle
import re
import numpy as np
import random
import argparse
import warnings
import traceback
import copy
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import statistics
import pandas as pd

from utils.ensembleUtils import get_regression_metaModel
from ensemble.regression_ensemble_utils import stacking_regression_util
from utils.utils import format_mean_std,format_mean_std_four
from utils.quant_log import QuantLogger
from utils.csv_dataset_loader import load_csv_dataset
from utils.time_limit import time_limit, TimeoutExceeded

STAGE_TIMEOUT_S = 300

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


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
            df_train[c] = train_col.astype(int)
            df_test[c] = df_test[c].astype(int)
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


# LoadLocal dataset
def load_origin_data(loc, seed):
    if os.path.exists(loc):
        with open(loc, 'rb') as f:
            ds = pickle.load(f)
        target_column_name = ds[4][-1]
        df = ds[1]
        dataset_description = ds[-1]
        df_train, df_test = train_test_split(df, test_size=0.25, random_state=seed)
        df_train, df_test = _encode_non_numeric_features_train_test(df_train, df_test, target_column_name)
        return df_train, df_test, target_column_name, dataset_description

    dataset_name = os.path.splitext(os.path.basename(loc))[0]
    csv_base = os.path.join(project_root, "data", "csv_data")
    df_train, df_test, _, target_column_name, dataset_description = load_csv_dataset(
        dataset_name=dataset_name,
        base_loc=csv_base,
        seed=seed,
        test_size=0.25,
    )
    df_train[target_column_name] = pd.to_numeric(df_train[target_column_name], errors="coerce")
    df_test[target_column_name] = pd.to_numeric(df_test[target_column_name], errors="coerce")
    df_train = df_train[~df_train[target_column_name].isna()].copy()
    df_test = df_test[~df_test[target_column_name].isna()].copy()
    df_train, df_test = _encode_non_numeric_features_train_test(df_train, df_test, target_column_name)
    return df_train, df_test, target_column_name, dataset_description

def base_model(seed):
    rforest = RandomForestRegressor(n_estimators=100, random_state=seed)

    return rforest

def to_pd(df, target_name):
    y = df[target_name]
    x = df.drop(target_name, axis=1)

    return x, y

def code_exec(code):
    try:
        compiled_code = compile(code, "<string>", "exec")
        scope = {"__builtins__": __builtins__}
        exec(compiled_code, scope, scope)
        return None, scope
    except Exception as e:
        print("Code could not be executed:", e)
        return str(e), None

def clean_llm_code(code: str) -> str:
    # Remove ``` code block markers at beginning and end
    code = re.sub(r"^```python\s*", "", code.strip(), flags=re.IGNORECASE)
    code = re.sub(r"```$", "", code.strip())

    # Remove <end> and non-code text (possibly from LLM)
    code = re.sub(r"<end>", "", code)

    # Remove explanation sections or error prompts at text beginning from LLM output
    lines = code.strip().splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("class myregressor") or line.strip().startswith("import") or line.strip().startswith(
                "from"):
            cleaned_lines.append(line)
        elif cleaned_lines:  # If code block recording has started, continue adding subsequent code
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

# 生成特征
def generate_feat(
        base_model,
        df_train,
        df_test,
        dataset_name,
        round_num,
        llm_model='gpt-3.5-turbo',
        iterations=10,
        target_column_name='class',
        dataset_description=None,
        task_type="regression",
        base_url:str=None,
        api_key:str=None,
        logger=None
):
    if base_url is None or api_key is None:
        raise ValueError("base_url and api_key must be provided.")
    stage1_clf = Stage1Classifier(base_classifier=base_model,
                                llm_model=llm_model,
                                iterations=iterations)

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

    df_train_aug = run_llm_code(stage1_clf.code, df_train, target_column_name)
    df_test_aug = run_llm_code(stage1_clf.code, df_test, target_column_name)

    df_train_aug = df_train_aug[stage1_clf.final_columns]
    df_test_aug = df_test_aug[stage1_clf.final_columns]

    return df_train_aug, df_test_aug

def print_stats(name, values):
    print(f"{name}: {np.mean(values):.2f} ± {np.std(values):.2f}")

def print_rmsle(name, values):
    print(f"{name}: {np.mean(values):.4f} ± {np.std(values):.4f}")


# =========================
# 学生网络定义（回归）
# =========================
class StudentNetRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
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
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def regression_kd_loss(student_pred, teacher_pred, y_true, lambda_kd=0.5):
    student_pred = student_pred.view(-1)
    teacher_pred = teacher_pred.view(-1)
    y_true = y_true.view(-1)
    loss_teacher = F.mse_loss(student_pred, teacher_pred)
    loss_true = F.mse_loss(student_pred, y_true)
    return (1 - lambda_kd) * loss_true + lambda_kd * loss_teacher


class DistilledRegressor:
    def __init__(self, model, x_scaler, y_scaler, device):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.device = device
        self.prefit = True

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.model.eval()
        X_scaled = self.x_scaler.transform(X)
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            preds_scaled = self.model(X_tensor).cpu().numpy().reshape(-1, 1)
        preds = self.y_scaler.inverse_transform(preds_scaled).flatten()
        return preds


def distill_to_student_regression(teacher_model, X_train, y_train, X_val, y_val, device='cpu', epochs=30):
    print("    开始回归蒸馏...")

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)

    y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
    y_val_arr = y_val.values if hasattr(y_val, 'values') else y_val
    y_train_arr = np.array(y_train_arr).reshape(-1, 1)
    y_val_arr = np.array(y_val_arr).reshape(-1, 1)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_arr).flatten()
    y_val_scaled = y_scaler.transform(y_val_arr).flatten()

    with torch.no_grad():
        teacher_tr = teacher_model.predict(X_train)
        teacher_va = teacher_model.predict(X_val)

    teacher_tr_scaled = y_scaler.transform(np.array(teacher_tr).reshape(-1, 1)).flatten()
    teacher_va_scaled = y_scaler.transform(np.array(teacher_va).reshape(-1, 1)).flatten()

    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32),
        torch.tensor(teacher_tr_scaled, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val_scaled, dtype=torch.float32),
        torch.tensor(teacher_va_scaled, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    student = StudentNetRegressor(input_dim=X_train_scaled.shape[1], hidden_dims=(256, 128), dropout=0.3).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)

    lambda_kd = 0.5
    best_val_rmse = float('inf')
    best_state = None

    for _ in range(epochs):
        student.train()
        for xb, yb, tb in train_loader:
            xb, yb, tb = xb.to(device), yb.to(device), tb.to(device)
            optimizer.zero_grad()
            pred = student(xb)
            loss = regression_kd_loss(pred, tb, yb, lambda_kd=lambda_kd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        student.eval()
        preds_val = []
        targets_val = []
        with torch.no_grad():
            for xb, yb, tb in val_loader:
                xb = xb.to(device)
                pred = student(xb).cpu().numpy().flatten()
                preds_val.extend(pred)
                targets_val.extend(yb.cpu().numpy().flatten())

        preds_val_real = y_scaler.inverse_transform(np.array(preds_val).reshape(-1, 1)).flatten()
        targets_val_real = y_scaler.inverse_transform(np.array(targets_val).reshape(-1, 1)).flatten()
        rmse = np.sqrt(np.mean((targets_val_real - preds_val_real) ** 2))
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_state = copy.deepcopy(student.state_dict())

    if best_state:
        student.load_state_dict(best_state)

    print(f"    蒸馏完成，学生网络验证集 RMSE: {best_val_rmse:.4f}")
    wrapped = DistilledRegressor(student, x_scaler, y_scaler, device)
    wrapped.distill_val_rmse = float(best_val_rmse)
    wrapped.distill_method = "mse"
    return wrapped


if __name__ == '__main__':
    # Parse command line parameters
    parser = argparse.ArgumentParser(description='Automated ensemble learning framework for regression tasks')
    
    # System configuration parameters
    parser.add_argument('-g', '--gpus', default="0", type=str, help='GPU settings (e.g., "0,1,2")')
    parser.add_argument('-s', '--default_seed', default=42, type=int, help='Random seed (for reproducibility)')
    parser.add_argument('-l', '--llm', default='gpt-3.5-turbo', type=str, 
                        help='Large language model selection (e.g., gpt-3.5-turbo, gpt-4, gpt-4o)')
    # parser.add_argument('-l', '--llm', default='gpt-4o', type=str, help='Large model')
    # Iteration configuration
    parser.add_argument('-e', '--exam_iterations', default=2, type=int, 
                        help='Number of experiment loops (each generates a new set of base models)')
    parser.add_argument('-f', '--feat_iterations', default=1, type=int, 
                        help='Feature engineering iterations (iteration count in stage1)')
    parser.add_argument('-m', '--model_iterations', default=5, type=int, 
                        help='Model generation iterations (how many base models to generate per experiment)')
    parser.add_argument('-p', '--param_iterations', default=1, type=int,
                        help='Hyperparameter optimization iterations (how many parameter optimizations per successful model)')
    # Dataset configuration
    parser.add_argument('-d', '--dataset', default="boston", type=str,
                        help='Dataset selection (e.g., boston, concrete, california, insurance, winequality)')
    # Function switches
    parser.add_argument('--enable_optimization', action='store_true', default=True,
                        help='Whether to enable hyperparameter optimization (True=enable, False=disable)')
    parser.add_argument('--enable_feedback', action='store_true', default=True,
                        help='Whether to enable model generation feedback mechanism (True=enable, False=disable)')
    parser.add_argument('--stage_timeout_s', default=STAGE_TIMEOUT_S, type=int, help='Stage timeout seconds')
    args = parser.parse_args()

    STAGE_TIMEOUT_S = int(args.stage_timeout_s)

    # ============================================================
    # Print runtime configuration information
    # ============================================================
    print("=" * 60)
    print("Automated Ensemble Learning Framework for Regression Tasks - Configuration")
    print("=" * 60)
    print(f"🔧 System Configuration:")
    print(f"   GPU: {args.gpus}")
    print(f"   Random Seed: {args.default_seed}")
    print(f"\n📚 Model Configuration:")
    print(f"   Large Language Model: {args.llm}")
    print(f"\n🔄 Iteration Configuration:")
    print(f"   Experiment Count: {args.exam_iterations}")
    print(f"   Feature Iterations: {args.feat_iterations}")
    print(f"   Model Iterations: {args.model_iterations}")
    print(f"   Hyperparameter Optimization: {args.param_iterations}")
    print(f"\n📊 Dataset: {args.dataset}")
    print(f"\n⚙️  Function Switches:")
    print(f"   Hyperparameter Optimization: {'Enabled' if args.enable_optimization else 'Disabled'}")
    print(f"   Model Generation Feedback: {'Enabled' if args.enable_feedback else 'Disabled'}")
    print(f"\n📊 Dataset: {args.dataset}")
    print(f"\n⚙️  Function Switches:")
    print(f"   Model Generation Feedback: {'Enabled' if args.enable_feedback else 'Disabled'}")
    print("=" * 60)
    print()

    # Model label, globally distinguish LLM-generated models
    model_tab = 1

    # ============================================================
    # OpenAI API Configuration
    # ============================================================
    # Priority: environment variables > manual settings
    # Note: For local execution, please set OPENAI_BASE_URL and OPENAI_API_KEY in environment variables
    #       or manually modify manual_url and manual_key below
    
    env_url = os.getenv("OPENAI_BASE_URL")
    env_key = os.getenv("OPENAI_API_KEY")
    
    # Manual settings (use the following if environment variables are not configured)
    manual_url = ""
    manual_key = "sk-"

    # Selection priority: environment variables > manual settings
    if env_url and env_key:
        base_url = env_url
        api_key = env_key
        print(f"✓ Using API configured from environment variables")
    elif manual_url and manual_key:
        base_url = manual_url
        api_key = manual_key
        print(f"✓ Using manually configured API")
    else:
        raise ValueError(
            "❌ No valid OpenAI API configuration found.\n"
            "Please set environment variables OPENAI_BASE_URL and OPENAI_API_KEY,"
            "or manually configure manual_url and manual_key in the code."
        )

    ds_name = args.dataset 
    print(f"✓ Loading dataset: {ds_name}")
    print()
    print(f"=========== Dataset {ds_name} ===========")
    logger = QuantLogger(
        task="regression",
        dataset=ds_name,
        out_dir=os.path.join(project_root, "result", "logs"),
    )
    # New: List to store results for each experiment
    mae_list, rmse_list, rmsle_list = [], [], []

    loc = f"{project_root}/data/" + ds_name + ".pkl"

    # ===== 4种配置的结果存储（Stacking）=====
    # 配置1：不蒸馏 + 不校准
    config1_mae, config1_rmse, config1_rmsle = [], [], []
    # 配置2：不蒸馏 + 校准
    config2_mae, config2_rmse, config2_rmsle = [], [], []
    config2_cov, config2_width = [], []
    # 配置3：蒸馏 + 校准
    config3_mae, config3_rmse, config3_rmsle = [], [], []
    config3_cov, config3_width = [], []
    # 配置4：蒸馏 + 不校准
    config4_mae, config4_rmse, config4_rmsle = [], [], []

    # Number of base models successfully added to ensemble per round
    models_per_experiment = []

    exam_iter = args.exam_iterations

    # ---------------------- Time & Token Statistics Initialization ----------------------
    all_time_start = time.time()
    feat_time_list = []  # Store feature generation time per round
    token_model_list = []  # Store model generation and optimization token count per round
    total_llm_time = 0.0  # Store total LLM call + parsing time

    # Number of experiments
    for exp in range(exam_iter):
        print(f"=========== Experiment {exp + 1}/{exam_iter} ===========")

        total_token_model_exp = 0  # Store total token count for each experiment

        # List to store results for each experiment
        test_mse_list = []
        test_rmse_list = []
        test_rmsle_list = []

        seed = args.default_seed + exp * 10

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        df_train, df_test, target_column_name, dataset_description = load_origin_data(loc, seed)
        baseline_model = base_model(seed)
        feat_iter = args.feat_iterations

        # Check if feature engineering should be skipped
        if feat_iter > 0:
            feat_start = time.time()
            df_train_aug, df_test_aug = generate_feat(
                base_model=baseline_model,
                df_train=df_train,
                df_test=df_test,
                dataset_name=ds_name,
                round_num=exp + 1,
                llm_model=args.llm,
                iterations=feat_iter,
                target_column_name=target_column_name,
                dataset_description=dataset_description,
                task_type="regression",
                base_url=base_url,
                api_key=api_key,
                logger=logger
            )
            feat_end = time.time()
            feat_elapsed = feat_end - feat_start
            feat_time_list.append(feat_elapsed)
            print(f"Feature generation completed, time elapsed: {feat_elapsed:.2f} seconds")
        else:
            print("Feature engineering skipped (feat_iterations=0)")
            feat_time_list.append(0.0)
            df_train_aug = df_train
            df_test_aug = df_test

        df_train_aug, df_valid_aug = train_test_split(df_train_aug, test_size=0.25, random_state=seed)
        train_aug_x, train_aug_y = to_pd(df_train_aug, target_column_name)
        test_aug_x, test_aug_y = to_pd(df_test_aug, target_column_name)
        val_aug_x, val_aug_y = to_pd(df_valid_aug, target_column_name)
        logger.log(
            stage="feature_summary",
            r=exp + 1,
            task_type="regression",
            feature_count=int(train_aug_x.shape[1]),
            base_model_count=0,
            meta_config={
                "feat_iterations": int(feat_iter),
                "llm": args.llm,
            },
        )

        s = build_prompt_samples(df_train_aug)

        model_prompt = get_regression_model_prompt(
            target_column_name=target_column_name,
            samples=s
        )

        model_messages = [
            {
                "role": "system",
                "content": (
                    "You are a top-level regression algorithm expert.\n"
                    "Your task is to help me iteratively search for the most suitable regression model.\n"
                    "Your primary goal is to minimize the RMSE (Root Mean Squared Error) on the test set.\n"
                    "You must focus on improving RMSE more than any other metric.\n"
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
        best_mse = float("inf")
        best_rmse = float("inf")
        best_rmsle = float("inf")
        best_code = None
        i = 0

        # Base model list for ensemble learning
        base_models_without_distillation = []  # 不蒸馏的模型列表
        base_models_with_distillation = []  # 蒸馏后的模型列表
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
                    task_type="regression",
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
                        res = generate_model_2(args.llm, model_messages, base_url, api_key)
                except TimeoutExceeded as te:
                    logger.event("timeout", 1)
                    logger.end_round(status="skipped", skip_reason="timeout", timeout_stage="llm_model_generate", error=str(te))
                    current_model_retry_count = 0
                    i += 1
                    continue
                code = res['code']
                total_tokens_model = res['total_tokens']
                total_token_model_exp += total_tokens_model
                # Use clean_llm_code function to clean code
                code = clean_llm_code(code)
                model_llm_end = time.time()
                total_llm_time += (model_llm_end - model_llm_start)
                # -----------------------------------------------------------

                # Dynamically modify class name
                new_class_name = f"myregressor_{i + 1}"
                code = re.sub(r'class\s+myregressor[_\w]*\s*(\([^)]*\))?\s*:', f'class {new_class_name}:', code, count=1)
                print(f"----------------------------Original Code-----------------------")
                print(code)
            except Exception as e:
                print("Error in LLM API." + str(e))
                logger.event("llm_api_error", 1)
                continue

            e, exec_scope = code_exec(code)

            if e is not None:
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
                    current_model_retry_count = 0  # Reset counter
                    i += 1  # Skip to next model
                    continue
                
                # Build error feedback
                error_hint = ""
                if "base_estimator" in e:
                    error_hint = "\n**HINT**: The parameter 'base_estimator' is not recognized in sklearn 1.4+. Check the documentation - some classes changed this parameter name. Consider alternatives in the official documentation."
                elif "got an unexpected keyword argument" in e:
                    error_hint = "\n**HINT**: An unexpected keyword argument was encountered. This might be due to sklearn version differences. Verify all parameter names in the official sklearn 1.4+ documentation."
                
                model_messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"""
                        Code execution failed with error: {e}
                        Code: ```python{code}```{error_hint}
                        
                        Retry attempt {current_model_retry_count}/{max_retries_per_model}. Please fix the code and generate the next version:
                        """,
                    },
                ]
                continue

            try:
                # Model instance
                model_class = exec_scope[new_class_name]
                model = model_class()
                model_tab = model_tab + 1
                try:
                    with time_limit(STAGE_TIMEOUT_S):
                        model.fit(train_aug_x, train_aug_y)
                        model_copy = copy.deepcopy(model)  # 在fit之后深拷贝已拟合的模型
                        model_list_append = model_copy
                        pred = model.predict(val_aug_x)
                except TimeoutExceeded as te:
                    logger.event("timeout", 1)
                    logger.end_round(status="skipped", skip_reason="timeout", timeout_stage="base_model_fit_predict", error=str(te))
                    current_model_retry_count = 0
                    i += 1
                    continue
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
                    i += 1  # Skip to next model
                    continue
                
                # Build detailed error feedback
                error_hint = ""
                if "base_estimator" in error_msg:
                    error_hint = "\n**HINT**: The parameter 'base_estimator' is not recognized in sklearn 1.4+. Check the documentation - some classes changed this parameter name. Consider alternatives in the official documentation."
                elif "got an unexpected keyword argument" in error_msg:
                    error_hint = "\n**HINT**: An unexpected keyword argument was encountered. This might be due to sklearn version differences. Verify all parameter names in the official sklearn 1.4+ documentation."
                elif error_type == "KeyError" and "missing_class=" in error_msg:
                    error_hint = "\n**HINT**: You must define a class named `myregressor` (it will be renamed automatically). Output python code only."
                
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
                continue

            # Model executed successfully, reset retry counter
            current_model_retry_count = 0

            # Calculate evaluation metrics
            test_mse = mean_absolute_error(val_aug_y, pred)
            test_rmse = np.sqrt(np.mean((val_aug_y.values - pred) ** 2))
            try:
                y_arr = np.asarray(val_aug_y)
                pred_arr = np.asarray(pred)
                if (y_arr < 0).any():
                    test_rmsle = np.nan
                else:
                    pred_clip = np.maximum(pred_arr, 0)
                    test_rmsle = float(np.sqrt(mean_squared_log_error(y_arr, pred_clip)))
            except Exception:
                test_rmsle = np.nan
            val_rmse = float(test_rmse)
            val_mae = float(test_mse)

            # todo Add parameter optimization after this
            param_best_code = code
            param_best_rmse = test_rmse
            param_best_mae = test_mse

            # =====================================================================
            # Hyperparameter Optimization Iteration
            # =====================================================================
            # Initialize parameter optimization prompt only if optimization is enabled
            if args.enable_optimization:
                param_prompt = f"""
                Here is the best regression model code so far, with its current RMSE score on the validation set:
                
                Current best RMSE: {test_rmse:.4f}
                Current best MAE: {test_mse:.4f}
                
                Model code:
                ```python
{param_best_code}
                ```
                
                Now please **optimize the hyperparameters** of this model to improve its RMSE performance.
                Only modify the hyperparameters, do NOT change the model architecture or class name.
                
                The code should:
                - Be a valid, executable Python regressor with the SAME class name as the original
                - Implement fit(X, y) and predict(X) methods
                - Have the same architecture but with better hyperparameters
                
                Output only the optimized Python code block. No explanation, only code!
                """

                param_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a regression optimization assistant.\n"
                            "Your task is to help me improve the test RMSE of the given regressor\n"
                            "by tuning hyperparameters only. Your answer must contain only executable Python code.\n\n"
                            "**PRIMARY GOAL:** Focus on finding better *combinations* of hyperparameters, not just isolated changes to single parameters.\n\n"
                            "**CRITICAL CONSTRAINT 1: Parallelism**\n"
                            "- **You MUST preserve or add `n_jobs=-1` to ensure multi-core training.**\n"
                            "- Do NOT remove this parameter during optimization.\n\n"
                            "**CRITICAL CONSTRAINT 2: Efficient Search Space**\n"
                            "For ensemble/tree-based models (e.g., RandomForest, XGBoost, LightGBM, CatBoost):\n"
                            "- **n_estimators (or equivalent):** MUST be <= 300 (e.g., 50-300).\n"
                            "- **max_depth (or equivalent):** MUST be <= 10 (e.g., 3-10).\n"
                            "- **num_leaves (for LightGBM):** MUST be <= 64.\n"
                            "Prioritize small, efficient parameter values."
                        )
                    },
                    {
                        "role": "user",
                        "content": param_prompt
                    },
                ]

            # Start parameter optimization iteration, only execute if optimization is enabled
            param_iter_range = range(args.param_iterations) if args.enable_optimization else range(0)
            for p_iter in param_iter_range:
                print(f"++++++ Optimization {p_iter + 1} +++++++++")
                try:
                    # ---------------------- LLM Parameter Optimization Time Statistics ----------------------
                    param_llm_start = time.time()
                    try:
                        with time_limit(STAGE_TIMEOUT_S):
                            res = generate_model_2(args.llm, param_messages, base_url, api_key)
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

                    param_new_class_name = f"myregressor_{i + 1}_param_{p_iter + 1}"
                    # Use regex to match any myregressor class name (including numbers that LLM may modify)
                    param_code = re.sub(
                        r'class\s+myregressor[_\w]*\s*(\([^)]*\))?\s*:', 
                        f'class {param_new_class_name}:',
                        param_code,
                        count=1  # Only replace first match
                    )

                    param_err, param_scope = code_exec(param_code)

                    if param_err is not None:
                        print(f"Hyperparameter optimization code compilation error: {param_err}")
                        param_messages += [
                            {"role": "assistant", "content": param_code},
                            {
                                "role": "user",
                                "content": f"""
                                The optimized regressor code execution failed with compilation error: {param_err}
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
                        myregressor_tuned = model_class()
                        try:
                            with time_limit(STAGE_TIMEOUT_S):
                                myregressor_tuned.fit(train_aug_x, train_aug_y)
                                model_copy = copy.deepcopy(myregressor_tuned)
                                pred_tuned = myregressor_tuned.predict(val_aug_x)
                                rmse_tuned = np.sqrt(np.mean((val_aug_y.values - pred_tuned) ** 2))
                                mae_tuned = mean_absolute_error(val_aug_y, pred_tuned)
                        except TimeoutExceeded:
                            logger.event("timeout", 1)
                            break

                        if rmse_tuned < param_best_rmse:
                            print(f"Parameter optimization improvement: {param_best_rmse:.4f} --> {rmse_tuned:.4f}")
                            param_best_rmse = rmse_tuned
                            param_best_mae = mae_tuned
                            param_best_code = param_code
                            model_list_append = model_copy

                        param_messages += [
                            {"role": "assistant", "content": param_code},
                            {"role": "user",
                             "content": f"Current RMSE: {rmse_tuned:.4f}, Best RMSE: {param_best_rmse:.4f}. Please improve further."},
                        ]

                    except Exception as exec_error:
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
            if best_rmse > param_best_rmse:
                best_rmse = param_best_rmse
                best_code = param_best_code

            # ========== 保存不蒸馏的模型 ==========
            base_models_without_distillation.append(model_list_append)

            # ========== 知识蒸馏：将优化后的模型蒸馏到MLP学生网络 ==========
            print(f"  模型 {i + 1} - 开始知识蒸馏到MLP学生网络...")
            distill_success = False
            distill_val_rmse = None
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                try:
                    with time_limit(STAGE_TIMEOUT_S):
                        student_model = distill_to_student_regression(
                            teacher_model=model_list_append,
                            X_train=train_aug_x,
                            y_train=train_aug_y,
                            X_val=val_aug_x,
                            y_val=val_aug_y,
                            device=device,
                            epochs=30
                        )
                except TimeoutExceeded as te:
                    logger.event("timeout", 1)
                    logger.end_round(status="skipped", skip_reason="timeout", timeout_stage="distill", error=str(te))
                    i += 1
                    continue

                # 保存蒸馏后的学生模型
                base_models_with_distillation.append(student_model)
                distill_success = True
                distill_val_rmse = getattr(student_model, "distill_val_rmse", None)
                print(f"  模型 {i + 1} - 蒸馏成功，使用学生网络作为集成基模型")

            except Exception as distill_error:
                logger.event("distill_error", 1)
                print(f"  模型 {i + 1} - 蒸馏失败: {distill_error}")
                print(f"  模型 {i + 1} - 回退使用原始优化模型")
                # 蒸馏失败，使用原始模型
                base_models_with_distillation.append(model_list_append)

            logger.end_round(
                status="success",
                val_metric_main="rmse",
                val_score=float(param_best_rmse),
                val_mae=float(param_best_mae) if 'param_best_mae' in locals() else None,
                val_rmsle=float(test_rmsle) if 'test_rmsle' in locals() and not np.isnan(test_rmsle) else None,
                best_score_so_far=float(best_rmse) if best_rmse is not None else None,
                retry_count=int(current_model_retry_count),
                llm_tokens=int(total_tokens_model) if 'total_tokens_model' in locals() else None,
                code_len=int(len(param_best_code)) if param_best_code is not None else None,
                decision={
                    "distill_used": True,
                    "distill_success": bool(distill_success),
                },
                distill_val_rmse=distill_val_rmse,
            )

            # Print current experiment detailed results
            print(f"Current experiment result {i + 1}/{args.model_iterations}")
            print(f"Test RMSE: {param_best_rmse:.4f}")
            # Continue while loop
            i = i + 1

            # Append prompt for next round of model generation - only add if feedback is enabled
            if args.enable_feedback and len(code) > 10:
                model_messages += [
                    {"role": "assistant", "content": param_best_code},
                    {
                        "role": "user",
                        "content": f"""
                        ✅ The regressor code executed successfully.

                        📈 Current model RMSE: {param_best_rmse:.4f}
                        🏆 Best historical RMSE so far: {best_rmse:.4f}

                        Please now propose a new regressor that is **more likely to improve the RMSE** on the given test data.
                        The model must differ from all previous ones **by model type or internal structure**.

                        ⚠️ Remember:
                        - You must only output valid Python code for a complete regressor named `myregressor`.
                        - The class must include all imports and implement: `fit` and `predict`.
                        - Do not repeat models you've already used.
                        - Prioritize models that provide reliable and stable predictions.

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
                        ✅ The regressor code executed successfully.

                        Please now propose a new regressor that is **more likely to improve the RMSE** on the given test data.
                        The model must differ from all previous ones **by model type or internal structure**.

                        ⚠️ Remember:
                        - You must only output valid Python code for a complete regressor named `myregressor`.
                        - The class must include all imports and implement: `fit` and `predict`.
                        - Do not repeat models you've already used.
                        - Prioritize models that provide reliable and stable predictions.

                        🎯 Next code block:
                        """,
                    },
                ]

        token_model_list.append(total_token_model_exp)

        # Record the number of valid base models generated in current experiment
        valid_model_count_no_dist = len(base_models_without_distillation)
        valid_model_count_with_dist = len(base_models_with_distillation)
        models_per_experiment.append(valid_model_count_no_dist)
        print(f"Round {exp + 1} valid base model count (no distillation): {valid_model_count_no_dist}, (with distillation): {valid_model_count_with_dist}")



        """
        Ensemble Learning - 4种配置对比（回归 Stacking）
        """
        # List of meta-model names to test
        metaModelName_list = [
            # 'RandomForestRegressor',
            # 'XGBRegressor',
            # 'LGBMRegressor',-----
            # 'CatBoostRegressor',------
            # 'SVR',
            # 'DecisionTreeRegressor',-----
            'LinearRegression',
            # 'Ridge',-------
            # 'Lasso',-------
            # 'ElasticNet',
            # 'MLPRegressor',
            # 'KNeighborsRegressor',
            # 'BaggingRegressor',
        ]

        # List of meta-models to test
        metaModelName_list = get_regression_metaModel(metaModelName_list)

        configurations = [
            {
                'name': '配置1: basemodel不蒸馏 + 不校准',
                'models': base_models_without_distillation,
                'calibrate': False,
                'results': (config1_mae, config1_rmse, config1_rmsle)
            },
            {
                'name': '配置2: basemodel不蒸馏 + 校准',
                'models': base_models_without_distillation,
                'calibrate': True,
                'results': (config2_mae, config2_rmse, config2_rmsle)
            },
            {
                'name': '配置3: basemodel蒸馏 + 校准',
                'models': base_models_with_distillation,
                'calibrate': True,
                'results': (config3_mae, config3_rmse, config3_rmsle)
            },
            {
                'name': '配置4: basemodel蒸馏 + 不校准',
                'models': base_models_with_distillation,
                'calibrate': False,
                'results': (config4_mae, config4_rmse, config4_rmsle)
            }
        ]

        for config in configurations:
            print(f"\n【第 {exp + 1} 轮】正在执行 {config['name']} ...")
            for meta_model in metaModelName_list:
                try:
                    with time_limit(STAGE_TIMEOUT_S):
                        result = stacking_regression_util(
                            config['models'],
                            meta_model,
                            train_aug_x,
                            train_aug_y,
                            test_aug_x,
                            test_aug_y,
                            X_val=val_aug_x,
                            y_val=val_aug_y,
                            calibrate=config['calibrate'],
                            alpha=0.1,
                            verbose=True,
                            random_state=42 + exp
                        )
                except TimeoutExceeded:
                    logger.event("timeout", 1)
                    logger.log(
                        stage="ensemble_eval",
                        r=exp + 1,
                        task_type="regression",
                        feature_count=int(train_aug_x.shape[1]),
                        base_model_count=int(len(config['models'])),
                        meta_config={
                            "config_name": config["name"],
                            "calibrate": bool(config["calibrate"]),
                            "meta_model": type(meta_model).__name__,
                        },
                        status="skipped",
                        skip_reason="timeout",
                        timeout_stage="ensemble_eval",
                    )
                    continue
                except Exception as e:
                    logger.event("runtime_error", 1)
                    logger.log(
                        stage="ensemble_eval",
                        r=exp + 1,
                        task_type="regression",
                        feature_count=int(train_aug_x.shape[1]),
                        base_model_count=int(len(config['models'])),
                        meta_config={
                            "config_name": config["name"],
                            "calibrate": bool(config["calibrate"]),
                            "meta_model": type(meta_model).__name__,
                        },
                        status="skipped",
                        skip_reason="runtime_error",
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    continue
                stacking_metrics = result['stacking_metrics']
                logger.log(
                    stage="ensemble_eval",
                    r=exp + 1,
                    task_type="regression",
                    feature_count=int(train_aug_x.shape[1]),
                    base_model_count=int(len(config['models'])),
                    meta_config={
                        "config_name": config["name"],
                        "calibrate": bool(config["calibrate"]),
                        "meta_model": type(meta_model).__name__,
                    },
                    test_metric_main="rmse",
                    test_score=float(stacking_metrics.get("rmse", float("nan"))),
                    test_trust_metric="coverage" if stacking_metrics.get("coverage", None) is not None else None,
                    test_trust=stacking_metrics.get("coverage", None),
                    width=stacking_metrics.get("width", None),
                    calibration_check_triggered=1 if config["calibrate"] else 0,
                )

                print(f"\n=========== 第{exp + 1}次Stacking集成结果--{type(meta_model).__name__} ({config['name']}) ===========")
                print(f"MAE : {stacking_metrics['mae']:.4f}")
                print(f"RMSE : {stacking_metrics['rmse']:.4f}")
                print(f"RMSLE: {stacking_metrics['rmsle']:.4f}")
                if config['calibrate'] and 'coverage' in stacking_metrics:
                    print(f"Coverage: {stacking_metrics['coverage']:.4f}")
                    print(f"Width   : {stacking_metrics['width']:.4f}")

                # Save results to corresponding lists
                mae_list, rmse_list, rmsle_list = config['results']
                mae_list.append(round(stacking_metrics['mae'], 4))
                rmse_list.append(round(stacking_metrics['rmse'], 4))
                rmsle_list.append(round(stacking_metrics['rmsle'], 4))

                if config['calibrate'] and 'coverage' in stacking_metrics:
                    if config['name'].startswith('配置2'):
                        config2_cov.append(stacking_metrics['coverage'])
                        config2_width.append(stacking_metrics['width'])
                    if config['name'].startswith('配置3'):
                        config3_cov.append(stacking_metrics['coverage'])
                        config3_width.append(stacking_metrics['width'])

    # Experiment iteration ended, statistics of ensemble learning results
    print(f"\n\n{'=' * 80}")
    print(f"{'=' * 80}")
    print(f"完成 {args.exam_iterations} 轮实验，集成学习性能对比统计")
    print(f"{'=' * 80}")
    print(f"{'=' * 80}")

    print("\n【配置1: 不蒸馏 + 不校准】")
    print('  MAE  : ' + format_mean_std(config1_mae))
    print('  RMSE : ' + format_mean_std(config1_rmse))
    print('  RMSLE: ' + format_mean_std_four(config1_rmsle))

    print("\n【配置2: 不蒸馏 + 校准】")
    print('  MAE  : ' + format_mean_std(config2_mae))
    print('  RMSE : ' + format_mean_std(config2_rmse))
    print('  RMSLE: ' + format_mean_std_four(config2_rmsle))
    print('  Coverage: ' + format_mean_std(config2_cov) if config2_cov else '  Coverage: N/A')
    print('  Width: ' + format_mean_std(config2_width) if config2_width else '  Width: N/A')

    print("\n【配置3: 蒸馏 + 校准】")
    print('  MAE  : ' + format_mean_std(config3_mae))
    print('  RMSE : ' + format_mean_std(config3_rmse))
    print('  RMSLE: ' + format_mean_std_four(config3_rmsle))
    print('  Coverage: ' + format_mean_std(config3_cov) if config3_cov else '  Coverage: N/A')
    print('  Width: ' + format_mean_std(config3_width) if config3_width else '  Width: N/A')

    print("\n【配置4: 蒸馏 + 不校准】")
    print('  MAE  : ' + format_mean_std(config4_mae))
    print('  RMSE : ' + format_mean_std(config4_rmse))
    print('  RMSLE: ' + format_mean_std_four(config4_rmsle))

    print(f"\n{'=' * 80}")

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
            f.write("=" * 80 + "\n")
            f.write("=" * 80 + "\n")
            f.write(f"集成学习性能对比统计 - {ds_name}\n")
            f.write("=" * 80 + "\n")
            f.write("=" * 80 + "\n\n")

            f.write("【配置1: 不蒸馏 + 不校准】\n")
            f.write('  MAE  : ' + format_mean_std(config1_mae) + '\n')
            f.write('  RMSE : ' + format_mean_std(config1_rmse) + '\n')
            f.write('  RMSLE: ' + format_mean_std_four(config1_rmsle) + '\n\n')

            f.write("【配置2: 不蒸馏 + 校准】\n")
            f.write('  MAE  : ' + format_mean_std(config2_mae) + '\n')
            f.write('  RMSE : ' + format_mean_std(config2_rmse) + '\n')
            f.write('  RMSLE: ' + format_mean_std_four(config2_rmsle) + '\n')
            f.write('  Coverage: ' + (format_mean_std(config2_cov) if config2_cov else 'N/A') + '\n')
            f.write('  Width: ' + (format_mean_std(config2_width) if config2_width else 'N/A') + '\n\n')

            f.write("【配置3: 蒸馏 + 校准】\n")
            f.write('  MAE  : ' + format_mean_std(config3_mae) + '\n')
            f.write('  RMSE : ' + format_mean_std(config3_rmse) + '\n')
            f.write('  RMSLE: ' + format_mean_std_four(config3_rmsle) + '\n')
            f.write('  Coverage: ' + (format_mean_std(config3_cov) if config3_cov else 'N/A') + '\n')
            f.write('  Width: ' + (format_mean_std(config3_width) if config3_width else 'N/A') + '\n\n')

            f.write("【配置4: 蒸馏 + 不校准】\n")
            f.write('  MAE  : ' + format_mean_std(config4_mae) + '\n')
            f.write('  RMSE : ' + format_mean_std(config4_rmse) + '\n')
            f.write('  RMSLE: ' + format_mean_std_four(config4_rmsle) + '\n\n')

            f.write("=" * 80 + "\n\n")
            f.write(time_stats_content)

        print(f"\n结果已保存到: {result_file_path}")
    except Exception as e:
        print(f"\n保存结果文件失败: {e}")
    logger.close()
