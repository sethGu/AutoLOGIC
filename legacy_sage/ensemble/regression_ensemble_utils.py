from sklearn.model_selection import KFold
import numpy as np
import copy
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error

def stacking_regression_util(base_models, meta_model, X, y, X_test, y_test,
                             n_folds=5, verbose=True, random_state=42):
    """
    使用 deepcopy 解决模型克隆问题的版本。
    ✔ 每折使用全新 deepcopy 模型
    ✔ 测试集使用 full-fit deepcopy 模型
    """

    X_train = X
    y_train = y

    n_models = len(base_models)
    train_meta = np.zeros((X_train.shape[0], n_models))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # full-fit 模型保存（用于测试集预测）
    full_fit_models = []

    # ---------------------------------------------------------
    # Step 1: 基模型 OOF 预测
    # ---------------------------------------------------------
    for model_idx, model in enumerate(base_models):
        if verbose:
            print(f"\nTraining base model {model_idx + 1}/{n_models}")

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            if verbose:
                print(f"  Fold {fold_idx + 1}/{n_folds}")

            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val   = X_train.iloc[val_idx]

            # ⭐ 使用 deepcopy 克隆模型
            m = copy.deepcopy(model)
            m.fit(X_fold_train, y_fold_train)

            val_preds = m.predict(X_fold_val)
            train_meta[val_idx, model_idx] = val_preds

        # ⭐ full-fit 模型（用于最终 test 集预测）
        full_model = copy.deepcopy(model)
        full_model.fit(X_train, y_train)
        full_fit_models.append(full_model)

    # ---------------------------------------------------------
    # Step 2: 训练元模型
    # ---------------------------------------------------------
    if verbose:
        print("\nTraining meta model...")

    meta_model.fit(train_meta, y_train)

    # ---------------------------------------------------------
    # Step 3: 使用 full-fit 模型预测 test_meta
    # ---------------------------------------------------------
    test_meta = np.zeros((X_test.shape[0], n_models))

    for model_idx, full_model in enumerate(full_fit_models):
        test_meta[:, model_idx] = full_model.predict(X_test)

    final_preds = meta_model.predict(test_meta)

    # ---------------------------------------------------------
    # Step 4: 评估
    # ---------------------------------------------------------
    rmse = np.sqrt(np.mean((y_test.values - final_preds) ** 2))
    mae = mean_absolute_error(y_test, final_preds)
    r2 = r2_score(y_test, final_preds)

    try:
        if np.any(y_test < 0) or np.any(final_preds < 0):
            rmsle = np.nan
            if verbose:
                print("⚠️ Warning: Negative values detected, RMSLE undefined.")
        else:
            rmsle = mean_squared_log_error(y_test, final_preds) ** 0.5
    except ValueError:
        rmsle = np.nan
        if verbose:
            print("⚠️ RMSLE calculation failed due to invalid values.")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmsle": rmsle
    }

    if verbose:
        print("\n✅ Evaluation Results:")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.4f}" if not np.isnan(v) else f"{k.upper()}: NaN")

    return {
        "full_fit_models": full_fit_models,
        "meta_model": meta_model,
        "stacking_metrics": metrics,
        "final_predictions": final_preds
    }



def stacking_regression_util_1(base_models, meta_model, X, y, X_test, y_test, n_folds=5, verbose=True, random_state=42):
    """
    执行 Stacking 回归集成训练和评估，支持 RMSE、MAE、R2 和 RMSLE。

    参数：
        base_models: list，基回归模型实例列表
        meta_model: sklearn 风格的回归模型
        X, y: pandas.DataFrame / Series，训练数据
        X_test, y_test: pandas.DataFrame / Series，测试数据
        n_folds: int，K 折交叉验证折数
        verbose: bool，是否打印训练日志
        random_state: int，随机种子

    返回：
        dict，包含训练好的元模型和回归指标
    """

    X_train = X
    y_train = y

    n_models = len(base_models)
    train_meta = np.zeros((X_train.shape[0], n_models))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Step 1: 一级模型训练 + 生成 meta 特征
    for model_idx, model in enumerate(base_models):
        if verbose:
            print(f"\nTraining base model {model_idx + 1}/{n_models}")

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]

            model.fit(X_fold_train, y_fold_train)
            val_preds = model.predict(X_fold_val)

            train_meta[val_idx, model_idx] = val_preds

    # Step 2: 元模型训练
    if verbose:
        print("\nTraining meta model...")
    meta_model.fit(train_meta, y_train)

    # Step 3: 测试集预测
    test_meta = np.zeros((X_test.shape[0], n_models))
    for model_idx, model in enumerate(base_models):
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        test_meta[:, model_idx] = test_preds

    final_preds = meta_model.predict(test_meta)

    # Step 4: 评估
    # rmse = mean_squared_error(y_test, final_preds, squared=False)   这行代码会报错，替换成下面手动计算的形式
    rmse = np.sqrt(np.mean((y_test.values - final_preds) ** 2))

    mae = mean_absolute_error(y_test, final_preds)
    r2 = r2_score(y_test, final_preds)

    # 安全计算 RMSLE（避免 log(0)）
    try:
        if np.any(y_test < 0) or np.any(final_preds < 0):
            rmsle = np.nan
            if verbose:
                print("⚠️ Warning: Negative values detected, RMSLE undefined.")
        else:
            rmsle = mean_squared_log_error(y_test, final_preds) ** 0.5
    except ValueError:
        rmsle = np.nan
        if verbose:
            print("⚠️ RMSLE calculation failed due to invalid values.")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmsle": rmsle
    }

    if verbose:
        print("\n✅ Evaluation Results:")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.4f}" if not np.isnan(v) else f"{k.upper()}: NaN")

    return {
        "meta_model": meta_model,
        "stacking_metrics": metrics
    }




def voting_regression_util(fitted_models, X_test, y_test, verbose=True):
    """
    执行 Voting（Averaging）回归集成预测和评估，支持 RMSE、MAE、R2 和 RMSLE。

    参数：
        fitted_models: list，已训练好的回归模型
        X_test: pandas.DataFrame，测试特征
        y_test: pandas.Series，测试标签
        verbose: bool，是否打印评估日志

    返回：
        dict，包含预测结果和评估指标
    """

    if not fitted_models:
        raise ValueError("❌ 模型列表不能为空")

    # 所有模型的预测结果收集
    all_preds = []
    for i, model in enumerate(fitted_models):
        if not hasattr(model, "predict"):
            raise AttributeError(f"模型 {i} 不支持 predict() 方法")
        pred = model.predict(X_test)
        all_preds.append(pred)

    # 平均预测结果作为 Voting 预测值
    all_preds = np.array(all_preds)  # shape: (n_models, n_samples)
    final_preds = np.mean(all_preds, axis=0)

    # 评估指标
    rmse = np.sqrt(np.mean((y_test.values - final_preds) ** 2))
    mae = mean_absolute_error(y_test, final_preds)
    r2 = r2_score(y_test, final_preds)

    try:
        if np.any(y_test < 0) or np.any(final_preds < 0):
            rmsle = np.nan
            if verbose:
                print("⚠️ Warning: Negative values detected, RMSLE undefined.")
        else:
            rmsle = mean_squared_log_error(y_test, final_preds) ** 0.5
    except ValueError:
        rmsle = np.nan
        if verbose:
            print("⚠️ RMSLE calculation failed due to invalid values.")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmsle": rmsle
    }

    if verbose:
        print("\n✅ Evaluation Results (Voting):")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.4f}" if not np.isnan(v) else f"{k.upper()}: NaN")

    return {
        "avg_pred": final_preds,
        "voting_metrics": metrics
    }



def bagging_regression_util(fitted_models, X_test, y_test, verbose=True):
    """
    执行 Bagging 回归集成预测和评估，传入已训练好的模型。
    支持 RMSE、MAE、R2 和 RMSLE。

    参数：
        fitted_models: list，多个已训练好的回归模型
        X_test: pandas.DataFrame，测试特征
        y_test: pandas.Series，测试标签
        verbose: bool，是否打印日志

    返回：
        dict，包含平均预测值和评估指标
    """

    if not fitted_models:
        raise ValueError("❌ 模型列表不能为空")

    all_preds = []
    for i, model in enumerate(fitted_models):
        if not hasattr(model, "predict"):
            raise AttributeError(f"模型 {i} 不支持 predict() 方法")
        pred = model.predict(X_test)
        all_preds.append(pred)

    all_preds = np.array(all_preds)  # shape: (n_models, n_samples)
    final_preds = np.mean(all_preds, axis=0)

    # ================== 性能评估 ==================
    rmse = np.sqrt(np.mean((y_test.values - final_preds) ** 2))
    mae = mean_absolute_error(y_test, final_preds)
    r2 = r2_score(y_test, final_preds)

    try:
        if np.any(y_test < 0) or np.any(final_preds < 0):
            rmsle = np.nan
            if verbose:
                print("⚠️ Warning: Negative values detected, RMSLE undefined.")
        else:
            rmsle = mean_squared_log_error(y_test, final_preds) ** 0.5
    except ValueError:
        rmsle = np.nan
        if verbose:
            print("⚠️ RMSLE calculation failed due to invalid values.")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmsle": rmsle
    }

    if verbose:
        print("\n✅ Evaluation Results (Bagging):")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v:.4f}" if not np.isnan(v) else f"{k.upper()}: NaN")

    return {
        "avg_pred": final_preds,
        "bagging_metrics": metrics
    }