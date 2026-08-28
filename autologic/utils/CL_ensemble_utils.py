from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import NotFittedError



def stacking_ensemble_multiclass(base_models, meta_model, X_train, y_train, X_test, y_test, n_folds=5, verbose=True,
                                 random_state=42):
    """
    多分类 Stacking 集成学习器，自动跳过不支持 predict_proba 的模型。

    参数：
        base_models: list，一级模型列表（已实例化）
        meta_model: sklearn 模型，二级融合模型（已实例化）
        X_train: DataFrame，训练特征
        y_train: Series，训练标签
        X_test: DataFrame，测试特征
        y_test: Series，测试标签
        n_folds: int，交叉验证折数
        verbose: bool，是否打印进度
        random_state: int，随机种子

    返回：
        dict，包括：
            - 'meta_model': 训练好 of 融合模型
            - 'preds': 最终预测标签
            - 'probas': 最终预测概率
            - 'metrics': 多分类评估指标
    """

    classes = np.unique(y_train)
    n_classes = len(classes)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    usable_models = []
    usable_model_names = []

    meta_features_train_list = []

    for model_idx, model in enumerate(base_models):
        model_name = type(model).__name__
        if verbose:
            print(f"\n📚 Training base model {model_idx + 1}/{len(base_models)} ({model_name})")

        meta_feat = np.zeros((X_train.shape[0], n_classes))
        skip_model = False

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            try:
                model.fit(X_fold_train, y_fold_train)
                val_proba = model.predict_proba(X_fold_val)
                meta_feat[val_idx, :] = val_proba
            except (AttributeError, NotFittedError, ValueError) as e:
                print(f"⚠️ Skipping model {model_name}: {e}")
                skip_model = True
                break

        if not skip_model:
            usable_models.append(model)
            usable_model_names.append(model_name)
            meta_features_train_list.append(meta_feat)

    if len(usable_models) == 0:
        raise ValueError("❌ No base model supports `predict_proba`. Please check your base model list.")

    # 拼接一级模型输出作as二级训练特征
    meta_features_train = np.concatenate(meta_features_train_list, axis=1)

    if verbose:
        print("\n🔧 Training meta model...")
    meta_model.fit(meta_features_train, y_train)

    # Test集一级模型输出


    meta_features_test_list = []
    for model_idx, model in enumerate(usable_models):
        model_name = usable_model_names[model_idx]
        try:
            model.fit(X_train, y_train)
            test_proba = model.predict_proba(X_test)
            meta_features_test_list.append(test_proba)
        except Exception as e:
            print(f"⚠️ Test-time model {model_name} failed: {e}")
            raise

    meta_features_test = np.concatenate(meta_features_test_list, axis=1)

    # 二级模型预测
    final_preds = meta_model.predict(meta_features_test)
    final_probas = meta_model.predict_proba(meta_features_test)

    # 标签 binarize use于多类 AUC
    y_test_binarized = label_binarize(y_test, classes=classes)

    # 多分类评估
    metrics = {
        "accuracy": accuracy_score(y_test, final_preds),
        "f1": f1_score(y_test, final_preds, average='macro'),
        "precision": precision_score(y_test, final_preds, average='macro'),
        "recall": recall_score(y_test, final_preds, average='macro'),
        "auc": roc_auc_score(y_test_binarized, final_probas, multi_class='ovr', average='macro')
    }

    if verbose:
        print("\n✅ Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    return {
        "meta_model": meta_model,
        "preds": final_preds,
        "probas": final_probas,
        "metrics": metrics
    }



def voting_ensemble(fitted_models, test_x, test_y=None, threshold=0.5):
    """
    多模型软投票预测与评估。
    所有模型的 predict_proba 必须返回一维正类概率。
    """
    if not fitted_models:
        raise ValueError("模型列表not能as空")

    probas = []
    for i, model in enumerate(fitted_models):
        if not hasattr(model, "predict_proba"):
            raise AttributeError(f"模型 {i} not支持 predict_proba()")
        proba = model.predict_proba(test_x)

        if proba.ndim != 1:
            raise ValueError(f"模型 {i}  of 输出notis一维正类概率，实际 shape: {proba.shape}")
        if np.any(np.isnan(proba)):
            raise ValueError(f"模型 {i} 输出in存at NaN")
        probas.append(proba)

    avg_proba = np.mean(probas, axis=0)

    pred = (avg_proba >= threshold).astype(int)

    metrics = {}
    if test_y is not None:
        acc = accuracy_score(test_y, pred)
        per = precision_score(test_y, pred, zero_division=0)
        rec = recall_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        auc = roc_auc_score(test_y, avg_proba)
        metrics = {
            "accuracy": acc,
            "f1": f1,
            "precision": per,
            "recall": rec,
            "auc": auc
        }

    return {
        "metrics": metrics,
        "proba": avg_proba,
        "pred": pred
    }

def multiclass_voting_ensemble(fitted_models, test_x, test_y=None, average='macro'):
    """
    多分类任务的 Soft Voting 集成方法。

    参数：
        fitted_models: list，已经训练好的支持 predict_proba 的模型
        test_x: array-like，测试特征
        test_y: array-like，测试标签（可选）
        average: str，use于多分类评估指标 of 平均方式（默认'macro'）

    返回：
        dict，包括预测标签、平均概率以及多分类评估指标（如 test_y 提供）
    """
    if not fitted_models:
        raise ValueError("模型列表not能as空")

    probas = []
    for i, model in enumerate(fitted_models):
        if not hasattr(model, "predict_proba"):
            raise AttributeError(f"模型 {i} not支持 predict_proba()")
        proba = model.predict_proba(test_x)

        if proba.ndim != 2:
            raise ValueError(f"模型 {i}  of  predict_proba 输出应as二维，实际as shape={proba.shape}")
        if np.any(np.isnan(proba)):
            raise ValueError(f"模型 {i} 输出in存at NaN")
        probas.append(proba)

    avg_proba = np.mean(probas, axis=0)
    pred = np.argmax(avg_proba, axis=1)

    metrics = {}
    if test_y is not None:
        classes = np.unique(test_y)
        y_test_bin = label_binarize(test_y, classes=classes)
        metrics = {
            "accuracy": accuracy_score(test_y, pred),
            "f1": f1_score(test_y, pred, average=average),
            "precision": precision_score(test_y, pred, average=average),
            "recall": recall_score(test_y, pred, average=average),
            "auc": roc_auc_score(y_test_bin, avg_proba, multi_class='ovr', average=average)
        }

    return {
        "metrics": metrics,
        "proba": avg_proba,
        "pred": pred
    }


def get_classification_param_prompt(
    best_code, best_auc, dataset_description,
    X_test, feature_columns, dataset_name=None, max_rows=10
):
    """
    生成适用于 LLM 分类模型参数优化的提示词
    """
    import pandas as pd
    if isinstance(X_test, pd.DataFrame):
        df_show = X_test.copy()
    else:
        df_show = pd.DataFrame(X_test, columns=feature_columns)

    table = df_show.head(max_rows).to_string(index=False)
    data_shape = df_show.shape if hasattr(df_show, 'shape') else (len(df_show), len(feature_columns))
    if dataset_name is None:
        dataset_name = "unknown"

    prompt = (
        f"Here is the best classification model code so far, with its current AUC score on the test set:\n\n"
        f"Current best AUC: {best_auc:.4f}\n\n"
        f"Model code:\n"
        f"```python\n{best_code}\n```\n"
        f"The downstream classification task is based on the following dataset.\n"
        f"Dataset name: {dataset_name}\n"
        f"Dataset description:\n{dataset_description}\n\n"
        # f"Feature names:\n{', '.join(feature_columns)}\n\n"
        # f"Test set shape: {data_shape}\n"
        # f"Here are the first {max_rows} rows of the test set:\n{table}\n\n"
        "Please ONLY optimize the hyperparameters\n"
        "in the given classification model code to further improve the AUC value.\n"
        "DO NOT change the algorithm type or model structure.\n"
        "Output only a new optimized Python code block.\n"
        "No explanation, only code!\n"
        "IMPORTANT CONTEXT:\n"
        "You are writing classification model code in Python using scikit-learn version 1.6.1.\n"
        "STRICT REQUIREMENT:\n"
        "ONLY use parameters that are supported by scikit-learn version 1.6.1.\n"
        "DO NOT use any parameters that are deprecated or only available in versions prior to 1.2.\n"
        "Refer ONLY to the scikit-learn 1.6.1 documentation for valid parameters and their default values."
    )

    return prompt
