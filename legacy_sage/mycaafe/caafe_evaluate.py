import copy
import pandas as pd
import numpy as np
from .data import get_X_y
from .preprocessing import make_datasets_numeric, make_dataset_numeric
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif, mutual_info_regression
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

def filter_generated_features(df, new_feature_cols, target_col, task_type):
    """
    根据统计检验和模型特征重要性筛选由LLM生成的新特征列。
    参数:
        df (pd.DataFrame): 包含特征和目标的 DataFrame
        new_feature_cols (list): 新生成特征的列名列表
        target_col (str): 目标列的名称
        task_type (str): 任务类型，'classification'或'regression'或‘clustering’
    返回:
        list: 保留下来的新特征列名列表
    """
    X_new = df[new_feature_cols]  # 新生成的特征子表
    # 第一步：统计检验筛选
    if task_type == 'classification':
        y = df[target_col]  # 目标变量
        # 对于分类任务，使用互信息法评估特征与目标的相关性
        # k='all' 保留所有特征仅计算分数
        selector = SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=0), k='all')
        selector.fit(X_new, y)
        # 每个特征的互信息分数（分数越高相关性越大）
        stat_scores = selector.scores_
        # 根据互信息分数选择排名靠前的特征 (例如保留前 N 个特征)
        top_k = min(7, len(new_feature_cols))
        # 按分数从高到低排序，取前 top_k 个索引
        top_indices = np.argsort(stat_scores)[::-1][:top_k]
        # 对应的特征名称列表
        stat_selected_features = [new_feature_cols[i] for i in top_indices]
        # 分类模型，使用100棵树的LightGBM
        model = LGBMClassifier(n_estimators=100, random_state=0,verbosity=-1)
        model.fit(X_new, y)
        # 提取每个特征的特征重要性分数
        importances = model.feature_importances_
        # 将特征重要性转换为 DataFrame 便于处理
        importance_df = pd.DataFrame({'feature': new_feature_cols, 'importance': importances})
        # 根据特征重要性选择排名靠前的特征 (例如保留前 N 个特征)
        model_selected_features = list(importance_df.sort_values('importance', ascending=False)['feature'][:top_k])
        # 综合统计检验和模型评估结果，取交集保留最终特征
        final_features = [f for f in stat_selected_features if f in model_selected_features]

        return final_features

    elif task_type == 'regression':
        y = df[target_col]
        # 对于回归任务，使用 F 检验（基于 Pearson 相关系数的方差分析）评估特征与目标的相关性
        selector = SelectKBest(f_regression, k='all')
        selector.fit(X_new, y)
        p_values = selector.pvalues_  # 每个特征的 p 值（显著性水平）
        # 筛选与目标显著相关的特征（例如 p 值小于 0.05）
        stat_selected_features = [new_feature_cols[i] for i, p in enumerate(p_values) if p < 0.05]
        # 若没有特征通过显著性检验，则选择 p 值最小（相关性最高）的特征
        if len(stat_selected_features) == 0 and len(new_feature_cols) > 0:
            stat_selected_features = [new_feature_cols[np.argmin(p_values)]]
        # 回归模型，使用100棵树的LightGBM
        model = LGBMRegressor(n_estimators=100, random_state=0,verbosity=-1)
        model.fit(X_new, y)
        # 提取每个特征的特征重要性分数
        importances = model.feature_importances_
        # 将特征重要性转换为 DataFrame 便于处理
        importance_df = pd.DataFrame({'feature': new_feature_cols, 'importance': importances})
        # 按重要性由高到低排序，根据特征重要性选择排名靠前的特征 (例如保留前 N 个特征)
        model_selected_features = list(importance_df.sort_values('importance', ascending=False)['feature'][:7])
        # 综合统计检验和模型评估结果，取交集保留最终特征
        final_features = [f for f in stat_selected_features if f in model_selected_features]

        return final_features

    elif task_type == 'clustering':
        # Step 1: 方差过滤（剔除变化太小的特征）
        variance_threshold = 1e-3
        retained_feats = [f for f in new_feature_cols if df[f].var() > variance_threshold]
        if len(retained_feats) == 0:
            return []

        # Step 2: 单特征聚类 -> 计算 silhouette score（结构性评分）
        silhouette_scores = {}
        for feat in retained_feats:
            X_feat = df[[feat]].dropna()
            if X_feat[feat].nunique() < 2:
                continue
            try:
                labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X_feat)
                score = silhouette_score(X_feat, labels)
                silhouette_scores[feat] = score
            except Exception:
                continue

        # 选择评分最高的前 N 个特征
        N = min(5, len(silhouette_scores))  # 最多保留5个
        selected_features = sorted(silhouette_scores, key=silhouette_scores.get, reverse=True)[:N]
        return selected_features

    else:
        raise ValueError("task_type 参数必须是 'classification'、'regression' 或 'clustering'")

def evaluate_dataset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    prompt_id,
    name,
    method,
    metric_used,
    target_name,
    max_time=300,
    seed=0,
):
    df_train, df_test = copy.deepcopy(df_train), copy.deepcopy(df_test)
    df_train, _, mappings = make_datasets_numeric(
        df_train, None, target_name, return_mappings=True
    )
    df_test = make_dataset_numeric(df_test, mappings=mappings)

    if df_test is not None:
        test_x, test_y = get_X_y(df_test, target_name=target_name)

    x, y = get_X_y(df_train, target_name=target_name)
    feature_names = list(df_train.drop(target_name, axis=1).columns)

    np.random.seed(0)
    if method == "autogluon" or method == "autosklearn2":
        if method == "autogluon":
            from tabpfn.scripts.tabular_baselines import autogluon_metric

            clf = autogluon_metric
        elif method == "autosklearn2":
            from tabpfn.scripts.tabular_baselines import autosklearn2_metric

            clf = autosklearn2_metric
        metric, ys, res = clf(
            x, y, test_x, test_y, feature_names, metric_used, max_time=max_time
        )  #
    elif type(method) == str:
        if method == "gp":
            from tabpfn.scripts.tabular_baselines import gp_metric

            clf = gp_metric
        elif method == "knn":
            from tabpfn.scripts.tabular_baselines import knn_metric

            clf = knn_metric
        elif method == "xgb":
            from tabpfn.scripts.tabular_baselines import xgb_metric

            clf = xgb_metric
        elif method == "catboost":
            from tabpfn.scripts.tabular_baselines import catboost_metric

            clf = catboost_metric
        elif method == "random_forest":
            from tabpfn.scripts.tabular_baselines import random_forest_metric

            clf = random_forest_metric
        elif method == "logistic":
            from tabpfn.scripts.tabular_baselines import logistic_metric

            clf = logistic_metric
        metric, ys, res = clf(
            x,
            y,
            test_x,
            test_y,
            [],
            metric_used,
            max_time=max_time,
            no_tune={},
        )
    # If sklearn classifier
    elif isinstance(method, BaseEstimator):
        method.fit(X=x, y=y.long())
        ys = method.predict_proba(test_x)
    else:
        metric, ys, res = method(
            x,
            y,
            test_x,
            test_y,
            [],
            metric_used,
        )
    # 如果是概率输出
    preds = np.argmax(ys, axis=1)
    acc = accuracy_score(test_y, preds)
    # roc = roc_auc_score(test_y, ys[:, 1])  # 只适用于二分类
    # acc = tabpfn.scripts.tabular_metrics.accuracy_metric(test_y, ys)
    # roc = tabpfn.scripts.tabular_metrics.auc_metric(test_y, ys)

    method_str = method if type(method) == str else "transformer"
    return {
        "acc": float(acc),
        # "roc": float(roc),
        "prompt": prompt_id,
        "seed": seed,
        "name": name,
        "size": len(df_train),
        "method": method_str,
        "max_time": max_time,
        "feats": x.shape[-1],
    }


def get_leave_one_out_importance(
    df_train, df_test, ds, method, metric_used, max_time=30
):
    """Get the importance of each feature for a dataset by dropping it in the training and prediction."""
    res_base = evaluate_dataset(
        ds,
        df_train,
        df_test,
        prompt_id="",
        name=ds[0],
        method=method,
        metric_used=metric_used,
        max_time=max_time,
    )

    importances = {}
    for feat_idx, feat in enumerate(set(df_train.columns)):
        if feat == ds[4][-1]:
            continue
        df_train_ = df_train.copy().drop(feat, axis=1)
        df_test_ = df_test.copy().drop(feat, axis=1)
        ds_ = copy.deepcopy(ds)

        res = evaluate_dataset(
            ds_,
            df_train_,
            df_test_,
            prompt_id="",
            name=ds[0],
            method=method,
            metric_used=metric_used,
            max_time=max_time,
        )
        importances[feat] = (round(res_base["roc"] - res["roc"], 3),)
    return importances
