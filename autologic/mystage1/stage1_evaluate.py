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


def _coerce_feature_frame_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Convert generated feature columns to finite numeric values for selectors."""
    X_num = X.copy()
    for col in X_num.columns:
        s = X_num[col]
        if pd.api.types.is_bool_dtype(s):
            X_num[col] = s.astype("int8")
        elif pd.api.types.is_numeric_dtype(s):
            X_num[col] = pd.to_numeric(s, errors="coerce")
        else:
            codes, _ = pd.factorize(s.astype("string"), sort=True)
            codes = codes.astype(float)
            codes[codes < 0] = np.nan
            X_num[col] = codes
    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    return X_num.fillna(0.0)


def filter_generated_features(
    df,
    new_feature_cols,
    target_col,
    task_type,
    *,
    top_k: int = 5,
    min_keep: int = 1,
    selection: str = "intersection",
    p_threshold: float = 0.05,
    variance_threshold: float = 1e-3,
    cluster_top_n: int = 5,
):
    X_new = _coerce_feature_frame_numeric(df[new_feature_cols])
    if task_type == 'classification':
        y = df[target_col]  
        selector = SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=0), k='all')
        selector.fit(X_new, y)
        stat_scores = np.array(selector.scores_, dtype=float)
        stat_scores = np.nan_to_num(stat_scores, nan=-np.inf)
        k = min(int(top_k), len(new_feature_cols))
        top_indices = np.argsort(stat_scores)[::-1][:k]
        stat_selected_features = [new_feature_cols[i] for i in top_indices]
        model = LGBMClassifier(n_estimators=100, random_state=0,verbosity=-1)
        model.fit(X_new, y)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'feature': new_feature_cols, 'importance': importances})
        model_selected_features = list(importance_df.sort_values('importance', ascending=False)['feature'][:k])
        if selection == "union":
            final_features = list(dict.fromkeys(stat_selected_features + model_selected_features))
        else:
            final_features = [f for f in stat_selected_features if f in set(model_selected_features)]
        if len(final_features) < int(min_keep):
            fallback = list(dict.fromkeys(model_selected_features + stat_selected_features))
            final_features = fallback[: max(int(min_keep), len(final_features))]
        return final_features[:k]

    elif task_type == 'regression':
        y = df[target_col]
        selector = SelectKBest(f_regression, k='all')
        selector.fit(X_new, y)
        p_values = selector.pvalues_ 
        stat_selected_features = [new_feature_cols[i] for i, p in enumerate(p_values) if p < p_threshold]
        if len(stat_selected_features) == 0 and len(new_feature_cols) > 0:
            stat_selected_features = [new_feature_cols[np.argmin(p_values)]]
        model = LGBMRegressor(n_estimators=100, random_state=0,verbosity=-1)
        model.fit(X_new, y)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'feature': new_feature_cols, 'importance': importances})
        k = min(int(top_k), len(new_feature_cols))
        model_selected_features = list(importance_df.sort_values('importance', ascending=False)['feature'][:k])
        if selection == "union":
            final_features = list(dict.fromkeys(stat_selected_features + model_selected_features))
        else:
            final_features = [f for f in stat_selected_features if f in set(model_selected_features)]
        if len(final_features) < int(min_keep):
            fallback = list(dict.fromkeys(model_selected_features + stat_selected_features))
            final_features = fallback[: max(int(min_keep), len(final_features))]
        return final_features[:k]

    elif task_type == 'clustering':
        retained_feats = [f for f in new_feature_cols if df[f].var() > variance_threshold]
        if len(retained_feats) == 0:
            return []

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

        if not silhouette_scores:
            k = min(int(cluster_top_n), len(retained_feats))
            return retained_feats[: max(int(min_keep), k)]
        k = min(int(cluster_top_n), len(silhouette_scores))
        selected_features = sorted(silhouette_scores, key=silhouette_scores.get, reverse=True)[:k]
        if len(selected_features) < int(min_keep):
            selected_features = selected_features + [f for f in retained_feats if f not in set(selected_features)]
        return selected_features[: max(int(min_keep), k)]

    else:
        raise ValueError("task_type Parameters必须is 'classification'、'regression' or 'clustering'")

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
        )  
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
    preds = np.argmax(ys, axis=1)
    acc = accuracy_score(test_y, preds)
    # roc = roc_auc_score(test_y, ys[:, 1])  
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
