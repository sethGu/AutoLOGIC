from sklearn.cluster import SpectralClustering
from scipy.sparse import lil_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
import numpy as np



def select_top_models_with_adaptive_threshold(test_ari_list, test_nmi_list, base_models, top_k=5, threshold_ratio=0.5):
    ari_array = np.array(test_ari_list)
    nmi_array = np.array(test_nmi_list)

    # 排除负数
    non_negative_indices = [i for i, (ari, nmi) in enumerate(zip(ari_array, nmi_array))
                            if ari > 0 and nmi > 0]
    if not non_negative_indices:
        print("⚠️ 没有非负指标模型")
        return []

    # 获取最大值并计算自适应阈值
    max_ari = np.max(ari_array[non_negative_indices])
    max_nmi = np.max(nmi_array[non_negative_indices])
    min_ari = max_ari * threshold_ratio
    min_nmi = max_nmi * threshold_ratio

    # 根据自适应阈值筛选模型
    valid_indices = [i for i in non_negative_indices if ari_array[i] >= min_ari and nmi_array[i] >= min_nmi]

    if not valid_indices:
        print("⚠️ 没有满足自适应阈值的模型")
        return []

    # 在满足条件的模型中选 top_k（按 ARI 降序）
    valid_ari = ari_array[valid_indices]
    sorted_order = np.argsort(valid_ari)[::-1]
    top_indices = [valid_indices[i] for i in sorted_order[:top_k]]

    return [base_models[i] for i in top_indices]



def is_model_nmf(model):
    if isinstance(model, NMF):
        return True
    try:
        if hasattr(model, 'model') and isinstance(model.model, NMF):
            return True
    except:
        pass
    return False



# 1. 优化后的 get_param_prompt（提示词升级，LLM能自我纠错）
def get_param_prompt(
    best_code, best_ari, dataset_description,
    X_aug, feature_columns, dataset_name=None, max_rows=10
):
    """
    生成适用于 LLM 参数优化的提示词（支持mycluster包裹真实聚类器，参数透传版）
    """
    import pandas as pd
    if isinstance(X_aug, np.ndarray):
        df_show = pd.DataFrame(X_aug, columns=feature_columns)
    else:
        df_show = X_aug.copy()
    table = df_show.head(max_rows).to_string(index=False)
    data_shape = X_aug.shape if hasattr(X_aug, 'shape') else (len(X_aug), len(feature_columns))
    if dataset_name is None:
        dataset_name = "unknown"

    prompt = (
        f"Here is the best clustering model code so far, with its current ARI (Adjusted Rand Index) score:\n\n"
        f"Current best ARI: {best_ari:.4f}\n\n"
        f"Model code:\n"
        f"```python\n{best_code}\n```\n"
        f"The downstream clustering task is based on the following dataset.\n"
        f"Dataset name: {dataset_name}\n"
        f"Dataset description:\n{dataset_description}\n\n"
        f"Feature names:\n{', '.join(feature_columns)}\n\n"
        f"Dataset shape: {data_shape}\n"
        f"Here are the first {max_rows} rows of the dataset used for clustering:\n{table}\n\n"
        "Please ONLY optimize the hyperparameters\n"
        "in the given clustering model code to further improve the ARI value.\n"
        "DO NOT change the algorithm type or model structure.\n"
        "Output only a new optimized Python code block.\n"
        "No explanation, only code!\n"
        "IMPORTANT CONTEXT:\n"
        "You are writing clustering model code in Python using scikit-learn version 1.6.1.\n"
        "STRICT REQUIREMENT:\n"
        "ONLY use parameters that are supported by scikit-learn version 1.6.1.\n"
        "DO NOT use any parameters that are deprecated or were only available in versions prior to 1.2.\n"
        "Refer ONLY to the scikit-learn 1.6.1 documentation for valid parameters and their default values.\n"
    )

    return prompt

"""
"At the end of your code block, output two lines:\n"
        "_params = dict(...), containing only parameters needed for mycluster's __init__ (such as n_clusters);\n"
        "_params_gmm = dict(...), containing parameters to be passed to the underlying GaussianMixture (such as n_init, max_iter, etc.).\n"
        "In your mycluster class, make sure you pass **_params_gmm to GaussianMixture in the constructor.\n"
        "For example: self.model = GaussianMixture(n_components=self.n_clusters, **_params_gmm)\n"
        "This will ensure parameter optimization only applies the correct arguments to the correct objects."
"""


def cluster_ensemble_caps(X, clusterers, true_labels, n_clusters=None, min_valid_models=2):
    """
    CAPS（Cluster Aggregation with Pairwise Similarities）聚类集成方法。

    参数:
    - X: ndarray, shape (n_samples, n_features)
    - clusterers: list of clusterer objects (must implement fit_predict)
    - true_labels: ndarray, shape (n_samples,)
    - n_clusters: 最终聚类簇数，默认用最多的基础聚类簇数
    - min_valid_models: 至少多少个聚类器有效，否则报错
    """
    n_samples = X.shape[0]
    base_labels = []

    # Step 1: 筛选有效聚类器
    for model in clusterers:
        try:
            labels = model.fit_predict(X)
            # 合法性检查
            if labels is None or len(labels) != n_samples:
                continue
            if len(np.unique(labels)) <= 1:
                continue
            base_labels.append(labels)
        except Exception as e:
            continue

    if len(base_labels) < min_valid_models:
        raise RuntimeError(f"有效模型数不足，仅 {len(base_labels)} 个，无法集成")

    n_models = len(base_labels)
    base_labels = np.array(base_labels)  # shape: (n_models, n_samples)

    # Step 2: 构建共聚类概率矩阵（co-association matrix）
    # S[i, j] = base_labels的每一行中i和j属于同一簇的比例
    S = np.zeros((n_samples, n_samples))
    for m in range(n_models):
        labels = base_labels[m]
        for i in range(n_samples):
            for j in range(i, n_samples):
                same_cluster = int(labels[i] == labels[j])
                S[i, j] += same_cluster
                if i != j:
                    S[j, i] += same_cluster  # 保证对称

    S = S / n_models  # 归一化到[0,1]

    # Step 3: 用共聚类概率矩阵作为相似度做聚类
    # 如果未指定n_clusters，取所有基础聚类中簇数的众数
    if n_clusters is None:
        from scipy.stats import mode
        all_n_clusters = [len(np.unique(labels)) for labels in base_labels]
        n_clusters = int(mode(all_n_clusters, keepdims=True)[0][0])

    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    final_labels = sc.fit_predict(S)

    # Step 4: 性能评估
    ari = adjusted_rand_score(true_labels, final_labels)
    nmi = normalized_mutual_info_score(true_labels, final_labels)

    return {
        "ensemble_metrics": {
            "ensemble_ari": ari,
            "ensemble_nmi": nmi,
            "used_model_count": n_models
        },
        "ensemble_labels": final_labels
    }

