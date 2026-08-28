from sklearn.cluster import SpectralClustering
from scipy.sparse import lil_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
import numpy as np


def _pairwise_ari_mean(labels_list, max_pairs: int = 200) -> float:
    labels_list = [np.asarray(x) for x in labels_list]
    n = len(labels_list)
    if n < 2:
        return float("nan")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    vals = []
    for i, j in pairs:
        try:
            vals.append(adjusted_rand_score(labels_list[i], labels_list[j]))
        except Exception:
            continue
    if not vals:
        return float("nan")
    return float(np.mean(vals))



def select_top_models_with_adaptive_threshold(test_ari_list, test_nmi_list, base_models, top_k=5, threshold_ratio=0.5):
    ari_array = np.array(test_ari_list)
    nmi_array = np.array(test_nmi_list)

    # Exclude negative numbers
    non_negative_indices = [i for i, (ari, nmi) in enumerate(zip(ari_array, nmi_array))
                            if ari > 0 and nmi > 0]
    if not non_negative_indices:
        print("⚠️ No non-negative indicator models")
        return []

    # Get maximum value and calculate adaptive threshold
    max_ari = np.max(ari_array[non_negative_indices])
    max_nmi = np.max(nmi_array[non_negative_indices])
    min_ari = max_ari * threshold_ratio
    min_nmi = max_nmi * threshold_ratio

    # Filter models based on adaptive threshold
    valid_indices = [i for i in non_negative_indices if ari_array[i] >= min_ari and nmi_array[i] >= min_nmi]

    if not valid_indices:
        print("⚠️ No models meeting adaptive threshold")
        return []

    # at满足条件 of 模型in选 top_k（按 ARI 降序）
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



# 1. 优化后 of  get_param_prompt（提示词升级，LLM能自我纠错）
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

    # Step 1: 筛选have效聚类器
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
        raise RuntimeError(f"have效模型数not足，仅 {len(base_labels)} ，无法集成")

    n_models = len(base_labels)
    base_labels = np.array(base_labels)  # shape: (n_models, n_samples)

    base_labels = base_labels.astype(int, copy=False)
    S = np.zeros((n_samples, n_samples), dtype=np.float32)
    for m in range(n_models):
        labels = base_labels[m]
        S += np.equal.outer(labels, labels).astype(np.float32)
    S /= float(n_models)

    # Step 3: use共聚类概率矩阵作as相似度做聚类
    # If未指定n_clusters，取所have基础聚类in簇数 of 众数
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


# ===========================
# 聚类蒸馏函数（将聚类模型蒸馏到MLP学生网络）
# ===========================
def distill_to_student_clustering(teacher_model, X_train, y_train, X_val, y_val, 
                                  device='cpu', epochs=30, n_clusters=None):
    """
    将教师聚类模型蒸馏到MLP学生网络。
    学生网络学习教师模型的聚类标签预测能力。
    
    参数:
    - teacher_model: 教师聚类模型（需实现fit和fit_predict）
    - X_train, y_train: 训练数据和真实标签
    - X_val, y_val: 验证数据和真实标签
    - device: 'cuda' 或 'cpu'
    - epochs: 训练轮数
    - n_clusters: 聚类数量，如果为None则从教师模型推断
    
    返回: DistilledClusterer 对象
    """
    import copy
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    def _to_numpy(x):
        if hasattr(x, "to_numpy"):
            return x.to_numpy()
        return np.asarray(x)

    def _as_dataframe(x, n_features: int):
        try:
            import pandas as pd
        except Exception:
            return x
        arr = np.asarray(x)
        if arr.ndim != 2:
            return x
        cols = [f"f{i}" for i in range(int(n_features))]
        return pd.DataFrame(arr, columns=cols)

    X_train = _to_numpy(X_train)
    X_val = _to_numpy(X_val)
    y_train_np = _to_numpy(y_train).astype(int, copy=False)
    y_val_np = _to_numpy(y_val).astype(int, copy=False)

    print("    开始聚类蒸馏...")

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)

    teacher = teacher_model
    teacher_can_predict = hasattr(teacher, "predict")
    teacher_labels_train = None
    teacher_labels_val = None

    if teacher_can_predict:
        try:
            teacher_labels_train = _to_numpy(teacher.predict(X_train)).astype(int, copy=False)
            teacher_labels_val = _to_numpy(teacher.predict(X_val)).astype(int, copy=False)
        except Exception as e:
            if "has no attribute 'values'" in str(e):
                X_train_df = _as_dataframe(X_train, X_train.shape[1])
                X_val_df = _as_dataframe(X_val, X_val.shape[1])
                try:
                    teacher_labels_train = _to_numpy(teacher.predict(X_train_df)).astype(int, copy=False)
                    teacher_labels_val = _to_numpy(teacher.predict(X_val_df)).astype(int, copy=False)
                except Exception:
                    teacher_labels_train = None
                    teacher_labels_val = None
            else:
                teacher_labels_train = None
                teacher_labels_val = None
            teacher_labels_train = None
            teacher_labels_val = None

    if teacher_labels_train is None:
        try:
            teacher = copy.deepcopy(teacher_model)
        except Exception:
            teacher = teacher_model

        if hasattr(teacher, "fit_predict"):
            try:
                teacher_labels_train = _to_numpy(teacher.fit_predict(X_train)).astype(int, copy=False)
            except Exception as e:
                if "has no attribute 'values'" in str(e):
                    X_train_df = _as_dataframe(X_train, X_train.shape[1])
                    teacher_labels_train = _to_numpy(teacher.fit_predict(X_train_df)).astype(int, copy=False)
                else:
                    raise
        else:
            try:
                teacher.fit(X_train)
                teacher_labels_train = _to_numpy(teacher.predict(X_train)).astype(int, copy=False)
            except Exception as e:
                if "has no attribute 'values'" in str(e):
                    X_train_df = _as_dataframe(X_train, X_train.shape[1])
                    teacher.fit(X_train_df)
                    teacher_labels_train = _to_numpy(teacher.predict(X_train_df)).astype(int, copy=False)
                else:
                    raise

        if hasattr(teacher, "predict"):
            try:
                teacher_labels_val = _to_numpy(teacher.predict(X_val)).astype(int, copy=False)
            except Exception as e:
                if "has no attribute 'values'" in str(e):
                    X_val_df = _as_dataframe(X_val, X_val.shape[1])
                    teacher_labels_val = _to_numpy(teacher.predict(X_val_df)).astype(int, copy=False)
                else:
                    raise
        else:
            nn_search = NearestNeighbors(n_neighbors=1, algorithm="auto")
            nn_search.fit(X_train_scaled)
            nn_idx = nn_search.kneighbors(X_val_scaled, return_distance=False).reshape(-1)
            teacher_labels_val = teacher_labels_train[nn_idx]

    teacher_labels_train = _to_numpy(teacher_labels_train).astype(int, copy=False)
    teacher_labels_val = _to_numpy(teacher_labels_val).astype(int, copy=False)
    all_teacher_labels = np.unique(np.concatenate([teacher_labels_train, teacher_labels_val], axis=0))
    label_to_id = {int(lb): int(i) for i, lb in enumerate(all_teacher_labels.tolist())}
    teacher_labels_train = np.vectorize(lambda z: label_to_id[int(z)], otypes=[int])(teacher_labels_train)
    teacher_labels_val = np.vectorize(lambda z: label_to_id[int(z)], otypes=[int])(teacher_labels_val)

    n_clusters = int(len(all_teacher_labels))

    print(f"    目标簇数: {n_clusters}")
    
    # 转换为torch张量
    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(teacher_labels_train, dtype=torch.long),
        torch.tensor(y_train_np, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(teacher_labels_val, dtype=torch.long),
        torch.tensor(y_val_np, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # 定义学生网络
    input_dim = X_train_scaled.shape[1]
    
    class StudentClusterer(nn.Module):
        def __init__(self, input_dim, n_clusters, hidden_dims=(128, 64), dropout=0.3):
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
            layers.append(nn.Linear(d, n_clusters))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.net(x)
    
    student = StudentClusterer(input_dim, n_clusters, hidden_dims=(128, 64), dropout=0.3).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_ari = -1
    best_state = None
    
    for epoch in range(epochs):
        student.train()
        for xb, teacher_lb, true_lb in train_loader:
            xb, teacher_lb = xb.to(device), teacher_lb.to(device)
            optimizer.zero_grad()
            pred = student(xb)
            # 使用教师模型的标签作为目标（蒸馏）
            loss = criterion(pred, teacher_lb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        
        # 验证
        student.eval()
        with torch.no_grad():
            all_preds = []
            all_true = []
            for xb, teacher_lb, true_lb in val_loader:
                xb = xb.to(device)
                pred = student(xb).argmax(dim=1).cpu().numpy()
                all_preds.extend(pred)
                all_true.extend(true_lb.numpy())
            
            from sklearn.metrics import adjusted_rand_score
            val_ari = adjusted_rand_score(all_true, all_preds)
            if val_ari > best_val_ari:
                best_val_ari = val_ari
                best_state = copy.deepcopy(student.state_dict())
    
    if best_state:
        student.load_state_dict(best_state)
    
    print(f"    蒸馏完成，学生网络验证集 ARI: {best_val_ari:.4f}")
    wrapped = DistilledClusterer(student, x_scaler, device, n_clusters)
    wrapped.distill_val_ari = float(best_val_ari)
    wrapped.distill_method = "ce"
    return wrapped


class DistilledClusterer:
    """蒸馏后的聚类学生模型包装器"""
    def __init__(self, model, x_scaler, device, n_clusters):
        self.model = model
        self.x_scaler = x_scaler
        self.device = device
        self.n_clusters = n_clusters
        self.name = "DistilledClusterer"
    
    def fit(self, X):
        """聚类模型的fit方法（不做任何操作）"""
        return self
    
    def fit_predict(self, X):
        """聚类模型的fit_predict方法"""
        import torch
        self.model.eval()
        X_scaled = self.x_scaler.transform(X)
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor).cpu().numpy()
            predictions = np.argmax(logits, axis=1)
        return predictions
    
    def predict(self, X):
        """聚类模型的predict方法"""
        return self.fit_predict(X)


# ===========================
# 聚类校准函数（基于聚类一致性的距离校准）
# ===========================
def calibrate_cluster_predictions(X, base_models_with_predictions, base_models_without_predictions=None,
                                  alpha=0.1, calibration_method='distance'):
    """
    对聚类结果进行校准。通过调整聚类预测的置信度/权重。
    
    参数:
    - X: 特征数据
    - base_models_with_predictions: 已生成预测的模型列表及其预测结果
    - base_models_without_predictions: 未生成预测的模型列表
    - alpha: 校准强度参数
    - calibration_method: 'distance' 使用样本到簇中心的距离进行校准
    
    返回: 校准后的预测结果字典
    """
    
    calibrated_results = {}
    
    if calibration_method == 'distance':
        # 基于样本到簇中心的距离进行校准
        # 距离近的样本置信度高，距离远的样本置信度低
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for model_name, predictions in base_models_with_predictions.items():
            # 计算每个样本到其所属簇中心的距离
            unique_clusters = np.unique(predictions)
            distances = np.zeros(len(X_scaled))
            
            for cluster_id in unique_clusters:
                mask = predictions == cluster_id
                cluster_center = X_scaled[mask].mean(axis=0)
                cluster_samples = X_scaled[mask]
                # 计算该簇中样本到簇中心的距离
                dists = np.linalg.norm(cluster_samples - cluster_center, axis=1)
                distances[mask] = dists
            
            # 归一化距离到[0, 1]
            max_dist = distances.max() + 1e-8
            normalized_distances = distances / max_dist
            
            # 置信度 = 1 - 归一化距离（距离近的样本置信度高）
            confidence = 1.0 - normalized_distances
            
            calibrated_results[model_name] = {
                'predictions': predictions,
                'confidence': confidence,
                'distance': distances
            }
    
    elif calibration_method == 'consistency':
        # 基于多个模型的预测一致性进行校准
        # 多个模型都预测相同簇的样本置信度高
        pass  # 在集成阶段处理
    
    return calibrated_results


def stacking_clustering_with_calibration(X_aug, base_models, y_true, 
                                         meta_model=None, calibrate=False, 
                                         alpha=0.1, random_state=42):
    """
    基于Stacking的聚类集成（包含可选的校准）。
    
    参数:
    - X_aug: 增强后的特征数据
    - base_models: 基础聚类模型列表
    - y_true: 真实标签
    - meta_model: 元模型（如果为None则使用多数投票）
    - calibrate: 是否进行校准
    - alpha: 校准参数
    - random_state: 随机种子
    
    返回: 集成结果字典
    """
    
    n_samples = X_aug.shape[0]
    base_labels_list = []
    
    # 获取所有基础模型的预测
    for model in base_models:
        try:
            labels = model.fit_predict(X_aug)
            if labels is not None and len(labels) == n_samples:
                base_labels_list.append(labels)
        except:
            continue
    
    if len(base_labels_list) == 0:
        raise ValueError("需要至少1个有效的基础模型")
    
    base_labels_array = np.array(base_labels_list)  # shape: (n_models, n_samples)
    base_pairwise_ari_mean = _pairwise_ari_mean(base_labels_list) if len(base_labels_list) >= 2 else None
    
    confidence_mean = None
    if not calibrate:
        if len(base_labels_list) == 1:
            ensemble_labels = base_labels_list[0]
        else:
            from scipy.stats import mode
            ensemble_labels = mode(base_labels_array, axis=0, keepdims=False)[0]
    else:
        if len(base_labels_list) == 1:
            ensemble_labels = base_labels_list[0]
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_aug)
            
            confidences = []
            for labels in base_labels_list:
                distances = np.zeros(n_samples)
                for cluster_id in np.unique(labels):
                    mask = labels == cluster_id
                    cluster_center = X_scaled[mask].mean(axis=0)
                    dists = np.linalg.norm(X_scaled[mask] - cluster_center, axis=1)
                    distances[mask] = dists
                
                max_dist = distances.max() + 1e-8
                confidence = 1.0 - (distances / max_dist) * alpha
                confidences.append(confidence)
            
            confidences = np.array(confidences)  # shape: (n_models, n_samples)
            confidence_mean = float(np.mean(confidences))
            
            ensemble_labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                votes = {}
                for m in range(len(base_labels_list)):
                    label = base_labels_array[m, i]
                    weight = confidences[m, i]
                    votes[label] = votes.get(label, 0.0) + float(weight)
                
                ensemble_labels[i] = max(votes, key=votes.get)
    
    # 评估
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(y_true, ensemble_labels)
    nmi = normalized_mutual_info_score(y_true, ensemble_labels)
    
    # 如果需要，记录校准相关的指标
    calibration_metrics = {}
    if calibrate:
        # 这里可以添加更多校准相关的指标
        calibration_metrics = {
            'calibration_method': 'distance_based',
            'calibration_strength': alpha,
            'confidence_mean': confidence_mean
        }
    
    return {
        'ensemble_metrics': {
            'ensemble_ari': ari,
            'ensemble_nmi': nmi,
            'base_pairwise_ari_mean': base_pairwise_ari_mean,
            'calibration': calibration_metrics if calibrate else None
        },
        'ensemble_labels': ensemble_labels
    }


def consensus_ensemble_topk(X, models, y_true, n_clusters=None, max_models=5, min_models=2):
    from sklearn.metrics import adjusted_rand_score

    scored = []
    for m in models:
        try:
            pred = m.fit_predict(X)
            if pred is None or len(pred) != X.shape[0]:
                continue
            ari = adjusted_rand_score(y_true, pred)
            scored.append((ari, m))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    if len(scored) < min_models:
        raise ValueError(f"需要至少{min_models}个有效的基础模型")

    k = len(scored) // 2
    k = max(k, min_models)
    k = min(k, max_models, len(scored))
    top_models = [m for _, m in scored[:k]]
    result = cluster_ensemble_caps(
        X=np.asarray(X),
        clusterers=top_models,
        true_labels=np.asarray(y_true),
        n_clusters=n_clusters,
        min_valid_models=min_models,
    )
    result["selected_model_count"] = int(k)
    result["selected_model_ari"] = [float(s) for s, _ in scored[:k]]
    return result


def consensus_ensemble_all(X, models, y_true, n_clusters=None, max_models=10, min_models=2):
    from sklearn.metrics import adjusted_rand_score

    scored = []
    for m in models:
        try:
            pred = m.fit_predict(X)
            if pred is None or len(pred) != X.shape[0]:
                continue
            if len(np.unique(pred)) <= 1:
                continue
            ari = adjusted_rand_score(y_true, pred)
            scored.append((ari, m))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    if len(scored) < min_models:
        raise ValueError(f"需要至少{min_models}个有效的基础模型")

    k = min(int(max_models), len(scored))
    top_models = [m for _, m in scored[:k]]
    result = cluster_ensemble_caps(
        X=np.asarray(X),
        clusterers=top_models,
        true_labels=np.asarray(y_true),
        n_clusters=n_clusters,
        min_valid_models=min_models,
    )
    result["selected_model_count"] = int(k)
    result["selected_model_ari"] = [float(s) for s, _ in scored[:k]]
    return result
