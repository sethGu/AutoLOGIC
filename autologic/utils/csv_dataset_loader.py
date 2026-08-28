"""
CSV 数据集加载工具（可复用模块）

默认约定：
- CSV 最后一列为目标列/标签列（用于计算聚类数量 n_clusters）
- 会删除目标列缺失的样本行
- 对特征列尽量做数值化转换（失败则保持原类型）
- 可选读取数据集描述文件：
  1) {base_loc}/data_description.txt
  2) {base_loc}/{dataset_name}-description.txt

典型用法：
    from csv_dataset_loader import load_csv_dataset
    df_train_raw, df_test_raw, n_clusters, target_column_name, dataset_description = load_csv_dataset(
        dataset_name="mydata",
        base_loc="/path/to/dataset_dir",
        seed=42,
        test_size=0.25,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def read_txt_file(file_path: str | Path, *, encoding: str = "utf-8") -> str:
    path = Path(file_path)
    with path.open("r", encoding=encoding) as f:
        return f.read()


def resolve_description_path(base_loc: str | Path, dataset_name: str) -> Optional[Path]:
    base = Path(base_loc)
    candidates = [
        base / "data_description.txt",
        base / f"{dataset_name}-description.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_csv_dataframe(
    csv_path: str | Path,
    *,
    convert_dtypes: bool = True,
    drop_missing_target: bool = True,
    coerce_numeric_features: bool = True,
) -> Tuple[pd.DataFrame, str]:
    path = Path(csv_path)
    df = pd.read_csv(path)
    if convert_dtypes:
        df = df.convert_dtypes()

    if df.shape[1] < 1:
        raise ValueError(f"CSV has no columns: {path}")

    target_column_name = df.columns[-1]

    if drop_missing_target:
        df = df[~df[target_column_name].isna()].copy()

    if coerce_numeric_features:
        for col in df.columns:
            if col == target_column_name:
                continue
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    return df, str(target_column_name)


def load_csv_dataset(
    dataset_name: str,
    base_loc: str | Path,
    *,
    seed: int = 42,
    test_size: float = 0.25,
    description_encoding: str = "utf-8",
) -> Tuple[pd.DataFrame, pd.DataFrame, int, str, str]:
    csv_path = Path(base_loc) / f"{dataset_name}.csv"
    df, target_column_name = load_csv_dataframe(csv_path)

    n_clusters = int(len(np.unique(df[target_column_name])))

    try:
        from sklearn.model_selection import train_test_split
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "load_csv_dataset requires scikit-learn (sklearn) for train_test_split. "
            "Install it or call load_csv_dataframe and split manually."
        ) from e

    df_train_raw, df_test_raw = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True
    )

    desc_path = resolve_description_path(base_loc, dataset_name)
    dataset_description = read_txt_file(desc_path, encoding=description_encoding) if desc_path else ""

    return df_train_raw, df_test_raw, n_clusters, target_column_name, dataset_description
