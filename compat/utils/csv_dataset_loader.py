from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _candidate_bases(base_loc: str | Path) -> list[Path]:
    base = Path(base_loc)
    autologic_root = Path(__file__).resolve().parents[3]
    return [
        base,
        Path(os.environ.get("AUTOLOGIC_CSV_DATA_DIR", autologic_root / "data" / "csv_data")),
        Path(os.environ.get("AUTOLOGIC_REGRESSION_DATA_DIR", autologic_root / "data" / "csv_data")),
    ]


def read_txt_file(file_path: str | Path, *, encoding: str = "utf-8") -> str:
    path = Path(file_path)
    with path.open("r", encoding=encoding) as f:
        return f.read()


def resolve_description_path(base_loc: str | Path, dataset_name: str) -> Optional[Path]:
    candidates = []
    for base in _candidate_bases(base_loc):
        candidates.extend([
            base / "data_description.txt",
            base / f"{dataset_name}-description.txt",
        ])
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_csv_path(base_loc: str | Path, dataset_name: str) -> Path:
    names = [dataset_name]
    aliases = {
        "forest": "forest-fires",
        "puma8nh": "puma8NH",
    }
    low = dataset_name.lower()
    if low in aliases:
        names.append(aliases[low])
    for base in _candidate_bases(base_loc):
        for name in names:
            path = base / f"{name}.csv"
            if path.exists():
                return path
    raise FileNotFoundError(f"CSV dataset not found: {dataset_name} under {base_loc}")


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
    csv_path = _resolve_csv_path(base_loc, dataset_name)
    df, target_column_name = load_csv_dataframe(csv_path)

    n_clusters = int(len(np.unique(df[target_column_name])))

    try:
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        raise ImportError("load_csv_dataset requires scikit-learn for train_test_split.") from exc

    df_train_raw, df_test_raw = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True
    )

    desc_path = resolve_description_path(csv_path.parent, csv_path.stem)
    dataset_description = read_txt_file(desc_path, encoding=description_encoding) if desc_path else ""

    return df_train_raw, df_test_raw, n_clusters, target_column_name, dataset_description
