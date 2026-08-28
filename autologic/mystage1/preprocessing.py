import pandas as pd
import copy
import numpy as np
from typing import Dict, Optional, Tuple


def create_mappings(df_train: pd.DataFrame) -> Dict[str, Dict[int, str]]:

    mappings = {}
    for col in df_train.columns:
        if (
            df_train[col].dtype.name == "category"
            or df_train[col].dtype.name == "object"
        ):
            mappings[col] = {v: i for i, v in 
                enumerate(df_train[col].astype("category").cat.categories)
            }
    return mappings


def convert_categorical_to_integer_f(column: pd.Series, mapping: Optional[Dict[int, str]] = None) -> pd.Series:

    if mapping is not None:
        if column.dtype.name == "category":
            column = column.cat.add_categories([-1])
        return column.map(mapping).fillna(-1).astype(int)
    return column


def split_target_column(df: pd.DataFrame, target: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    X = df[[c for c in df.columns if c != target]]
    if not (target and target in df.columns):
        return X, None

    y = df[target]
    try:
        return X, y.astype(int)
    except Exception:
        try:
            y_codes = y.astype("category").cat.codes
            return X, y_codes.astype(int)
        except Exception:
            return X, None


def make_dataset_numeric(df: pd.DataFrame, mappings: Dict[str, Dict[int, str]]) -> pd.DataFrame:

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.apply(
        lambda col: convert_categorical_to_integer_f(
            col, mapping=mappings.get(col.name)
        ),
        axis=0,
    )
    df = df.astype(float)
    df = df.fillna(0.0)

    return df


def make_datasets_numeric(df_train: pd.DataFrame, df_test: Optional[pd.DataFrame], target_column: str, return_mappings: Optional[bool] = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[Dict[str, Dict[int, str]]]]:

    df_train = copy.deepcopy(df_train)
    df_train = df_train.infer_objects()
    if df_test is not None:
        df_test = copy.deepcopy(df_test)
        df_test = df_test.infer_objects()

    mappings = create_mappings(df_train)

    non_target = [c for c in df_train.columns if c != target_column]
    df_train[non_target] = make_dataset_numeric(df_train[non_target], mappings)

    if df_test is not None:
        df_test[non_target] = make_dataset_numeric(df_test[non_target], mappings)

    if return_mappings:
        return df_train, df_test, mappings

    return df_train, df_test
