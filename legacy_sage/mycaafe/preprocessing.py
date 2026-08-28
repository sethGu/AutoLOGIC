import pandas as pd
import copy
import numpy as np
from typing import Dict, Optional, Tuple


def create_mappings(df_train: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    """
    Creates a dictionary of mappings for categorical columns in the given dataframe.

    Parameters:
    df_train (pandas.DataFrame): The dataframe to create mappings for.

    Returns:
    Dict[str, Dict[int, str]]: A dictionary of mappings for categorical columns in the dataframe.
    """
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
    """
    Converts a categorical column to integer values using the given mapping.

    Parameters:
    column (pandas.Series): The column to convert.
    mapping (Dict[int, str], optional): The mapping to use for the conversion. Defaults to None.

    Returns:
    pandas.Series: The converted column.
    """
    if mapping is not None:
        # if column is categorical
        if column.dtype.name == "category":
            column = column.cat.add_categories([-1])
        return column.map(mapping).fillna(-1).astype(int)
    return column


def split_target_column(df: pd.DataFrame, target: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Splits the given dataframe into the feature dataframe and the target column.

    Parameters:
    df (pandas.DataFrame): The dataframe to split.
    target (str, optional): The name of the target column. Defaults to None.

    Returns:
    Tuple[pandas.DataFrame, Optional[pandas.Series]]: The feature dataframe and the target column (if it exists).
    """
    return (
        df[[c for c in df.columns if c != target]],
        df[target].astype(int) if (target and target in df.columns) else None,
    )


def make_dataset_numeric(df: pd.DataFrame, mappings: Dict[str, Dict[int, str]]) -> pd.DataFrame:
    """
    Converts the categorical columns in the given dataframe to integer values using the given mappings.

    Parameters:
    df (pandas.DataFrame): The dataframe to convert.
    mappings (Dict[str, Dict[int, str]]): The mappings to use for the conversion.

    Returns:
    pandas.DataFrame: The converted dataframe.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.apply(
        lambda col: convert_categorical_to_integer_f(
            col, mapping=mappings.get(col.name)
        ),
        axis=0,
    )
    df = df.astype(float)

    return df


def make_datasets_numeric(df_train: pd.DataFrame, df_test: Optional[pd.DataFrame], target_column: str, return_mappings: Optional[bool] = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[Dict[str, Dict[int, str]]]]:
    """
    Converts the categorical columns in the given training and test dataframes to integer values using mappings created from the training dataframe.

    Parameters:
    df_train (pandas.DataFrame): The training dataframe to convert.
    df_test (pandas.DataFrame, optional): The test dataframe to convert. Defaults to None.
    target_column (str): The name of the target column.
    return_mappings (bool, optional): Whether to return the mappings used for the conversion. Defaults to False.

    Returns:
    Tuple[pandas.DataFrame, Optional[pandas.DataFrame], Optional[Dict[str, Dict[int, str]]]]: The converted training dataframe, the converted test dataframe (if it exists), and the mappings used for the conversion (if `return_mappings` is True).
    """
    df_train = copy.deepcopy(df_train)
    df_train = df_train.infer_objects()
    if df_test is not None:
        df_test = copy.deepcopy(df_test)
        df_test = df_test.infer_objects()

    # Create the mappings using the train and test datasets
    mappings = create_mappings(df_train)

    # Apply the mappings to the train and test datasets
    non_target = [c for c in df_train.columns if c != target_column]
    df_train[non_target] = make_dataset_numeric(df_train[non_target], mappings)

    if df_test is not None:
        df_test[non_target] = make_dataset_numeric(df_test[non_target], mappings)

    if return_mappings:
        return df_train, df_test, mappings

    return df_train, df_test
