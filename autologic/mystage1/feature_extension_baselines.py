import pandas as pd
import os
from .run_llm_code import run_llm_code
from typing import Tuple


def extend_using_dfs(df_train: pd.DataFrame, df_test: pd.DataFrame, target_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:

    import featuretools as ft

    es = ft.EntitySet(id="Test")
    es = es.add_dataframe(
        dataframe_name="data",
        dataframe=pd.concat([df_train, df_test]),
        index="index",
    )

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="data",
        trans_primitives=["add_numeric", "multiply_numeric"],
    )

    df_train, df_test = (
        feature_matrix.iloc[: len(df_train), :].reset_index(drop=True),
        feature_matrix.iloc[len(df_train) :, :].reset_index(drop=True),
    )

    return df_train, df_test


def extend_using_autofeat(df_train, df_test, target_train):

    from autofeat import FeatureSelector, AutoFeatRegressor, AutoFeatClassifier

    from sklearn.preprocessing import OrdinalEncoder

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    for col in df_train.columns:
        if df_train[col].dtype == "object" or df_train[col].dtype.name == "category":
            df_train[col] = df_train[col].astype(str)
            df_test[col] = df_test[col].astype(str)

            df_train[col] = encoder.fit_transform(df_train[[col]]).ravel()
            df_test[col] = encoder.transform(df_test[[col]]).ravel()

            df_train[col] = df_train[col].astype(float)
            df_test[col] = df_test[col].astype(float)

    df_train = df_train.fillna(-1)
    df_test = df_test.fillna(-1)

    classifier = AutoFeatClassifier(verbose=1, feateng_steps=1)
    df_train = classifier.fit_transform(df_train, target_train.astype("int"))
    df_test = classifier.transform(df_test)

    return df_train, df_test


def extend_using_stage1(df_train, df_test, ds, seed, prompt_id, code_overwrite=None):
    if code_overwrite:
        code = code_overwrite
    else:
        data_dir = os.environ.get("DATA_DIR", "data/")
        f = open(f"{data_dir}/generated_code/{ds[0]}_{prompt_id}_{seed}_code.txt", "r")
        code = f.read()
        f.close()

    df_train = run_llm_code(
        code,
        df_train,
        convert_categorical_to_integer=not ds[0].startswith("kaggle"),
    )
    df_test = run_llm_code(
        code, df_test, convert_categorical_to_integer=not ds[0].startswith("kaggle")
    )

    return df_train, df_test
