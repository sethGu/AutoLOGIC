import os
import sys
# 添加项目根目录to
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),  
                 "..")
)
sys.path.append(project_root)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder

from .run_llm_code import run_llm_code
from .preprocessing import (
    make_datasets_numeric,
    split_target_column,
    make_dataset_numeric,
)
from .data import get_X_y
from .stage1 import generate_features
from .stage1_evaluate import filter_generated_features
from .metrics import auc_metric, accuracy_metric
import pandas as pd
from typing import Optional
import json

pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)  
pd.set_option('display.max_colwidth', None)  

class Stage1Classifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        base_classifier: None,
        optimization_metric: str = "accuracy",
        iterations: int = 10,
        llm_model: str = "gpt-3.5-turbo",
        n_splits: int = 10,
        n_repeats: int = 2,
    ) -> None:
        self.base_classifier = base_classifier
        if self.base_classifier is None:
            from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
            import torch
            from functools import partial

            self.base_classifier = TabPFNClassifier(
                N_ensemble_configurations=16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.base_classifier.fit = partial(
                self.base_classifier.fit, overwrite_warning=True
            )
        self.llm_model = llm_model
        self.iterations = iterations
        self.optimization_metric = optimization_metric
        self.n_splits = n_splits
        self.n_repeats = n_repeats

    def fit_pandas(self, df, dataset_description, target_column_name, dataset_name, round_num,task_type = "classification",base_url:str=None,api_key:str=None, logger=None, **kwargs):

        original_columns = list(df.drop(columns=[target_column_name]).columns)

        X, y = (
            df.drop(columns=[target_column_name]).values,
            df[target_column_name].values,
        )
        return self.fit(
            X, y, dataset_description, original_columns, target_column_name, dataset_name, round_num, task_type = task_type, base_url = base_url, api_key = api_key, logger=logger
        )

    def fit(
        self, X, y, dataset_description, original_columns, target_name, dataset_name, round_num,task_type = "classification",base_url:str=None,api_key:str=None, logger=None
        
    ):
        

        self.dataset_description = dataset_description
        self.original_columns = list(original_columns)
        self.target_name = target_name

        self.X_ = X
        self.y_ = y

        if X.shape[0] > 3000 and self.base_classifier.__class__.__name__ == "TabPFNClassifier":
            print(
                "WARNING: TabPFN may take a long time to run on large datasets. Consider using alternatives (e.g. RandomForestClassifier)"
            )
        elif X.shape[0] > 10000 and self.base_classifier.__class__.__name__ == "TabPFNClassifier":
            print("WARNING: Stage1 may take a long time to run on large datasets.")

        ds = [
            "dataset",
            X,
            y,
            [],
            self.original_columns + [target_name],
            {},
            dataset_description,
        ]
        df_train = pd.DataFrame(
            X,
            columns=self.original_columns,
        )
        df_train[target_name] = y

        self.code, prompt, messages, self.code_blocks = generate_features(
            ds,
            df_train,
            model=self.llm_model,
            iterative=self.iterations,
            metric_used=auc_metric,
            iterative_method=self.base_classifier,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            task_type=task_type,
            base_url=base_url,
            api_key=api_key,
            logger=logger
        )
        code_dir = os.path.join(project_root, "tests", "code", task_type)
        os.makedirs(code_dir, exist_ok=True)
        full_code_path = os.path.join(code_dir, f"{dataset_name}_{self.llm_model}_code{round_num}.py")
        with open(full_code_path, "w", encoding="utf-8") as f:
            f.write(self.code)
        with open(full_code_path, "r", encoding="utf-8") as f:
            code = f.read()
        self.code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        df_train_new = run_llm_code(
            self.code,
            df_train,
            self.target_name
        )

        le = LabelEncoder()
        for col in df_train_new.select_dtypes(include=['category']).columns:
            df_train_new[col] = le.fit_transform(df_train_new[col])
        df_train_new.replace([float('inf'), float('-inf')], 0, inplace=True)
        df_train_new.fillna(0, inplace=True)

        new_features_columns = df_train_new.columns.difference(df_train.columns).tolist()
        if new_features_columns:
            self.final_features_columns = filter_generated_features(df_train_new,new_features_columns,target_name,task_type)
        else:
            self.final_features_columns = []
        selected_set = set(self.final_features_columns or [])
        feature_to_code = {}
        ordered_codes = []
        if selected_set and getattr(self, "code_blocks", None):
            base_df = df_train.copy(deep=True)
            base_cols = list(base_df.columns)
            for cb in self.code_blocks:
                if not cb or len(cb.strip()) == 0:
                    continue
                try:
                    df_tmp = run_llm_code(cb, base_df, self.target_name)
                except Exception:
                    continue
                new_cols = [c for c in df_tmp.columns if c not in base_cols]
                hit = False
                for c in new_cols:
                    if c in selected_set and c not in feature_to_code:
                        feature_to_code[c] = cb
                        hit = True
                if hit:
                    ordered_codes.append(cb)

        selected_code_path = os.path.join(code_dir, f"{dataset_name}_{self.llm_model}_selected_code{round_num}.py")
        with open(selected_code_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(ordered_codes))
        selected_map_path = os.path.join(code_dir, f"{dataset_name}_{self.llm_model}_selected_feature_map{round_num}.json")
        with open(selected_map_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "llm_model": self.llm_model,
                    "task_type": task_type,
                    "round_num": round_num,
                    "selected_features": list(self.final_features_columns or []),
                    "feature_to_code": feature_to_code,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        if logger is not None:
            try:
                logger.log(
                    stage="feature_selected",
                    r=round_num,
                    task_type=task_type,
                    selected_feature_count=int(len(self.final_features_columns or [])),
                    selected_features=list(self.final_features_columns or []),
                    selected_code_path=selected_code_path,
                    selected_map_path=selected_map_path,
                )
            except Exception:
                pass

        final_columns = df_train.columns.union(self.final_features_columns).tolist()

        cols = [col for col in final_columns if col != self.target_name] + [self.target_name]
        self.final_columns = cols

        df_train_final = df_train_new[self.final_columns]

        df_train_final, _, self.mappings = make_datasets_numeric(
            df_train_final, df_test=None, target_column=target_name, return_mappings=True
        )

        df_train_final, y = split_target_column(df_train_final, target_name)

        X, y = df_train_final.values, y.values.astype(int)

        self.classes_ = unique_labels(y)

        self.base_classifier.fit(X, y)

        return self

    def predict_preprocess(self, X):


        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=self.X_.columns)
        X, _ = split_target_column(X, self.target_name)

        X = run_llm_code(
            self.code,
            X,
            self.target_name
        )
        le = LabelEncoder()
        for col in X.select_dtypes(include=['category']).columns:
            X[col] = le.fit_transform(X[col])
        X.replace([float('inf'), float('-inf')], 0, inplace=True)
        X.fillna(0, inplace=True)

        predict_columns = [x for x in self.final_columns if x != self.target_name]
        df_train_final = X[predict_columns]

        df_train_final = make_dataset_numeric(df_train_final, mappings=self.mappings)

        df_train_final = df_train_final.values

        return df_train_final



    def predict_proba(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict_proba(X)

    def predict(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict(X)
