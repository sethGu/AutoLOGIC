import os
import sys
# 添加项目根目录到
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   # 当前文件所在目录
                 "..")                  # 
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
from .caafe import generate_features
from .caafe_evaluate import filter_generated_features
from .metrics import auc_metric, accuracy_metric
import pandas as pd
from typing import Optional

# 设置显示的最大行数和列数
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整列宽
pd.set_option('display.max_colwidth', None)  # 不截断列内容

class CAAFEClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses the CAAFE algorithm to generate features and a base classifier to make predictions.

    Parameters:
    base_classifier (object, optional): The base classifier to use. If None, a default TabPFNClassifier will be used. Defaults to None.
    optimization_metric (str, optional): The metric to optimize during feature generation. Can be 'accuracy' or 'auc'. Defaults to 'accuracy'.
    iterations (int, optional): The number of iterations to run the CAAFE algorithm. Defaults to 10.
    llm_model (str, optional): The LLM model to use for generating features. Defaults to 'gpt-3.5-turbo'.
    n_splits (int, optional): The number of cross-validation splits to use during feature generation. Defaults to 10.
    n_repeats (int, optional): The number of times to repeat the cross-validation during feature generation. Defaults to 2.
    """
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

    def fit_pandas(self, df, dataset_description, target_column_name, dataset_name, round_num,task_type = "classification",base_url:str=None,api_key:str=None, **kwargs):
        """
        Fit the classifier to a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame to fit the classifier to.
        dataset_description (str): A description of the dataset.
        target_column_name (str): The name of the target column in the DataFrame.
        **kwargs: Additional keyword arguments to pass to the base classifier's fit method.
        """
        original_columns = list(df.drop(columns=[target_column_name]).columns)

        X, y = (
            df.drop(columns=[target_column_name]).values,
            df[target_column_name].values,
        )
        return self.fit(
            X, y, dataset_description, original_columns, target_column_name, dataset_name, round_num, task_type = task_type, base_url = base_url, api_key = api_key
        )

    def fit(
        self, X, y, dataset_description, original_columns, target_name, dataset_name, round_num,task_type = "classification",base_url:str=None,api_key:str=None
        
    ):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : np.ndarray
            The training data features.
        y : np.ndarray
            The training data target values.
        dataset_description : str
            A description of the dataset.
        feature_names : List[str]
            The names of the features in the dataset.
        target_name : str
            The name of the target variable in the dataset.
        disable_caafe : bool, optional
            Whether to disable the CAAFE algorithm, by default False.
        """

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
            print("WARNING: CAAFE may take a long time to run on large datasets.")

        ds = [
            "dataset",
            X,
            y,
            [],
            self.original_columns + [target_name],
            {},
            dataset_description,
        ]
        # Add X and y as one dataframe
        df_train = pd.DataFrame(
            X,
            columns=self.original_columns,
        )
        df_train[target_name] = y

            # 生成所有的特征代码
        self.code, prompt, messages = generate_features(
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
            api_key=api_key
        )
        with open(f"{project_root}/tests/code/classification/{dataset_name}_{self.llm_model}_code{round_num}.py", "w", encoding="utf-8") as f:
            f.write(self.code)
        with open(f"{project_root}/tests/code/classification/{dataset_name}_{self.llm_model}_code{round_num}.py", "r", encoding="utf-8") as f:
            code = f.read()
        self.code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        # 在训练集上执行所有生成的特征代码
        df_train_new = run_llm_code(
            self.code,
            df_train,
            self.target_name
        )

        le = LabelEncoder()
        # 选择所有的非数值类型列（包括 'category' 类型）进行转换
        for col in df_train_new.select_dtypes(include=['category']).columns:
            df_train_new[col] = le.fit_transform(df_train_new[col])
        # 替换 inf 和 -inf 为 0
        df_train_new.replace([float('inf'), float('-inf')], 0, inplace=True)
        # 替换 NaN 为 0
        df_train_new.fillna(0, inplace=True)

        # 新增特征列
        new_features_columns = df_train_new.columns.difference(df_train.columns).tolist()
        # 筛选特征
        self.final_features_columns = filter_generated_features(df_train_new,new_features_columns,target_name,task_type)

        final_columns = df_train.columns.union(self.final_features_columns).tolist()

        # 将目标列名放在最后
        cols = [col for col in final_columns if col != self.target_name] + [self.target_name]
        self.final_columns = cols

        df_train_final = df_train_new[self.final_columns]

        df_train_final, _, self.mappings = make_datasets_numeric(
            df_train_final, df_test=None, target_column=target_name, return_mappings=True
        )

        df_train_final, y = split_target_column(df_train_final, target_name)

        X, y = df_train_final.values, y.values.astype(int)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.base_classifier.fit(X, y)

        # Return the classifier
        return self

    def predict_preprocess(self, X):
        """
        Helper functions for preprocessing the data before making predictions.

        Parameters:
        X (pandas.DataFrame): The DataFrame to make predictions on.

        Returns:
        numpy.ndarray: The preprocessed input data.
        """
        # check_is_fitted(self)

        if type(X) != pd.DataFrame:
            X = pd.DataFrame(X, columns=self.X_.columns)
        X, _ = split_target_column(X, self.target_name)

        X = run_llm_code(
            self.code,
            X,
            self.target_name
        )
        le = LabelEncoder()
        # 选择所有的非数值类型列（包括 'category' 类型）进行转换
        for col in X.select_dtypes(include=['category']).columns:
            X[col] = le.fit_transform(X[col])
        # 替换 inf 和 -inf 为 0
        X.replace([float('inf'), float('-inf')], 0, inplace=True)
        # 替换 NaN 为 0
        X.fillna(0, inplace=True)

        predict_columns = [x for x in self.final_columns if x != self.target_name]
        df_train_final = X[predict_columns]

        df_train_final = make_dataset_numeric(df_train_final, mappings=self.mappings)

        df_train_final = df_train_final.values

        return df_train_final


    """
    predict_proba 方法用于预测输入数据的类别概率。它首先调用 predict_preprocess 方法对数据进行预处理，然后调用基础分类器的 predict_proba 方法进行预测。
    """
    def predict_proba(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict_proba(X)

    """
    predict 方法用于预测输入数据的类别标签。它同样先调用 predict_preprocess 方法对数据进行预处理，然后调用基础分类器的 predict 方法进行预测。
    """
    def predict(self, X):
        X = self.predict_preprocess(X)
        return self.base_classifier.predict(X)
