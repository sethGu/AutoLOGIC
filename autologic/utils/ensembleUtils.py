import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from typing import Union, Optional, Dict
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone

def stacking_ensemble(
    base_preds_train: np.ndarray,
    base_preds_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    meta_learner: Optional[BaseEstimator] = None,
    return_metrics: bool = False
) -> Union[None, Dict[str, float]]:
    """
    Perform classification_stacking ensemble using base model predictions.

    Parameters:
    - base_preds_train: ndarray, shape (n_train_samples, n_base_models)
        Predictions from base models on training data.
    - base_preds_test: ndarray, shape (n_test_samples, n_base_models)
        Predictions from base models on test data.
    - y_train: ndarray, shape (n_train_samples,)
        Ground truth labels for training data.
    - y_test: ndarray, shape (n_test_samples,)
        Ground truth labels for test data.
    - meta_learner: scikit-learn estimator (default = LogisticRegression)
        The metamodel to combine base predictions.
    - return_metrics: bool
        Whether to return a dictionary of evaluation metrics.

    Returns:
    - dict of metrics if return_metrics=True, else None.
    """

    if meta_learner is None:
        meta_learner = LogisticRegression(random_state=42)

    # Validate input dimensions
    assert base_preds_train.shape[0] == y_train.shape[0], "Mismatch in training samples"
    assert base_preds_test.shape[0] == y_test.shape[0], "Mismatch in test samples"
    assert base_preds_train.shape[1] == base_preds_test.shape[1], "Mismatch in number of base models"

    # Train meta-learner
    meta_learner.fit(base_preds_train, y_train)

    # Predict class labels
    stacking_preds = meta_learner.predict(base_preds_test)

    # Predict probabilities (for AUC)
    try:
        stacking_probs = meta_learner.predict_proba(base_preds_test)[:, 1]
        auc = roc_auc_score(y_test, stacking_probs)
    except AttributeError:
        stacking_probs = None
        auc = 'N/A'

    # Evaluation metrics
    acc = accuracy_score(y_test, stacking_preds)
    precision = precision_score(y_test, stacking_preds, zero_division=0)
    recall = recall_score(y_test, stacking_preds, zero_division=0)
    f1 = f1_score(y_test, stacking_preds, zero_division=0)

    if return_metrics:
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc
        }


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def voting_ensemble(fitted_models, train_x, train_y, test_x, test_y=None):
    """
    Ensemble multiple trained models for soft voting prediction and calculate evaluation metrics.

    Parameters:
    - fitted_models: list of fitted model instances, each model should support .predict_proba()
    - train_x, train_y: Training data (unused, optional)
    - test_x: Test set features
    - test_y: Test set labels (for computing evaluation metrics)

    Returns:
    - If test_y is provided: Returns a tuple containing five metrics (acc, per, rec, f1, auc)
    - Otherwise: Returns numpy.ndarray of probability predictions (1D array, positive class probabilities)
    """
    # Average probabilities for soft voting
    probas = []
    for model in fitted_models:
        proba = model.predict_proba(test_x)
        if proba.ndim == 2 and proba.shape[1] > 1:
            proba = proba[:, 1]
        probas.append(proba)

    avg_proba = np.mean(probas, axis=0)

    if test_y is not None:
        pred = (avg_proba >= 0.5).astype(int)
        acc = accuracy_score(test_y, pred)
        per = precision_score(test_y, pred)
        rec = recall_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        auc = roc_auc_score(test_y, avg_proba)
        return acc, per, rec, f1, auc

    return avg_proba



def getMetaModel_list(modelNameList):
    """
    Returns a list of initialized classification models corresponding to the model name list.

    Supported model names include:
    - 'RandomForestClassifier'
    - 'XGBClassifier'
    - 'LGBMClassifier'
    - 'CatBoostClassifier'
    - 'SVC'
    - 'DecisionTreeClassifier'
    - 'LogisticRegression'
    - 'BaggingClassifier'

    Parameters:
    - modelNameList (List[str]): List of model name strings

    Returns:
    - List[object]: List of model instances (initialized)
    """

    from sklearn.ensemble import (
        RandomForestClassifier,
        BaggingClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    model_dict = {
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LGBMClassifier': LGBMClassifier(random_state=42),
        'CatBoostClassifier': CatBoostClassifier(verbose=0, random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'BaggingClassifier': BaggingClassifier(random_state=42)
    }

    models = []
    for name in modelNameList:
        if name not in model_dict:
            raise ValueError(f"Model name '{name}' is not supported, please check spelling or add definition.")
        models.append(model_dict[name])

    return models

def get_regression_metaModel(modelNameList):
    """
    Returns a list of initialized regression models corresponding to the model name list.

    Supported model names include:
    - 'RandomForestRegressor'
    - 'XGBRegressor'
    - 'LGBMRegressor'
    - 'CatBoostRegressor'
    - 'SVR'
    - 'DecisionTreeRegressor'
    - 'LinearRegression'
    - 'Ridge'
    - 'Lasso'
    - 'ElasticNet'
    - 'MLPRegressor'
    - 'KNeighborsRegressor'
    - 'BaggingRegressor'

    Parameters:
    - modelNameList (List[str]): List of model name strings

    Returns:
    - List[object]: List of initialized regression model instances
    """

    from sklearn.ensemble import (
        RandomForestRegressor,
        BaggingRegressor
    )
    from sklearn.linear_model import (
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet
    )
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    model_dict = {
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBRegressor': XGBRegressor(objective='reg:squarederror', random_state=42),
        'LGBMRegressor': LGBMRegressor(random_state=42),
        'CatBoostRegressor': CatBoostRegressor(verbose=0, random_state=42),
        'SVR': SVR(),
        'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'MLPRegressor': MLPRegressor(random_state=42, max_iter=1000),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'BaggingRegressor': BaggingRegressor(random_state=42)
    }

    models = []
    for name in modelNameList:
        if name not in model_dict:
            raise ValueError(f"Model name '{name}' is not supported, please check spelling or add definition.")
        models.append(model_dict[name])

    return models
