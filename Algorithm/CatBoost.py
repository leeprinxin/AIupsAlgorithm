import catboost
import numpy as np
import pandas as pd
from ._Regressor import _Regressor
from ._Classifier import _Classifier
from .common_util import *
from .common_util import is_gpu, gpu_id

class Regressor(_Regressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if is_gpu:
            self.kwargs.update(
                {"task_type": 'GPU', "devices": str(gpu_id), 'bootstrap_type': 'Poisson'}
            )
        else:
            keys_to_delete = ["task_type", "devices", "bootstrap_type"]
            for key in keys_to_delete:
                self.kwargs.pop(key, None)
        self.model = None


    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        self.model = catboost.CatBoostRegressor(**self.kwargs)
        self.model.fit(X, y, verbose=False)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> list[float] or np.ndarray:
        return np.nan_to_num((np.nan_to_num(self.model.feature_importances_)/sum(np.nan_to_num(self.model.feature_importances_)))).tolist()

    def export_original_model(self):
        return self.model

    def set_params(self, **kwargs):
        if self.model is None:
            if is_gpu:
                kwargs.update(
                    {"task_type": 'GPU', "devices": str(gpu_id), 'bootstrap_type': 'Poisson'}
                )
            else:
                keys_to_delete = ["task_type", "devices", "bootstrap_type"]
                for key in keys_to_delete:
                    kwargs.pop(key, None)

        self.model.set_params(**kwargs)

    def get_params(self, **kwargs):
        return self.model.get_params()

class Classifier(_Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if is_gpu:
            self.kwargs.update(
                {"task_type": 'GPU', "devices": str(gpu_id), 'bootstrap_type': 'Poisson'}
            )
        else:
            keys_to_delete = ["task_type", "devices", "bootstrap_type"]
            for key in keys_to_delete:
                self.kwargs.pop(key, None)
        self.model = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        self.model = catboost.CatBoostClassifier(**self.kwargs)
        self.model.fit(X, y, verbose=False)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> list[float] or np.ndarray:
        return np.nan_to_num((np.nan_to_num(self.model.feature_importances_)/sum(np.nan_to_num(self.model.feature_importances_)))).tolist()

    def export_original_model(self):
        return self.model

    def set_params(self, **kwargs):
        if self.model is None:
            if is_gpu:
                kwargs.update(
                    {"task_type": 'GPU', "devices": str(gpu_id), 'bootstrap_type': 'Poisson'}
                )
            else:
                keys_to_delete = ["task_type", "devices", "bootstrap_type"]
                for key in keys_to_delete:
                    kwargs.pop(key, None)

        self.model.set_params(**kwargs)

    def get_params(self, **kwargs):
        return self.model.get_params()