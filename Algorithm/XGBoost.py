from xgboost import XGBRegressor, XGBClassifier
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
                {"tree_method": 'gpu_hist', "gpu_id": gpu_id, "predictor": "gpu_predictor"}
            )
        else:
            self.kwargs.update(
                {"tree_method": 'auto'}
            )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        self.model = XGBRegressor(**self.kwargs)
        self.model.fit(X, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if is_gpu:
            self.model.set_params(**{"tree_method": 'gpu_hist', "gpu_id": gpu_id, "predictor": "gpu_predictor"})
        else:
            self.model.set_params(**{"tree_method": 'auto'})
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> list[float] or np.ndarray:
        return np.nan_to_num(self.model.feature_importances_).tolist()

    def export_original_model(self):
        return self.model

    def set_params(self, **kwargs):
        if is_gpu:
            kwargs.update(
                {"tree_method": 'gpu_hist', "gpu_id": gpu_id, "predictor": "gpu_predictor"}
            )
        else:
            kwargs.update(
                {"tree_method": 'auto'}
            )
        self.model.set_params(**kwargs)

    def get_params(self,  **kwargs):
        return self.model.get_params()

class Classifier(_Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if is_gpu:
            self.kwargs.update(
                {"tree_method": 'gpu_hist', "gpu_id": gpu_id, "predictor": "gpu_predictor"}
            )
        else:
            self.kwargs.update(
                {"tree_method": 'auto'}
            )


    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        self.model = XGBClassifier(**self.kwargs)
        self.model.fit(X, y)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if is_gpu:
            self.model.set_params(**{"tree_method": 'gpu_hist', "gpu_id": gpu_id, "predictor": "gpu_predictor"})
        else:
            self.model.set_params(**{"tree_method": 'auto'})
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        if is_gpu:
            self.model.set_params(**{"tree_method": 'gpu_hist', "gpu_id": gpu_id, "predictor": "gpu_predictor"})
        else:
            self.model.set_params(**{"tree_method": 'auto'})
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> list[float] or np.ndarray:
        return np.nan_to_num(self.model.feature_importances_).tolist()

    def export_original_model(self):
        return self.model

    def set_params(self, **kwargs):
        if is_gpu:
            kwargs.update(
                {"tree_method": 'gpu_hist', "gpu_id": gpu_id, "predictor": "gpu_predictor"}
            )
        else:
            kwargs.update(
                {"tree_method": 'auto'}
            )
        self.model.set_params(**kwargs)

    def get_params(self,  **kwargs):
        return self.model.get_params()
