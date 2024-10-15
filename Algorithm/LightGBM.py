import lightgbm as lgb
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
                {"device": 'gpu', "gpu_platform_id": 0, "gpu_device_id": gpu_id}
            )
        else:
            self.kwargs.update(
                {"device": 'cpu'}
            )
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        self.model = lgb.LGBMRegressor(**self.kwargs)
        self.model.fit(X, y, verbose=False)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if is_gpu:
            self.model.set_params(**{"device": 'gpu', "gpu_platform_id": 0, "gpu_device_id": gpu_id})
        else:
            self.model.set_params(**{"device": 'cpu'})
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> list[float] or np.ndarray:
        return np.nan_to_num((np.nan_to_num(self.model.feature_importances_) / sum(np.nan_to_num(self.model.feature_importances_)))).tolist()

    def export_original_model(self):
        return self.model

    def set_params(self, **kwargs):
        if is_gpu:
            kwargs.update(
                {"device": 'gpu', "gpu_platform_id": 0, "gpu_device_id": gpu_id}
            )
        else:
            kwargs.update(
                {"device": 'cpu'}
            )
        self.model.set_params(**kwargs)

    def get_params(self):
        return self.model.get_params()

class Classifier(_Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if is_gpu:
            self.kwargs.update(
                {"device": 'gpu', "gpu_platform_id": 0, "gpu_device_id": gpu_id}
            )
        else:
            self.kwargs.update(
                {"device": 'cpu'}
            )


    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        self.model = lgb.LGBMClassifier(**kwargs)
        self.model.fit(X, y, verbose=False)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if is_gpu:
            self.model.set_params(**{"device": 'gpu', "gpu_platform_id": 0, "gpu_device_id": gpu_id})
        else:
            self.model.set_params(**{"device": 'cpu'})
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        if is_gpu:
            self.model.set_params(**{"tree_method": 'gpu_hist', "gpu_id": gpu_id, "predictor": "gpu_predictor"})
        else:
            self.model.set_params(**{"tree_method": 'auto'})
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> list[float] or np.ndarray:
        return np.nan_to_num((np.nan_to_num(self.model.feature_importances_) / sum(np.nan_to_num(self.model.feature_importances_)))).tolist()

    def export_original_model(self):
        return self.model

    def set_params(self, **kwargs):
        if is_gpu:
            kwargs.update(
                {"device": 'gpu', "gpu_platform_id": 0, "gpu_device_id": gpu_id}
            )
        else:
            kwargs.update(
                {"device": 'cpu'}
            )
        self.model.set_params(**kwargs)

    def get_params(self,  **kwargs):
        return self.model.get_params()