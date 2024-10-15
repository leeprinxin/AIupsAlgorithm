from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd
from ._Regressor import _Regressor
from ._Classifier import _Classifier
from .common_util import *

class Regressor(_Regressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        self.model = RandomForestRegressor(**self.kwargs)
        self.model.fit(X, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.model.predict(X)

    @property
    def feature_importances_(self) -> list[float] or np.ndarray:
        return np.nan_to_num((np.nan_to_num(self.model.feature_importances_)/sum(np.nan_to_num(self.model.feature_importances_)))).tolist()

    def export_original_model(self):
        return self.model

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)

    def get_params(self):
        return self.model.get_params()

class Classifier(_Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        self.model = RandomForestClassifier(**self.kwargs)
        self.model.fit(X, y)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> list[float] or np.ndarray:
        return np.nan_to_num((np.nan_to_num(self.model.feature_importances_)/sum(np.nan_to_num(self.model.feature_importances_)))).tolist()


    def export_original_model(self) :
        return self.model

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)

    def get_params(self,  **kwargs):
        return self.model.get_params()