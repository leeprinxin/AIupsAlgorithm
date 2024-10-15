from sklearn.linear_model import SGDRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ._Regressor import _Regressor
from ._Classifier import _Classifier
from .common_util import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Regressor(_Regressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scaler = StandardScaler()
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], **kwargs):
        self.X_train = X
        self.y_train = y

        # Fit the scaler on the training data and transform it
        X_scaled = self.scaler.fit_transform(X)

        # Initialize and fit the SGDRegressor model
        self.model = SGDRegressor(**self.kwargs)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # Transform the input data using the fitted scaler
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    @property
    def feature_importances_(self) -> list[float] or np.ndarray:
        scaler = MinMaxScaler(feature_range=(0, 1))
        coefficients = self.model.coef_
        scaled_coefficients = scaler.fit_transform(coefficients.reshape(-1, 1))
        return np.nan_to_num(scaled_coefficients / sum(scaled_coefficients)).tolist()

    def export_original_model(self):
        return self.model

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)

    def get_params(self,  **kwargs):
        return self.model.get_params()