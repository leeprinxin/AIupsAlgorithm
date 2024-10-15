import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import sys
from typing import Union

class _Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        """
        __init__(self, 演算法超參數定義)
        """
        super().__init__()
        self.kwargs = kwargs


    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            **kwargs):
        """
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
        y : pd.Series or np.ndarray
        """
        pass


    def predict(self,
                X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray

        Returns
        ----------
        y_hat : ndarray of shape (n_samples,)
        """
        pass

    @property
    def feature_importances_(self) -> list[float] or np.ndarray:
        """
        Returns
        ----------
        feature_importances : list[float] or np.ndarray。
        """
        pass


    def export_original_model(self):
        pass

    def get_params(self, **kwargs):
        pass

    def set_params(self):
        pass

    def __str__(self):
        return f"{self.model}"