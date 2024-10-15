__all__ = [
    "CatBoost",
    "XGBoost",
    "LightGBM",
    "RandomForest",
    "GradientBoost",
    "ExtraTree"
    "Ridge",
    "PLS",
    "Lasso",
    "KNN",
    "ElasticNet",
    "LinearRegression",
    "SGD"
]
from . import CatBoost, XGBoost, LightGBM, RandomForest, GradientBoost, ExtraTree, Ridge, PLS, Lasso, ExtraTree, Ridge, LinearRegression, ElastNet, SGD, KNN

__AlgoList__ = {
    "regression": {
        "randomforest": RandomForest,
        "knn": KNN,
        "elasticnet": ElastNet,
        "ridge": Ridge,
        "lasso": Lasso,
        "xgboost": XGBoost,
        "lightgbm": LightGBM,
        "pls": PLS,
        "linearregression": LinearRegression,
        "catboost": CatBoost,
        "extratree": ExtraTree,
        "sgd": SGD,
        "gradientboost": GradientBoost
    },
    "classification":{
        "randomforest": RandomForest,
        "knn": KNN,
        "elasticnet": ElastNet,
        "ridge": Ridge,
        "lasso": Lasso,
        "xgboost": XGBoost,
        "lightgbm": LightGBM,
        "pls": PLS,
        "linearregression": LinearRegression,
        "catboost": CatBoost,
        "extratree": ExtraTree,
        "sgd": SGD,
        "gradientboost": GradientBoost
    }
}