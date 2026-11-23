"""Modul registri model (Factory Design Pattern) untuk estimator machine learning."""

from typing import Any
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from runready.config import SPLIT_CONFIG


def get_model(name: str, **kwargs: Any) -> RegressorMixin:
    """Factory untuk mengembalikan instance model berdasarkan nama dengan hyperparameter yang dikunci."""
    seed = kwargs.pop("random_seed", SPLIT_CONFIG.random_seed)

    if name == "LinReg":
        return LinearRegression(**kwargs)
    elif name == "RF":
        params = {"n_estimators": 500, "min_samples_leaf": 3, "random_state": seed, "n_jobs": -1}
        params.update(kwargs)
        return RandomForestRegressor(**params)
    elif name == "XGB":
        params = {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "random_state": seed,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return XGBRegressor(**params)
    else:
        raise ValueError(f"Model tidak dikenal: {name}. Pilih dari: LinReg, RF, XGB.")
