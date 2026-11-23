"""Modul pelatihan model dan out-of-fold (OOF) cross-validation."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from runready.models.registry import get_model


def oof_predict(
    model: Any, df: pd.DataFrame, features: List[str] if False else list, n_splits: int = 5
) -> np.ndarray:
    """Melakukan prediksi Out-of-Fold (OOF) menggunakan GroupKFold per athlete_id.
    Target regresi dilatih pada pace (menit/km), lalu dikalikan jarak lomba."""
    g = df["athlete_id"]
    y = df["target_finish_min"]
    dist = df["race_distance_km"]

    pred = np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, val_idx in gkf.split(df, y, groups=g):
        m = clone(model)
        X_tr = df.loc[train_idx, features]
        y_tr = (y / dist).iloc[train_idx]
        
        m.fit(X_tr, y_tr)
        
        X_va = df.loc[val_idx, features]
        pred[val_idx] = m.predict(X_va) * dist.iloc[val_idx]

    return pred


def fit_final_model(
    df: pd.DataFrame, model_name: str = "LinReg", features: List[str] = None
) -> Tuple[Any, np.ndarray]:
    """Melatih model final pada seluruh data train dan menghitung residual relatif OOF."""
    model = get_model(model_name)
    y = df["target_finish_min"]
    dist = df["race_distance_km"]

    oof_preds = oof_predict(model, df, features)
    residuals_rel = (y.values - oof_preds) / oof_preds

    # Fit final model pada seluruh data
    final_estimator = clone(model)
    final_estimator.fit(df[features], y / dist)

    return final_estimator, residuals_rel
