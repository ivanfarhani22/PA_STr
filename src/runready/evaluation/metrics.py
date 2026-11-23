"""Modul metrik evaluasi regresi."""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Menghitung metrik MAE, RMSE, MAPE (%), dan R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = 100.0 * np.mean(np.abs(y_true - y_pred) / y_true)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}


def evaluate_per_distance(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Evaluasi metrik per kategori jarak lomba."""
    out = {}
    for name, group in df.groupby("race_type_std"):
        idx = group.index
        t, p = y_true[idx], y_pred[idx]
        out[name] = {
            "n": len(group),
            "MAE": mean_absolute_error(t, p),
            "RMSE": np.sqrt(mean_squared_error(t, p)),
            "MAPE_%": 100.0 * np.mean(np.abs(t - p) / t),
            "R2": r2_score(t, p),
        }
    return pd.DataFrame(out).T.round(2)
