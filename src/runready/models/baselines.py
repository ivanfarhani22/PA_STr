"""Modul model baseline untuk pembanding."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def baseline_calibrated(df: pd.DataFrame) -> np.ndarray:
    """Baseline pace x jarak terkalibrasi linier."""
    x = (df["pace_4w"] * df["race_distance_km"]).to_frame("x")
    y = df["target_finish_min"]
    base = LinearRegression().fit(x, y)
    return base.predict(x)


def riegel_approx(df: pd.DataFrame) -> np.ndarray:
    """Baseline formula Riegel."""
    d1 = df["longest_4w"]
    pace = df["pace_4w"]
    dist = df["race_distance_km"]
    # Hindari pembagian dengan nol
    d1_safe = np.where(d1 > 0, d1, dist)
    return (d1_safe * pace * (dist / d1_safe) ** 1.06).values
