"""Modul kalibrasi dan Brier score untuk evaluasi kesiapan."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from runready.config import READINESS_CONFIG, ReadinessConfig


def evaluate_calibration(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    residuals: np.ndarray,
    mode: str = "rel",
    config: ReadinessConfig = READINESS_CONFIG,
) -> Tuple[float, float, float, pd.DataFrame]:
    """Menghitung Brier score, referensi, skill score, dan tabel kalibrasi per bin."""
    ks = config.cot_multipliers
    
    # Hitung probabilitas kesiapan untuk tiap k
    P_list, O_list = [], []
    s_resid = np.sort(residuals)
    n_res = len(s_resid)

    for k in ks:
        cot = y_hat * k
        thr = (cot / y_hat - 1) if mode == "rel" else (cot - y_hat)
        # Proporsi residual historis yang membuat estimasi akhir <= COT
        prob = np.searchsorted(s_resid, thr, side="right") / n_res
        obs = (y_true <= cot).astype(float)
        
        P_list.append(prob)
        O_list.append(obs)

    P = np.concatenate(P_list)
    O = np.concatenate(O_list)

    brier = float(np.mean((P - O) ** 2))
    ref = float(O.mean() * (1 - O.mean()))
    skill = float(1 - brier / ref) if ref > 0 else 0.0

    bins = config.calibration_bins
    cat_bins = pd.cut(P, bins, include_lowest=True)
    
    df_calib = pd.DataFrame({"pred": P, "obs": O})
    table = df_calib.groupby(cat_bins, observed=True).agg(
        n=("obs", "size"),
        prediksi=("pred", "mean"),
        teramati=("obs", "mean"),
    ).round(2)

    return brier, ref, skill, table
