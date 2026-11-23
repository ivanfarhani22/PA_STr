"""Modul pembuatan label proxy lomba (Strategy Design Pattern)."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from runready.config import LABEL_CONFIG, LabelConfig


def calculate_median_pace(df: pd.DataFrame) -> pd.Series:
    """Menghitung median pace per pelari."""
    return df.groupby("athlete_id")["pace_min_km"].median().rename("median_pace")


def count_prior_activities(
    df: pd.DataFrame, athlete_timestamps: Dict[int, np.ndarray], days: int = 28
) -> pd.Series:
    """Menghitung jumlah aktivitas latihan dalam rentang waktu tertentu sebelum tanggal lomba."""
    # Implementasi efisien berbasis numpy lookup
    counts = []
    for row in df.itertuples():
        t_arr = athlete_timestamps.get(row.athlete_id, np.array([]))
        end = row.timestamp.to_datetime64()
        start = end - pd.Timedelta(days=days).to_timedelta64()
        n = int(((t_arr >= start) & (t_arr < end)).sum())
        counts.append(n)
    return pd.Series(counts, index=df.index, name="n_prior_4w")


class ProxyLabelStrategy:
    """Strategi pembuatan label proxy berdasarkan jarak standar dan kecepatan relatif."""

    def __init__(self, config: LabelConfig = LABEL_CONFIG):
        self.config = config

    def build_labels(
        self, df_clean: pd.DataFrame, speedup: Optional[float] = None
    ) -> pd.DataFrame:
        """Membangun dataset label proxy lomba dari aktivitas bersih."""
        sp = speedup if speedup is not None else self.config.speedup_threshold
        df = df_clean.copy()

        if "median_pace" not in df.columns:
            med = calculate_median_pace(df)
            df = df.join(med, on="athlete_id")

        ts_dict = {a: g["timestamp"].values for a, g in df.groupby("athlete_id")}

        parts = []
        for name, (lo, hi) in self.config.standard_distances.items():
            subset = df[df["distance_km"].between(lo, hi)].copy()
            subset = subset[subset["pace_min_km"] <= subset["median_pace"] * (1 - sp)]
            subset["race_type_std"] = name
            parts.append(subset)

        if not parts:
            return pd.DataFrame()

        r = pd.concat(parts).reset_index(drop=True)
        r["n_prior_4w"] = count_prior_activities(r, ts_dict, days=self.config.prior_days)

        filtered = r[r["n_prior_4w"] >= self.config.min_prior_activities].copy()
        filtered = filtered.rename(
            columns={"timestamp": "race_date", "duration_min": "finish_time_min"}
        )

        cols = [
            "athlete_id",
            "race_date",
            "race_type_std",
            "distance_km",
            "finish_time_min",
            "pace_min_km",
            "elevation_gain_m",
            "elevation_gain_per_km",
            "n_prior_4w",
        ]
        final = filtered[cols].sort_values(["athlete_id", "race_date"]).reset_index(drop=True)
        return final

    def evaluate_sensitivity(self, df_clean: pd.DataFrame) -> Dict[float, pd.DataFrame]:
        """Mengevaluasi sensitivitas berbagai ambang speedup."""
        results = {}
        for sp in self.config.sensitivity_speedups:
            results[sp] = self.build_labels(df_clean, speedup=sp)
        return results
