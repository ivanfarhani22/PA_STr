"""Modul ekstraksi fitur riwayat latihan dan karakteristik lomba."""

from typing import Dict, List
import pandas as pd
import numpy as np

from runready.config import FEATURE_CONFIG, FeatureConfig


def extract_window_features(
    history_df: pd.DataFrame, race_date: pd.Timestamp, weeks: int
) -> Dict[str, float]:
    """Mengekstrak fitur riwayat latihan dalam jendela waktu (minggu) sebelum tanggal lomba."""
    start = race_date - pd.Timedelta(weeks=weeks)
    x = history_df[(history_df["timestamp"] >= start) & (history_df["timestamp"] < race_date)]

    if len(x) == 0:
        return {
            f"km_wk_{weeks}w": 0.0,
            f"pace_{weeks}w": 0.0,
            f"speed_{weeks}w_kmh": 0.0,
            f"elev_{weeks}w": 0.0,
            f"n_{weeks}w": 0,
            f"longest_{weeks}w": 0.0,
            f"hr_{weeks}w": np.nan,
        }

    km = float(x["distance_km"].sum())
    mnt = float(x["duration_min"].sum())
    pace = mnt / km if km > 0 else 0.0
    speed = (60 * km / mnt) if mnt > 0 else 0.0

    hr_val = float(x["heart_rate_avg"].mean()) if "heart_rate_avg" in x.columns and not x["heart_rate_avg"].isna().all() else np.nan

    return {
        f"km_wk_{weeks}w": km / weeks,
        f"pace_{weeks}w": pace,
        f"speed_{weeks}w_kmh": speed,
        f"elev_{weeks}w": float(x["elevation_gain_m"].sum()),
        f"n_{weeks}w": int(len(x)),
        f"longest_{weeks}w": float(x["distance_km"].max()),
        f"hr_{weeks}w": hr_val,
    }


def build_feature_table(
    activities_df: pd.DataFrame,
    races_df: pd.DataFrame,
    config: FeatureConfig = FEATURE_CONFIG,
) -> pd.DataFrame:
    """Membangun tabel fitur lengkap menggabungkan riwayat latihan dan karakteristik lomba."""
    hist = {a: g.sort_values("timestamp") for a, g in activities_df.groupby("athlete_id")}

    rows = []
    for r in races_df.itertuples():
        athlete_hist = hist.get(r.athlete_id, pd.DataFrame())
        feat_dict = {}
        for w in config.windows_weeks:
            feat_dict.update(extract_window_features(athlete_hist, r.race_date, w))
        rows.append(feat_dict)

    feat_df = pd.concat([races_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    # Karakteristik lomba
    feat_df["race_distance_km"] = feat_df["distance_km"]
    feat_df["race_elev_m"] = feat_df["elevation_gain_m"]
    feat_df["race_elev_per_km"] = feat_df["elevation_gain_per_km"]
    feat_df["km_effort"] = feat_df["distance_km"] + feat_df["elevation_gain_m"] / 100

    if "finish_time_min" in feat_df.columns:
        feat_df = feat_df.rename(columns={"finish_time_min": "target_finish_min"})

    return feat_df
