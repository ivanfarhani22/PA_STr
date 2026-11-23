"""Modul pembagian dataset train/test berbasis pelari."""

import numpy as np
import pandas as pd

from runready.config import SPLIT_CONFIG, SplitConfig


def split_features_by_athlete(
    features_df: pd.DataFrame, config: SplitConfig = SPLIT_CONFIG
) -> pd.DataFrame:
    """Membagi dataset fitur berdasarkan pelari (bukan baris aktivitas) untuk menghindari data leakage."""
    df = features_df.copy()

    if "race_id" not in df.columns:
        df.insert(0, "race_id", [f"R{i:04d}" for i in range(len(df))])

    if "label_type" not in df.columns:
        df["label_type"] = "proxy"

    athletes = np.sort(df["athlete_id"].unique())
    rng = np.random.default_rng(config.random_seed)
    n_test = round(config.test_size_athlete_ratio * len(athletes))
    test_athletes = set(rng.choice(athletes, size=n_test, replace=False))

    df["split"] = np.where(df["athlete_id"].isin(test_athletes), "test", "train")
    return df
