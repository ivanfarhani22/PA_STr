"""Modul pembersihan data aktivitas runner (Pipeline Design Pattern)."""

from dataclasses import dataclass
import logging
from typing import Callable, List, Tuple
import pandas as pd

from runready.config import CLEANING_CONFIG, CleaningConfig

logger = logging.getLogger(__name__)


@dataclass
class CleaningStepResult:
    """Hasil dari eksekusi satu langkah pembersihan data."""
    step_name: str
    description: str
    rows_dropped: int
    rows_remaining: int


class CleaningStep:
    """Satu langkah independen dalam pipeline pembersihan data."""

    def __init__(self, name: str, description: str, mask_fn: Callable[[pd.DataFrame], pd.Series]):
        self.name = name
        self.description = description
        self.mask_fn = mask_fn

    def apply(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningStepResult]:
        """Terapkan mask filter ke dataframe dan kembalikan dataframe bersih beserta log."""
        drop_mask = self.mask_fn(df)
        n_dropped = int(drop_mask.sum())
        df_clean = df[~drop_mask].copy()
        result = CleaningStepResult(
            step_name=self.name,
            description=self.description,
            rows_dropped=n_dropped,
            rows_remaining=len(df_clean),
        )
        return df_clean, result


class CleaningPipeline:
    """Pipeline pembersihan data berurutan."""

    def __init__(self, steps: List[CleaningStep], config: CleaningConfig = CLEANING_CONFIG):
        self.steps = steps
        self.config = config

    def execute(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[CleaningStepResult]]:
        """Menjalankan seluruh langkah pembersihan secara berurutan."""
        df = df_raw.copy()
        results: List[CleaningStepResult] = []

        # Langkah 1: Timestamp < min_year
        step_ts = CleaningStep(
            "filter_timestamp",
            f"timestamp sebelum {self.config.min_year}",
            lambda d: d["timestamp"].dt.year < self.config.min_year,
        )
        df, res = step_ts.apply(df)
        results.append(res)

        # Langkah 2: Duplikat (athlete_id, timestamp)
        step_dup = CleaningStep(
            "filter_duplicates",
            "duplikat athlete+timestamp",
            lambda d: d.duplicated(subset=["athlete_id", "timestamp"]),
        )
        df, res = step_dup.apply(df)
        results.append(res)

        # Langkah 3: Jarak/durasi <= 0
        step_invalid = CleaningStep(
            "filter_invalid_distance_duration",
            "jarak/durasi <= 0",
            lambda d: (d["distance_km"] <= 0) | (d["duration_min"] <= 0),
        )
        df, res = step_invalid.apply(df)
        results.append(res)

        # Langkah 4: Durasi > max_duration_hours
        step_dur = CleaningStep(
            "filter_max_duration",
            f"durasi > {int(self.config.max_duration_hours)} jam",
            lambda d: d["duration_min"] > self.config.max_duration_hours * 60,
        )
        df, res = step_dur.apply(df)
        results.append(res)

        # Rekalkulasi pace dan elevasi per km agar konsisten
        df["pace_min_km"] = df["duration_min"] / df["distance_km"]
        df["elevation_gain_per_km"] = df["elevation_gain_m"] / df["distance_km"]

        # Langkah 5: Pace di luar range
        step_pace = CleaningStep(
            "filter_pace_range",
            f"pace di luar {self.config.min_pace}-{self.config.max_pace} menit/km",
            lambda d: ~d["pace_min_km"].between(self.config.min_pace, self.config.max_pace),
        )
        df, res = step_pace.apply(df)
        results.append(res)

        # Langkah 6: Elevasi per km > max_elevation_per_km
        step_elev = CleaningStep(
            "filter_elevation_range",
            f"elevasi/km > {int(self.config.max_elevation_per_km)} m/km",
            lambda d: d["elevation_gain_per_km"] > self.config.max_elevation_per_km,
        )
        df, res = step_elev.apply(df)
        results.append(res)

        # Hapus kolom yang tidak terpakai/kosong
        for col in self.config.drop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Urutkan per pelari dan waktu
        df = df.sort_values(["athlete_id", "timestamp"]).reset_index(drop=True)
        return df, results


def clean_kaggle_activities(
    df_raw: pd.DataFrame, config: CleaningConfig = CLEANING_CONFIG
) -> Tuple[pd.DataFrame, List[CleaningStepResult]]:
    """Fungsi utama untuk membersihkan data mentah kaggle."""
    pipeline = CleaningPipeline(steps=[], config=config)
    return pipeline.execute(df_raw)
