"""Konfigurasi terpusat untuk proyek RunReady."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Base Directory (Akar Proyek)
BASE_DIR: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataPaths:
    """Path file data dan artefak model."""
    raw_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    models_dir: Path = BASE_DIR / "models"
    reports_dir: Path = BASE_DIR / "reports"
    figures_dir: Path = BASE_DIR / "reports" / "figures"

    kaggle_raw: Path = BASE_DIR / "data" / "raw" / "standard_kaggle.csv"
    marathon_raw: Path = BASE_DIR / "data" / "raw" / "standard_marathon.csv"
    utmb_raw: Path = BASE_DIR / "data" / "raw" / "standard_utmb_races.csv"

    clean_activities: Path = BASE_DIR / "data" / "processed" / "activities_clean.csv"
    race_labels: Path = BASE_DIR / "data" / "processed" / "race_labels.csv"
    features: Path = BASE_DIR / "data" / "processed" / "features.csv"
    features_split: Path = BASE_DIR / "data" / "processed" / "features_split.csv"

    oof_predictions: Path = BASE_DIR / "reports" / "oof_predictions.csv"
    model_comparison: Path = BASE_DIR / "reports" / "model_comparison.csv"

    final_model: Path = BASE_DIR / "models" / "final_model.pkl"
    residuals_rel: Path = BASE_DIR / "models" / "residuals_rel.npy"


@dataclass(frozen=True)
class CleaningConfig:
    """Parameter ambang pembersihan data kaggle."""
    min_year: int = 2010
    max_duration_hours: float = 24.0
    min_pace: float = 2.5
    max_pace: float = 15.0
    max_elevation_per_km: float = 300.0
    drop_columns: Tuple[str, ...] = ("cadence_avg",)


@dataclass(frozen=True)
class LabelConfig:
    """Parameter pembuat label proxy lomba."""
    standard_distances: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "5K": (4.8, 5.6),
            "10K": (9.6, 11.0),
            "HM": (20.5, 22.5),
            "M": (41.0, 43.5),
        }
    )
    speedup_threshold: float = 0.08
    prior_days: int = 28
    min_prior_activities: int = 4
    sensitivity_speedups: Tuple[float, ...] = (0.05, 0.08, 0.10)


@dataclass(frozen=True)
class FeatureConfig:
    """Parameter ekstraksi fitur riwayat latihan dan lomba."""
    windows_weeks: Tuple[int, ...] = (4, 8, 12)
    primary_window_weeks: int = 4
    full_features: Tuple[str, ...] = (
        "km_wk_4w",
        "pace_4w",
        "speed_4w_kmh",
        "elev_4w",
        "n_4w",
        "longest_4w",
        "race_distance_km",
        "race_elev_m",
        "race_elev_per_km",
        "km_effort",
    )
    reduced_features: Tuple[str, ...] = (
        "km_wk_4w",
        "speed_4w_kmh",
        "race_distance_km",
    )


@dataclass(frozen=True)
class SplitConfig:
    """Parameter pembagian dataset train/test."""
    test_size_athlete_ratio: float = 0.20
    random_seed: int = 42


@dataclass(frozen=True)
class ReadinessConfig:
    """Parameter kalkulasi kesiapan dan kalibrasi."""
    cot_multipliers: Tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
    calibration_bins: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    categories: Tuple[Tuple[float, str], ...] = (
        (90.0, "Sangat Siap"),
        (75.0, "Siap"),
        (50.0, "Cukup Siap"),
        (25.0, "Kurang Siap"),
        (0.0, "Belum Siap"),
    )
    out_of_range_km_effort: float = 50.0
    out_of_range_elev_per_km: float = 30.0


@dataclass(frozen=True)
class UTMBConfig:
    """Parameter analisis dataset UTMB."""
    trail_threshold_m_per_km: float = 20.0
    min_distance_km: float = 5.0
    min_finishers: int = 10


# Instansiasi default konfigurasi global
PATHS = DataPaths()
CLEANING_CONFIG = CleaningConfig()
LABEL_CONFIG = LabelConfig()
FEATURE_CONFIG = FeatureConfig()
SPLIT_CONFIG = SplitConfig()
READINESS_CONFIG = ReadinessConfig()
UTMB_CONFIG = UTMBConfig()
