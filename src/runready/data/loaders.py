"""Repository sederhana untuk memuat dan menyimpan data serta artefak model."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd

from runready.config import PATHS


def load_kaggle_raw(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Memuat data mentah Kaggle."""
    target_path = Path(path) if path else PATHS.kaggle_raw
    return pd.read_csv(target_path, parse_dates=["timestamp"])


def load_marathon_raw(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Memuat data mentah marathon eksternal."""
    target_path = Path(path) if path else PATHS.marathon_raw
    return pd.read_csv(target_path)


def load_utmb_raw(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Memuat data mentah event UTMB."""
    target_path = Path(path) if path else PATHS.utmb_raw
    return pd.read_csv(target_path)


def load_clean_activities(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Memuat data aktivitas yang sudah dibersihkan."""
    target_path = Path(path) if path else PATHS.clean_activities
    return pd.read_csv(target_path, parse_dates=["timestamp"])


def load_race_labels(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Memuat data label proxy lomba."""
    target_path = Path(path) if path else PATHS.race_labels
    return pd.read_csv(target_path, parse_dates=["race_date"])


def load_features(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Memuat data tabel fitur."""
    target_path = Path(path) if path else PATHS.features
    return pd.read_csv(target_path, parse_dates=["race_date"])


def load_features_split(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Memuat data tabel fitur yang sudah dibagi train/test."""
    target_path = Path(path) if path else PATHS.features_split
    return pd.read_csv(target_path, parse_dates=["race_date"])


def save_dataframe(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> None:
    """Menyimpan DataFrame ke CSV, otomatis membuat folder orang tua jika belum ada."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=index)


def save_model_artifacts(
    model_payload: Dict[str, Any],
    residuals_rel: np.ndarray,
    model_path: Optional[Union[str, Path]] = None,
    residuals_path: Optional[Union[str, Path]] = None,
) -> None:
    """Menyimpan dictionary model dan array residual relatif ke folder models."""
    m_path = Path(model_path) if model_path else PATHS.final_model
    r_path = Path(residuals_path) if residuals_path else PATHS.residuals_rel

    m_path.parent.mkdir(parents=True, exist_ok=True)
    r_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_payload, m_path)
    np.save(r_path, residuals_rel)
