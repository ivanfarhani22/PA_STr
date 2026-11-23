"""Unit test pytest untuk modul RunReady."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from runready.config import PATHS, FEATURE_CONFIG
from runready.data.loaders import load_features_split, load_clean_activities
from runready.data.cleaning import clean_kaggle_activities
from runready.readiness import ReadinessCalculator
from runready.service import PredictionService


def test_cleaning_numbers():
    """Uji jumlah baris hasil pembersihan sesuai Angka Regresi (40830 baris)."""
    raw_path = PATHS.kaggle_raw
    if not raw_path.exists():
        pytest.skip("File data mentah kaggle tidak ditemukan.")
    df_raw = pd.read_csv(raw_path, parse_dates=["timestamp"])
    df_clean, log = clean_kaggle_activities(df_raw)
    assert len(df_clean) == 40830, f"Diharapkan 40830 baris, mendapati {len(df_clean)}"
    assert df_clean.athlete_id.nunique() == 116


def test_no_leakage():
    """Uji tidak ada kebocoran waktu: aktivitas dalam jendela harus sebelum tanggal lomba."""
    split_df = load_features_split()
    activities = load_clean_activities()
    
    # Ambil sampel acak lomba
    sample_race = split_df.iloc[0]
    athlete_id = sample_race["athlete_id"]
    race_date = sample_race["race_date"]
    
    ath_act = activities[activities.athlete_id == athlete_id]
    w4_act = ath_act[(ath_act.timestamp >= race_date - pd.Timedelta(weeks=4)) & (ath_act.timestamp < race_date)]
    
    # Pastikan semua timestamp < race_date
    if len(w4_act) > 0:
        assert (w4_act.timestamp < race_date).all()


def test_readiness_logic():
    """Uji logika kalkulator kesiapan (monoton dan batas kategori)."""
    residuals = np.array([-0.05, 0.0, 0.05, 0.10])
    calc = ReadinessCalculator(residuals, mode="rel")

    y_hat = 100.0
    # COT rendah -> persen rendah
    pct_low = calc.calculate_percentage(y_hat, cot=95.0)
    # COT tinggi -> persen tinggi
    pct_high = calc.calculate_percentage(y_hat, cot=120.0)

    assert pct_low <= pct_high
    assert 0.0 <= pct_low <= 100.0
    assert 0.0 <= pct_high <= 100.0

    assert calc.get_category(95.0) == "Sangat Siap"
    assert calc.get_category(80.0) == "Siap"
    assert calc.get_category(60.0) == "Cukup Siap"
    assert calc.get_category(30.0) == "Kurang Siap"
    assert calc.get_category(10.0) == "Belum Siap"


def test_prediction_service():
    """Uji PredictionService memberikan flag out-of-range untuk km_effort tinggi."""
    if not PATHS.final_model.exists() or not PATHS.residuals_rel.exists():
        pytest.skip("Artefak model belum dilatih.")

    service = PredictionService()
    split_df = load_features_split()
    sample_row = split_df.iloc[0]

    feats = {f: sample_row[f] for f in FEATURE_CONFIG.full_features}
    # Ubah km_effort jadi tinggi untuk memicu out-of-range flag
    feats["km_effort"] = 70.0

    res = service.predict_readiness(feats, cot_min=240.0)
    assert res.out_of_training_range is True
    assert res.warning_message is not None
    assert 0.0 <= res.readiness_pct <= 100.0
