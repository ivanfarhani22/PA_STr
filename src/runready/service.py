"""Facade Service untuk prediksi dan evaluasi kesiapan runner."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from runready.config import PATHS, FEATURE_CONFIG, READINESS_CONFIG
from runready.readiness import ReadinessCalculator


@dataclass(frozen=True)
class PredictionResult:
    """Dataclass frozen untuk hasil prediksi dan kesiapan."""
    estimated_finish_time_min: float
    cot_min: float
    readiness_pct: float
    readiness_category: str
    out_of_training_range: bool
    warning_message: Optional[str] = None


class PredictionService:
    """Facade Service untuk satu-satunya pintu masuk prediksi dan analisis kesiapan (siap untuk web API / FastAPI)."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        residuals_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_path) if model_path else PATHS.final_model
        self.residuals_path = Path(residuals_path) if residuals_path else PATHS.residuals_rel

        # Muat model dan residual satu kali saat inisialisasi
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.features = payload["features"]
        self.target = payload.get("target", "pace_min_km")

        self.residuals = np.load(self.residuals_path)
        self.calculator = ReadinessCalculator(self.residuals, mode="rel")

    def predict_readiness(
        self, features_dict: Dict[str, float], cot_min: float
    ) -> PredictionResult:
        """Melakukan prediksi waktu finis, persentase kesiapan terhadap COT, kategori, dan validasi out-of-range."""
        # Konversi dict ke DataFrame
        df_in = pd.DataFrame([features_dict])

        # Pastikan fitur lengkap
        for f in self.features:
            if f not in df_in.columns:
                raise ValueError(f"Fitur wajib '{f}' tidak ditemukan dalam input.")

        X = df_in[self.features]
        distance_km = float(df_in["race_distance_km"].values[0])

        # Prediksi pace lalu kalikan jarak
        pred_pace = float(self.model.predict(X)[0])
        est_finish_min = pred_pace * distance_km

        # Cek peringatan out-of-range (misal km_effort > 50 atau race_elev_per_km > 30)
        km_effort = float(df_in.get("km_effort", pd.Series([distance_km])).values[0])
        elev_per_km = float(df_in.get("race_elev_per_km", pd.Series([0.0])).values[0])

        out_of_range = False
        warning_msg = None

        if (
            km_effort > READINESS_CONFIG.out_of_range_km_effort
            or elev_per_km > READINESS_CONFIG.out_of_range_elev_per_km
        ):
            out_of_range = True
            warning_msg = (
                f"Peringatan: Karakteristik lomba (km_effort={km_effort:.1f}, elev/km={elev_per_km:.1f} m/km) "
                "berada di luar rentang pelatihan utama (lomba datar/road)."
            )

        # Hitung persentase dan kategori kesiapan
        pct = self.calculator.calculate_percentage(est_finish_min, cot_min)
        cat = self.calculator.get_category(pct)

        return PredictionResult(
            estimated_finish_time_min=round(est_finish_min, 2),
            cot_min=round(cot_min, 2),
            readiness_pct=round(pct, 1),
            readiness_category=cat,
            out_of_training_range=out_of_range,
            warning_message=warning_msg,
        )
