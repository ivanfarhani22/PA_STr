"""Modul kalkulator kesiapan pelari terhadap COT."""

import numpy as np
from runready.config import READINESS_CONFIG, ReadinessConfig


class ReadinessCalculator:
    """Kalkulator kesiapan pelari berdasarkan estimasi waktu finis dan residual historis."""

    def __init__(self, residuals: np.ndarray, mode: str = "rel", config: ReadinessConfig = READINESS_CONFIG):
        self.residuals = np.asarray(residuals)
        self.mode = mode
        self.config = config

    def calculate_percentage(self, y_hat: float, cot: float) -> float:
        """Menghitung persentase kesiapan (proporsi residual historis di mana estimasi <= COT)."""
        est = y_hat * (1 + self.residuals) if self.mode == "rel" else y_hat + self.residuals
        return float(100.0 * np.mean(est <= cot))

    def get_category(self, pct: float) -> str:
        """Menentukan kategori kesiapan berdasarkan persentase."""
        for threshold, name in self.config.categories:
            if pct >= threshold:
                return name
        return "Belum Siap"
