"""Pipeline untuk pelatihan model, evaluasi kesiapan, uji eksternal maraton, dan UTMB."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from runready.config import PATHS, FEATURE_CONFIG
from runready.data.loaders import load_features_split, load_marathon_raw, load_utmb_raw, save_model_artifacts
from runready.models.registry import get_model
from runready.models.trainer import oof_predict, fit_final_model
from runready.models.baselines import baseline_calibrated, riegel_approx
from runready.evaluation.metrics import evaluate_regression, evaluate_per_distance
from runready.evaluation.calibration import evaluate_calibration
from runready.readiness import ReadinessCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("runready.pipelines")


def run_train_pipeline() -> None:
    logger.info("Menjalankan pipeline pelatihan model dan evaluasi internal...")
    split_df = load_features_split()
    tr = split_df[split_df.split == "train"].reset_index(drop=True)
    te = split_df[split_df.split == "test"].reset_index(drop=True)

    FULL = list(FEATURE_CONFIG.full_features)

    # 1. CV on train & test internal
    models = {
        "LinReg": get_model("LinReg"),
        "RF": get_model("RF"),
        "XGB": get_model("XGB"),
    }

    rows = []
    test_preds = {}
    for name, model in models.items():
        cv_preds = oof_predict(model, tr, FULL)
        cv_metrics = evaluate_regression(tr["target_finish_min"], cv_preds)

        # Fit pada seluruh train, test pada test internal
        m_fit, _ = fit_final_model(tr, name, FULL)
        tp = m_fit.predict(te[FULL]) * te["race_distance_km"].values
        test_preds[name] = tp
        test_metrics = evaluate_regression(te["target_finish_min"], tp)

        rows.append({
            "model": name,
            **{f"CV_{k}": v for k, v in cv_metrics.items() if k != "R2"},
            **{f"TEST_{k}": v for k, v in test_metrics.items()},
        })

    # Baseline terkalibrasi
    bp = baseline_calibrated(te)
    rows.append({"model": "Baseline terkalibrasi", **{f"TEST_{k}": v for k, v in evaluate_regression(te["target_finish_min"], bp).items()}})

    df_res = pd.DataFrame(rows).round(2)
    logger.info("\n== CV (train) vs TEST internal ==\n" + df_res.to_string(index=False))

    # Simpan artefak final (menggunakan LinReg dan residual train)
    final_model, residuals_rel = fit_final_model(tr, "LinReg", FULL)
    save_model_artifacts(
        model_payload={"model": final_model, "features": FULL, "target": "pace_min_km"},
        residuals_rel=residuals_rel,
    )
    logger.info("Artefak model final dan residual berhasil disimpan ke folder models/.")


def run_external_pipeline() -> None:
    logger.info("Menjalankan pipeline validasi eksternal (Dataset Maraton)...")
    split_df = load_features_split()
    tr = split_df[split_df.split == "train"].reset_index(drop=True)
    RED = list(FEATURE_CONFIG.reduced_features)

    ex = load_marathon_raw().dropna(subset=["km4week", "sp4week_kmh", "marathon_time_min"]).reset_index(drop=True)
    ex = ex.rename(columns={"km4week": "km_wk_4w", "sp4week_kmh": "speed_4w_kmh"})
    ex["race_distance_km"] = 42.195

    m_red, _ = fit_final_model(tr, "LinReg", RED)
    p8 = m_red.predict(ex[RED]) * 42.195
    yex = ex["marathon_time_min"].values

    metrics_ex = evaluate_regression(yex, p8)
    bias = float((p8 - yex).mean())
    logger.info(f"Eksternal Dataset 8: MAE {metrics_ex['MAE']:.2f} | MAPE {metrics_ex['MAPE_%']:.2f}% | R2 {metrics_ex['R2']:.2f} | bias {bias:.2f}")


def run_readiness_pipeline() -> None:
    logger.info("Menjalankan pipeline evaluasi kesiapan & kalibrasi...")
    split_df = load_features_split()
    tr = split_df[split_df.split == "train"].reset_index(drop=True)
    te = split_df[split_df.split == "test"].reset_index(drop=True)
    FULL = list(FEATURE_CONFIG.full_features)

    # OOF residual train
    _, residuals_rel = fit_final_model(tr, "LinReg", FULL)
    
    # Prediksi test internal
    m_fit, _ = fit_final_model(tr, "LinReg", FULL)
    yh_te = m_fit.predict(te[FULL]) * te["race_distance_km"].values
    y_te = te["target_finish_min"].values

    brier, ref, skill, table = evaluate_calibration(yh_te, y_te, residuals_rel, mode="rel")
    logger.info(f"Kesiapan TEST internal -> Brier: {brier:.4f} | Referensi: {ref:.4f} | Skill: {skill:.2f}")
    logger.info("\nTabel Kalibrasi:\n" + str(table))


def run_utmb_pipeline() -> None:
    logger.info("Menjalankan pipeline analisis UTMB Trail...")
    u = load_utmb_raw()
    u = u[(u.quality_flag == "ok") & (u.distance_km >= 5) & (u.elevation_gain_m >= 0) &
          (u.mean_finish_time_h > 0) & (u.n_finishers >= 10)].copy()
    u["epk"] = u.elevation_gain_m / u.distance_km

    y, g, t = np.log(u["mean_finish_time_h"]), u["race_uid"], u["mean_finish_time_h"]
    X_set = pd.DataFrame({"ld": np.log(u["distance_km"]), "epk": u["epk"]})

    preds = np.zeros(len(u))
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LinearRegression
    for a, b in GroupKFold(5).split(X_set, y, g):
        preds[b] = LinearRegression().fit(X_set.iloc[a], y.iloc[a]).predict(X_set.iloc[b])
    pt = np.exp(preds)
    mape = 100.0 * np.mean(np.abs(pt - t) / t)
    logger.info(f"UTMB Model (jarak + elevasi/km) -> MAPE: {mape:.2f}% (Target ~13.62%)")


if __name__ == "__main__":
    run_train_pipeline()
    run_external_pipeline()
    run_readiness_pipeline()
    run_utmb_pipeline()
