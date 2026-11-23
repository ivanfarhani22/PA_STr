"""Pipeline skrip untuk menjalankan pembersihan, pembuatan label, fitur, dan split."""

import logging
from runready.config import PATHS
from runready.data.loaders import (
    load_kaggle_raw,
    save_dataframe,
)
from runready.data.cleaning import clean_kaggle_activities
from runready.labels import ProxyLabelStrategy
from runready.features import build_feature_table
from runready.splitting import split_features_by_athlete

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("runready.pipelines")


def run_clean_pipeline() -> None:
    logger.info("Menjalankan pipeline pembersihan data Kaggle...")
    df_raw = load_kaggle_raw()
    n0 = len(df_raw)
    df_clean, log = clean_kaggle_activities(df_raw)
    save_dataframe(df_clean, PATHS.clean_activities)

    logger.info(f"Awal: {n0} | Akhir: {len(df_clean)} | Pelari: {df_clean.athlete_id.nunique()}")
    for reason, n in log:
        logger.info(f"  dibuang {n:>6}  <- {reason}")


def run_labels_pipeline() -> None:
    logger.info("Menjalankan pipeline pembuatan label proxy...")
    df_clean = load_kaggle_raw(PATHS.clean_activities) # or clean loader
    # Gunakan clean loader yang sesuai
    from runready.data.loaders import load_clean_activities
    df_clean = load_clean_activities()

    strategy = ProxyLabelStrategy()
    
    logger.info("== Sensitivitas ambang speedup ==")
    sens = strategy.evaluate_sensitivity(df_clean)
    for sp, r in sens.items():
        c = r.race_type_std.value_counts().reindex(["5K", "10K", "HM", "M"]).fillna(0).astype(int)
        logger.info(f"speedup {int(sp*100):>2}% | sampel {len(r):>5} | pelari {r.athlete_id.nunique():>3} | {c.to_dict()}")

    final_labels = strategy.build_labels(df_clean)
    save_dataframe(final_labels, PATHS.race_labels)
    logger.info(f"Final label tersimpan: {len(final_labels)} sampel, {final_labels.athlete_id.nunique()} pelari.")


def run_features_pipeline() -> None:
    logger.info("Menjalankan pipeline ekstraksi fitur...")
    from runready.data.loaders import load_clean_activities, load_race_labels
    activities = load_clean_activities()
    races = load_race_labels()

    features_df = build_feature_table(activities, races)
    save_dataframe(features_df, PATHS.features)
    logger.info(f"Fitur tersimpan: {len(features_df)} sampel, {features_df.athlete_id.nunique()} pelari.")


def run_split_pipeline() -> None:
    logger.info("Menjalankan pipeline pembagian data train/test...")
    from runready.data.loaders import load_features
    features_df = load_features()
    split_df = split_features_by_athlete(features_df)
    save_dataframe(split_df, PATHS.features_split)
    
    tr = split_df[split_df.split == "train"]
    te = split_df[split_df.split == "test"]
    logger.info(f"Train: {len(tr)} sampel, {tr.athlete_id.nunique()} pelari | Test: {len(te)} sampel, {te.athlete_id.nunique()} pelari")


if __name__ == "__main__":
    run_clean_pipeline()
    run_labels_pipeline()
    run_features_pipeline()
    run_split_pipeline()
