import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

tr = pd.read_csv("data/processed/features.csv")
ex = pd.read_csv("data/raw/standard_marathon.csv").dropna(
    subset=["km4week", "sp4week_kmh", "marathon_time_min"]).reset_index(drop=True)

# 1. konsistensi kecepatan vs pace latihan
print("Korelasi sp4week_kmh vs 60/training_pace:",
      round(np.corrcoef(ex.sp4week_kmh, 60 / ex.training_pace_min_km)[0, 1], 3))
print("Rasio median:", round((ex.sp4week_kmh / (60 / ex.training_pace_min_km)).median(), 3))

# 2. skala jarak mingguan
print("\nkm_wk_4w dataset 1 :", tr.km_wk_4w.describe().round(1)[["25%", "50%", "75%", "max"]].to_dict())
print("km4week dataset 8  :", ex.km4week.describe().round(1)[["25%", "50%", "75%", "max"]].to_dict())

# 3. sensitivitas
D = 42.195
ytr = tr.target_finish_min / tr.race_distance_km
def run(label, feats_tr, feats_ex, d):
    m = LinearRegression().fit(tr[feats_tr], ytr)
    p = m.predict(d[feats_ex]) * D
    e = d.marathon_time_min - p
    print(f"{label:<42} MAE {mean_absolute_error(d.marathon_time_min, p):5.2f} | "
          f"MAPE {100*(abs(e)/d.marathon_time_min).mean():5.2f}% | "
          f"R2 {r2_score(d.marathon_time_min, p):5.2f} | bias {(-e).mean():6.2f}")

print()
for div in (1, 4):
    d = ex.rename(columns={"sp4week_kmh": "speed_4w_kmh"}).copy()
    d["km_wk_4w"] = d.km4week / div
    d["race_distance_km"] = D
    run(f"km + speed + jarak (km4week / {div})", ["km_wk_4w", "speed_4w_kmh", "race_distance_km"],
        ["km_wk_4w", "speed_4w_kmh", "race_distance_km"], d)
    run(f"speed + jarak saja (tanpa km) [div {div}]", ["speed_4w_kmh", "race_distance_km"],
        ["speed_4w_kmh", "race_distance_km"], d) if div == 1 else None

# 4. tanpa baris sp4week_rescaled
d = ex[ex.quality_flag != "sp4week_rescaled"].rename(columns={"sp4week_kmh": "speed_4w_kmh"}).copy()
d["km_wk_4w"] = d.km4week; d["race_distance_km"] = D
run("tanpa baris sp4week_rescaled", ["km_wk_4w", "speed_4w_kmh", "race_distance_km"],
    ["km_wk_4w", "speed_4w_kmh", "race_distance_km"], d)