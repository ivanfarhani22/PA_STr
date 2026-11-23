import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

tr = pd.read_csv("data/processed/features.csv")
ex = pd.read_csv("data/processed/../raw/standard_marathon.csv")
print("Kolom dataset 8:", list(ex.columns))

KM_DIV = 1  # ubah ke 4 bila km4week ternyata TOTAL 4 minggu, bukan rata-rata per minggu
ex = ex.rename(columns={"km4week": "km_wk_4w", "sp4week_kmh": "speed_4w_kmh"})
ex["km_wk_4w"] = ex["km_wk_4w"] / KM_DIV
ex["race_distance_km"] = 42.195
ex = ex.dropna(subset=["km_wk_4w", "speed_4w_kmh", "marathon_time_min"]).reset_index(drop=True)

FEATS = ["km_wk_4w", "speed_4w_kmh", "race_distance_km"]
def prep(d): return d[FEATS], d.target_finish_min / d.race_distance_km
Xtr, ytr = prep(tr)
Xex, yex = ex[FEATS], ex.marathon_time_min

def report(name, t_pred):
    e = yex - t_pred
    print(f"{name:<34} MAE {mean_absolute_error(yex, t_pred):6.2f} | "
          f"RMSE {np.sqrt(mean_squared_error(yex, t_pred)):6.2f} | "
          f"MAPE {100*(abs(e)/yex).mean():5.2f}% | R2 {r2_score(yex, t_pred):5.2f} | bias {(-e).mean():6.2f}")

print(f"\nUji eksternal: {len(ex)} pelari (label asli)\n")
D = ex.race_distance_km
report("Baseline: jarak / kecepatan latihan", 60 * D / ex.speed_4w_kmh)
report("Baseline terkalibrasi (di dataset 8)", LinearRegression()
       .fit(ex[["speed_4w_kmh"]].assign(x=60 * D / ex.speed_4w_kmh)[["x"]], yex)
       .predict((60 * D / ex.speed_4w_kmh).to_frame("x")))  # batas atas: kalibrasi pada data uji sendiri

models = {"LinReg": LinearRegression(),
          "RF": RandomForestRegressor(500, min_samples_leaf=3, random_state=42, n_jobs=-1),
          "XGB": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                              subsample=0.8, random_state=42, n_jobs=-1)}
for n, m in models.items():
    m.fit(Xtr, ytr)
    report(f"{n} (latih semua jarak, dataset 1)", m.predict(Xex) * D)

mar = tr[tr.race_type_std == "M"]
Xm, ym = prep(mar)
for n, m in models.items():
    m.fit(Xm, ym)
    report(f"{n} (latih maraton saja, n={len(mar)})", m.predict(Xex) * D)