import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

tr = pd.read_csv("data/processed/features.csv")
FULL = ["km_wk_4w", "pace_4w", "speed_4w_kmh", "elev_4w", "n_4w", "longest_4w",
        "race_distance_km", "race_elev_m", "race_elev_per_km", "km_effort"]
RED = ["km_wk_4w", "speed_4w_kmh", "race_distance_km"]   # setara dataset 8
y, D, g = tr.target_finish_min.values, tr.race_distance_km.values, tr.athlete_id.values

folds = np.zeros(len(tr), int)
for k, (_, te) in enumerate(GroupKFold(5).split(tr, y, g)):
    folds[te] = k

def oof(feats):
    p = np.zeros(len(tr))
    for k in range(5):
        a = folds != k
        m = LinearRegression().fit(tr.loc[a, feats], (y / D)[a])
        p[~a] = m.predict(tr.loc[~a, feats]) * D[~a]
    return p

def probs(yhat, cot, pools, gid, mode):
    p = np.zeros(len(yhat))
    for k, pool in pools.items():
        s, i = np.sort(pool), gid == k
        thr = cot[i] / yhat[i] - 1 if mode == "rel" else cot[i] - yhat[i]
        p[i] = np.searchsorted(s, thr, side="right") / len(s)
    return p

KS = [0.90, 0.95, 1.00, 1.05, 1.10, 1.15]   # COT hipotetis = k x estimasi

def evaluate(title, yhat, ytrue, pools, gid, mode):
    P, O = [], []
    for k in KS:
        cot = yhat * k
        P.append(probs(yhat, cot, pools, gid, mode)); O.append((ytrue <= cot).astype(float))
    P, O = np.concatenate(P), np.concatenate(O)
    brier = np.mean((P - O) ** 2)
    ref = O.mean() * (1 - O.mean())
    print(f"\n== {title} ==")
    print(f"Brier {brier:.4f} | referensi (prevalensi) {ref:.4f} | skill {1 - brier / ref:.2f}")
    b = pd.cut(P, [0, .1, .25, .5, .75, .9, 1.0], include_lowest=True)
    t = pd.DataFrame({"pred": P, "obs": O}).groupby(b, observed=True).agg(
        n=("obs", "size"), prediksi=("pred", "mean"), teramati=("obs", "mean")).round(2)
    print(t)

# ---------- A. Validasi internal dataset 1 (residual dari lipatan lain) ----------
yh = oof(FULL)
rel, ab = (y - yh) / yh, y - yh
print("Galat relatif (std):", rel.std().round(3), "| per jarak:",
      pd.Series(rel).groupby(tr.race_type_std.values).std().round(3).to_dict())
for mode, r in (("rel", rel), ("abs", ab)):
    pools = {k: r[folds != k] for k in range(5)}
    evaluate(f"A. Internal dataset 1, residual {mode}", yh, y, pools, folds, mode)

# ---------- B. Eksternal: dataset 8 (label asli, maraton) ----------
ex = pd.read_csv("data/raw/standard_marathon.csv").dropna(
    subset=["km4week", "sp4week_kmh", "marathon_time_min"]).reset_index(drop=True)
ex = ex.rename(columns={"km4week": "km_wk_4w", "sp4week_kmh": "speed_4w_kmh"})
ex["race_distance_km"] = 42.195
m_red = LinearRegression().fit(tr[RED], y / D)
yh8 = m_red.predict(ex[RED]) * 42.195
y8 = ex.marathon_time_min.values
yr = oof(RED); rr = (y - yr) / yr
gid0 = np.zeros(len(ex), int)
evaluate("B1. Eksternal dataset 8, residual semua jarak", yh8, y8, {0: rr}, gid0, "rel")
evaluate("B2. Eksternal dataset 8, residual maraton saja",
         yh8, y8, {0: rr[tr.race_type_std.values == "M"]}, gid0, "rel")

# ---------- C. Contoh penggunaan ----------
from readiness import readiness_pct, category
print("\n== Contoh ==")
for k in (0.95, 1.0, 1.05, 1.10):
    pct = readiness_pct(yh8[0], yh8[0] * k, rr)
    print(f"estimasi {yh8[0]:.0f} mnt, COT {yh8[0]*k:.0f} mnt -> {pct:.0f}% ({category(pct)})")

# ---------- D. Simpan artefak untuk sistem web ----------
Path("models").mkdir(exist_ok=True)
final = LinearRegression().fit(tr[FULL], y / D)
joblib.dump({"model": final, "features": FULL, "target": "pace_min_km"}, "models/final_model.pkl")
np.save("models/residuals_rel.npy", rel)
print("\nTersimpan: models/final_model.pkl, models/residuals_rel.npy")