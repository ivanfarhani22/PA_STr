import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

SEED = 42
df = pd.read_csv("data/processed/features.csv", parse_dates=["race_date"])
df.insert(0, "race_id", [f"R{i:04d}" for i in range(len(df))])
df["label_type"] = "proxy"

# test internal tetap: 20% PELARI (bukan baris), dikunci dengan seed
ath = np.sort(df.athlete_id.unique())
test_ath = set(np.random.default_rng(SEED).choice(ath, size=round(0.2 * len(ath)), replace=False))
df["split"] = np.where(df.athlete_id.isin(test_ath), "test", "train")
df.to_csv("data/processed/features_split.csv", index=False)

tr = df[df.split == "train"].reset_index(drop=True)
te = df[df.split == "test"].reset_index(drop=True)
print(f"Train: {len(tr)} sampel, {tr.athlete_id.nunique()} pelari | "
      f"Test: {len(te)} sampel, {te.athlete_id.nunique()} pelari")
print("Per jarak (train/test):", tr.race_type_std.value_counts().to_dict(), te.race_type_std.value_counts().to_dict())

FULL = ["km_wk_4w", "pace_4w", "speed_4w_kmh", "elev_4w", "n_4w", "longest_4w",
        "race_distance_km", "race_elev_m", "race_elev_per_km", "km_effort"]
RED = ["km_wk_4w", "speed_4w_kmh", "race_distance_km"]   # setara dataset 8

def metr(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)),
            "MAPE_%": 100 * (abs(y - p) / y).mean(), "R2": r2_score(y, p)}

def oof(model, d, feats):
    p = np.zeros(len(d))
    for a, b in GroupKFold(5).split(d, groups=d.athlete_id):
        m = clone(model).fit(d.loc[a, feats], (d.target_finish_min / d.race_distance_km)[a])
        p[b] = m.predict(d.loc[b, feats]) * d.race_distance_km[b]
    return p

MODELS = {"LinReg": LinearRegression(),
          "RF": RandomForestRegressor(500, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
          "XGB": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                              subsample=0.8, random_state=SEED, n_jobs=-1)}

# ---------- 1. CV pada TRAIN, lalu test internal ----------
rows, test_pred = [], {}
for n, m in MODELS.items():
    cvm = metr(tr.target_finish_min, oof(m, tr, FULL))
    fit = clone(m).fit(tr[FULL], tr.target_finish_min / tr.race_distance_km)
    tp = fit.predict(te[FULL]) * te.race_distance_km.values
    test_pred[n] = tp
    rows.append({"model": n, **{f"CV_{k}": v for k, v in cvm.items() if k != "R2"},
                 **{f"TEST_{k}": v for k, v in metr(te.target_finish_min, tp).items()}})
base = LinearRegression().fit((tr.pace_4w * tr.race_distance_km).to_frame("x"), tr.target_finish_min)
bp = base.predict((te.pace_4w * te.race_distance_km).to_frame("x"))
rows.append({"model": "Baseline terkalibrasi", **{f"TEST_{k}": v for k, v in metr(te.target_finish_min, bp).items()}})
print("\n== CV (train) vs TEST internal ==")
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n== TEST: MAPE per jarak (LinReg) ==")
e = pd.DataFrame({"j": te.race_type_std, "mape": 100 * abs(te.target_finish_min - test_pred["LinReg"]) / te.target_finish_min})
print(e.groupby("j").mape.agg(["count", "mean"]).round(2))

# ---------- 2. Kesiapan: residual dari TRAIN, diuji di TEST ----------
yh_tr = oof(MODELS["LinReg"], tr, FULL)
rel_tr = (tr.target_finish_min.values - yh_tr) / yh_tr
yh_te, y_te = test_pred["LinReg"], te.target_finish_min.values
KS = [0.90, 0.95, 1.00, 1.05, 1.10, 1.15]

def calib(title, yhat, ytrue, pool):
    s = np.sort(pool); P, O = [], []
    for k in KS:
        cot = yhat * k
        P.append(np.searchsorted(s, cot / yhat - 1, side="right") / len(s)); O.append((ytrue <= cot).astype(float))
    P, O = np.concatenate(P), np.concatenate(O)
    br, ref = np.mean((P - O) ** 2), O.mean() * (1 - O.mean())
    print(f"\n== {title} ==\nBrier {br:.4f} | referensi {ref:.4f} | skill {1 - br / ref:.2f}")
    b = pd.cut(P, [0, .1, .25, .5, .75, .9, 1.0], include_lowest=True)
    print(pd.DataFrame({"pred": P, "obs": O}).groupby(b, observed=True).agg(
        n=("obs", "size"), prediksi=("pred", "mean"), teramati=("obs", "mean")).round(2))

calib("Kesiapan di TEST internal (pelari belum pernah dilihat)", yh_te, y_te, rel_tr)

# ---------- 3. Eksternal dataset 8 (model dilatih HANYA dari train) ----------
ex = pd.read_csv("data/raw/standard_marathon.csv").dropna(
    subset=["km4week", "sp4week_kmh", "marathon_time_min"]).reset_index(drop=True)
ex = ex.rename(columns={"km4week": "km_wk_4w", "sp4week_kmh": "speed_4w_kmh"})
ex["race_distance_km"] = 42.195
mr = LinearRegression().fit(tr[RED], tr.target_finish_min / tr.race_distance_km)
p8 = mr.predict(ex[RED]) * 42.195
print("\n== Eksternal dataset 8 (latih: train saja) ==")
print({k: round(v, 2) for k, v in metr(ex.marathon_time_min, p8).items()}, "| bias", round((p8 - ex.marathon_time_min).mean(), 2))
yr = oof(LinearRegression(), tr, RED)
calib("Kesiapan eksternal dataset 8", p8, ex.marathon_time_min.values, (tr.target_finish_min.values - yr) / yr)

# ---------- 4. Simpan artefak (konsisten: model dan residual dari TRAIN) ----------
Path("models").mkdir(exist_ok=True)
joblib.dump({"model": MODELS["LinReg"].fit(tr[FULL], tr.target_finish_min / tr.race_distance_km),
             "features": FULL, "target": "pace_min_km"}, "models/final_model.pkl")
np.save("models/residuals_rel.npy", rel_tr)
print("\nTersimpan: models/final_model.pkl, models/residuals_rel.npy, data/processed/features_split.csv")