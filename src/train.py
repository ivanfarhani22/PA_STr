import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("data/processed/features.csv", parse_dates=["race_date"])
y = df.target_finish_min
g = df.athlete_id
Path("models").mkdir(exist_ok=True)

USE_WEIGHT = False   # True = tiap pelari berbobot sama (kurangi bias pelari dominan)
WINDOW = 4           # jendela utama (minggu)

def feats(w):
    return [f"km_wk_{w}w", f"pace_{w}w", f"speed_{w}w_kmh", f"elev_{w}w",
            f"n_{w}w", f"longest_{w}w",
            "race_distance_km", "race_elev_m", "race_elev_per_km", "km_effort"]

if USE_WEIGHT:
    w_all = (1 / g.map(g.value_counts())).values
    w_all = w_all / w_all.mean()
else:
    w_all = None

def fit_params(idx):
    return {"sample_weight": w_all[idx]} if w_all is not None else {}

def oof_predict(model, grid, X):
    """Prediksi out-of-fold: GroupKFold(5) per pelari, tuning di dalam tiap lipatan."""
    pred = np.zeros(len(y))
    best = []
    for tr, te in GroupKFold(5).split(X, y, g):
        if grid:
            m = GridSearchCV(model, grid, cv=GroupKFold(3),
                             scoring="neg_root_mean_squared_error", n_jobs=-1)
            m.fit(X.iloc[tr], y.iloc[tr], groups=g.iloc[tr], **fit_params(tr))
            best.append(m.best_params_)
        else:
            m = clone(model)
            m.fit(X.iloc[tr], y.iloc[tr], **fit_params(tr))
        pred[te] = m.predict(X.iloc[te])
    return pred, best

def metrics(pred):
    return {"MAE": mean_absolute_error(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "R2": r2_score(y, pred)}

def per_distance(pred):
    out = {}
    for k, gg in df.groupby("race_type_std"):
        p, t = pred[gg.index], y[gg.index]
        out[k] = {"n": len(gg), "MAE": mean_absolute_error(t, p),
                  "MAPE_%": (abs(t - p) / t).mean() * 100}
    return pd.DataFrame(out).T.round(2)

# ---------- 1. Baseline (tanpa pelatihan) ----------
naive = (df[f"pace_{WINDOW}w"] * df.race_distance_km).values
d1 = df[f"longest_{WINDOW}w"]
riegel = (d1 * df[f"pace_{WINDOW}w"] * (df.race_distance_km / d1) ** 1.06).values

# ---------- 2. Perbandingan jendela (parameter default, tanpa tuning) ----------
print("== Jendela riwayat latihan (RMSE out-of-fold, parameter default) ==")
for w in (4, 8, 12):
    X = df[feats(w)]
    r = {}
    for name, mdl in {"LinReg": LinearRegression(),
                      "RF": RandomForestRegressor(300, random_state=42, n_jobs=-1),
                      "XGB": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          random_state=42, n_jobs=-1)}.items():
        p, _ = oof_predict(mdl, None, X)
        r[name] = round(metrics(p)["RMSE"], 2)
    print(f"{w:>2} minggu: {r}")

# ---------- 3. Model dengan tuning pada jendela utama ----------
X = df[feats(WINDOW)]
models = {
    "LinReg": (LinearRegression(), None),
    "RF": (RandomForestRegressor(random_state=42, n_jobs=-1),
           {"n_estimators": [200, 500], "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 3, 5]}),
    "XGB": (XGBRegressor(random_state=42, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [3, 5],
             "learning_rate": [0.03, 0.1], "subsample": [0.8, 1.0]}),
}

results = {"Baseline pace x jarak": metrics(naive), "Baseline Riegel": metrics(riegel)}
preds = {"Baseline pace x jarak": naive, "Baseline Riegel": riegel}
for name, (mdl, grid) in models.items():
    p, best = oof_predict(mdl, grid, X)
    results[name], preds[name] = metrics(p), p
    if best:
        print(f"\nParameter terpilih per lipatan ({name}):")
        for b in best: print("  ", b)

print(f"\n== Hasil keseluruhan (jendela {WINDOW} minggu, bobot={USE_WEIGHT}) ==")
print(pd.DataFrame(results).T.round(3))

for name in preds:
    print(f"\n-- Per jarak: {name}")
    print(per_distance(preds[name]))

# ---------- 4. Simpan prediksi OOF dan model final ----------
oof = df[["athlete_id", "race_date", "race_type_std", "target_finish_min"]].copy()
for name, p in preds.items():
    oof[name] = p
oof.to_csv("data/processed/oof_predictions.csv", index=False)

best_name = min(["LinReg", "RF", "XGB"], key=lambda n: results[n]["RMSE"])
mdl, grid = models[best_name]
if grid:
    final = GridSearchCV(mdl, grid, cv=GroupKFold(5),
                         scoring="neg_root_mean_squared_error", n_jobs=-1)
    final.fit(X, y, groups=g, **fit_params(np.arange(len(y))))
    final = final.best_estimator_
else:
    final = clone(mdl).fit(X, y, **fit_params(np.arange(len(y))))
joblib.dump({"model": final, "features": feats(WINDOW), "name": best_name},
            "models/best_model.pkl")
print(f"\nModel terbaik (RMSE terendah): {best_name} -> models/best_model.pkl")