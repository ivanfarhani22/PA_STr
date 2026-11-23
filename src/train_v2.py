import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("data/processed/features.csv")
g, y, D = df.athlete_id, df.target_finish_min, df.race_distance_km
FEATS = ["km_wk_4w", "pace_4w", "speed_4w_kmh", "elev_4w", "n_4w", "longest_4w",
         "race_distance_km", "race_elev_m", "race_elev_per_km", "km_effort"]
X = df[FEATS]
w_bal = (1 / g.map(g.value_counts())).values
w_bal = w_bal / w_bal.mean()

TARGET = {"finish": (y, lambda p: p),
          "pace": (y / D, lambda p: p * D),
          "logfinish": (np.log(y), np.exp)}

MODELS = {
    "LinReg": LinearRegression(),
    "RF": RandomForestRegressor(500, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "XGB": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                        subsample=0.8, random_state=42, n_jobs=-1),
}

def oof(model, Xm, t, w=None):
    pred = np.zeros(len(t))
    for tr, te in GroupKFold(5).split(Xm, t, g):
        m = clone(model)
        m.fit(Xm.iloc[tr], t.iloc[tr], sample_weight=None if w is None else w[tr])
        pred[te] = m.predict(Xm.iloc[te])
    return pred

def evaluate(pred):
    r = {"RMSE": np.sqrt(mean_squared_error(y, pred)),
         "MAE": mean_absolute_error(y, pred), "R2": r2_score(y, pred)}
    for k, gg in df.groupby("race_type_std"):
        p, t = pred[gg.index], y[gg.index]
        r[f"MAPE_{k}"] = (abs(t - p) / t).mean() * 100
        r[f"R2in_{k}"] = r2_score(t, p)
    return r

rows = []
# baseline terkalibrasi: waktu = k x (pace_4w x jarak)
naive = (df.pace_4w * D).to_frame("naive")
rows.append({"model": "Baseline terkalibrasi", "target": "finish", "bobot": False,
             **evaluate(oof(LinearRegression(), naive, y))})

for tname, (t, inv) in TARGET.items():
    for weighted in (False, True):
        for mname, mdl in MODELS.items():
            p = inv(oof(mdl, X, t, w_bal if weighted else None))
            rows.append({"model": mname, "target": tname, "bobot": weighted, **evaluate(p)})

out = pd.DataFrame(rows).sort_values("RMSE").round(2)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.to_string(index=False))
out.to_csv("data/processed/model_comparison.csv", index=False)