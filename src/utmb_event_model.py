import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

u = pd.read_csv("data/raw/standard_utmb_races.csv")
u = u[(u.quality_flag == "ok") & (u.distance_km >= 5) & (u.elevation_gain_m >= 0) &
      (u.mean_finish_time_h > 0) & (u.n_finishers >= 10)].copy()
u["epk"] = u.elevation_gain_m / u.distance_km
print(f"Event: {len(u)} | lomba unik (race_uid): {u.race_uid.nunique()}")

y, g, t = np.log(u.mean_finish_time_h), u.race_uid, u.mean_finish_time_h
sets = {"jarak saja": pd.DataFrame({"ld": np.log(u.distance_km)}),
        "jarak + elevasi/km": pd.DataFrame({"ld": np.log(u.distance_km), "epk": u.epk}),
        "jarak + elevasi/km + elevasi total": pd.DataFrame({"ld": np.log(u.distance_km), "epk": u.epk,
                                                              "elev": u.elevation_gain_m / 1000})}
print("\n== CV per race_uid (satu lomba tidak muncul di train dan test sekaligus) ==")
for name, X in sets.items():
    p = np.zeros(len(u))
    for a, b in GroupKFold(5).split(X, y, g):
        p[b] = LinearRegression().fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b])
    pt = np.exp(p)
    print(f"{name:<38} MAPE {100*(abs(pt-t)/t).mean():5.2f}% | MAE {abs(pt-t).mean():5.2f} jam")

m = LinearRegression().fit(sets["jarak + elevasi/km"], y)
print(f"\nKoefisien (log waktu): jarak {m.coef_[0]:.3f} | elevasi/km {m.coef_[1]:.4f} "
      f"-> +10 m/km kira-kira +{100*(np.exp(10*m.coef_[1])-1):.1f}% waktu")

u["ratio_last_mean"] = u.last_time_h / u.mean_finish_time_h
print("\nRasio waktu finisher terakhir / waktu rata-rata (acuan kasar batas waktu):")
print(u.groupby("race_category").ratio_last_mean.describe().round(2))
print("\nDNF rate per kategori:\n", u.groupby("race_category").dnf_rate.mean().round(3))