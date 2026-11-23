import pandas as pd
import numpy as np

act = pd.read_csv("data/processed/activities_clean.csv", parse_dates=["timestamp"])
races = pd.read_csv("data/processed/race_labels.csv", parse_dates=["race_date"])

hist = {a: g.sort_values("timestamp") for a, g in act.groupby("athlete_id")}
WINDOWS = (4, 8, 12)  # minggu

def window_feats(h, race_date, w):
    start = race_date - pd.Timedelta(weeks=w)
    x = h[(h.timestamp >= start) & (h.timestamp < race_date)]  # hanya SEBELUM lomba
    if len(x) == 0:
        return {}
    km, mnt = x.distance_km.sum(), x.duration_min.sum()
    return {
        f"km_wk_{w}w": km / w,                 # rata-rata jarak per minggu
        f"pace_{w}w": mnt / km,                # rata-rata pace = sum t / sum d
        f"speed_{w}w_kmh": 60 * km / mnt,      # setara sp4week di dataset 8
        f"elev_{w}w": x.elevation_gain_m.sum(),
        f"n_{w}w": len(x),
        f"longest_{w}w": x.distance_km.max(),
        f"hr_{w}w": x.heart_rate_avg.mean(),   # bisa NaN
    }

rows = []
for r in races.itertuples():
    f = {}
    for w in WINDOWS:
        f.update(window_feats(hist[r.athlete_id], r.race_date, w))
    rows.append(f)

feat = pd.concat([races.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

# fitur karakteristik lomba
feat["race_distance_km"] = feat.distance_km
feat["race_elev_m"] = feat.elevation_gain_m
feat["race_elev_per_km"] = feat.elevation_gain_per_km
feat["km_effort"] = feat.distance_km + feat.elevation_gain_m / 100
feat = feat.rename(columns={"finish_time_min": "target_finish_min"})

feat.to_csv("data/processed/features.csv", index=False)

print(f"Sampel: {len(feat)} | Pelari: {feat.athlete_id.nunique()}")
print("\nNilai kosong (%):")
print((feat.isna().mean() * 100).round(1)[lambda s: s > 0])
print("\nRingkasan fitur utama:")
print(feat[["km_wk_4w", "pace_4w", "speed_4w_kmh", "elev_4w", "n_4w", "km_effort"]].describe().round(2).T)
print("\nKorelasi dengan waktu finis (per jarak):")
for name, g in feat.groupby("race_type_std"):
    c = g[["km_wk_4w", "pace_4w", "speed_4w_kmh", "elev_4w"]].corrwith(g.target_finish_min).round(2)
    print(name, len(g), c.to_dict())