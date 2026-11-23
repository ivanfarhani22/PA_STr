import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/activities_clean.csv", parse_dates=["timestamp"])

# nama: (jarak_min, jarak_maks) dalam km
STD = {"5K": (4.8, 5.6), "10K": (9.6, 11.0), "HM": (20.5, 22.5), "M": (41.0, 43.5)}

med = df.groupby("athlete_id").pace_min_km.median().rename("median_pace")
df = df.join(med, on="athlete_id")

# timestamp per pelari, untuk menghitung jumlah latihan 28 hari sebelumnya
ts = {a: g.timestamp.values for a, g in df.groupby("athlete_id")}

def n_prior(row, days=28):
    t = ts[row.athlete_id]
    end = row.timestamp.to_datetime64()
    start = end - pd.Timedelta(days=days).to_timedelta64()
    return int(((t >= start) & (t < end)).sum())

def build(speedup, min_prior=4):
    parts = []
    for name, (lo, hi) in STD.items():
        s = df[df.distance_km.between(lo, hi)]
        s = s[s.pace_min_km <= s.median_pace * (1 - speedup)].copy()
        s["race_type_std"] = name
        parts.append(s)
    r = pd.concat(parts)
    r["n_prior_4w"] = r.apply(n_prior, axis=1)
    return r[r.n_prior_4w >= min_prior]

# --- sensitivitas ambang ---
print("== Sensitivitas (min 4 latihan dalam 28 hari sebelumnya) ==")
for sp in (0.05, 0.08, 0.10):
    r = build(sp)
    c = r.race_type_std.value_counts().reindex(STD.keys()).fillna(0).astype(int)
    print(f"speedup {int(sp*100):>2}% | sampel {len(r):>5} | pelari {r.athlete_id.nunique():>3} | {c.to_dict()}")

# --- keluaran final ---
SPEEDUP = 0.08
final = build(SPEEDUP).rename(columns={"timestamp": "race_date", "duration_min": "finish_time_min"})
cols = ["athlete_id", "race_date", "race_type_std", "distance_km", "finish_time_min",
        "pace_min_km", "elevation_gain_m", "elevation_gain_per_km", "n_prior_4w"]
final = final[cols].sort_values(["athlete_id", "race_date"])
final.to_csv("data/processed/race_labels.csv", index=False)

print(f"\nFinal (speedup {int(SPEEDUP*100)}%): {len(final)} sampel, {final.athlete_id.nunique()} pelari")
print("\nSampel per pelari:")
print(final.groupby("athlete_id").size().describe().round(1))
print("\nWaktu finis (menit) per jarak:")
print(final.groupby("race_type_std").finish_time_min.describe().round(1))