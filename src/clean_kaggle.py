import pandas as pd
from pathlib import Path

RAW = Path("data/raw/standard_kaggle.csv")
OUT = Path("data/processed/activities_clean.csv")

df = pd.read_csv(RAW, parse_dates=["timestamp"])
n0 = len(df)
log = []

def drop(mask, reason):
    global df
    log.append((reason, int(mask.sum())))
    df = df[~mask]

drop(df.timestamp.dt.year < 2010, "timestamp sebelum 2010")
drop(df.duplicated(["athlete_id", "timestamp"]), "duplikat athlete+timestamp")
drop((df.distance_km <= 0) | (df.duration_min <= 0), "jarak/durasi <= 0")
drop(df.duration_min > 24 * 60, "durasi > 24 jam")

# hitung ulang pace dan elevasi per km agar konsisten (Pers. 3.1 dan 3.2)
df = df.assign(
    pace_min_km=df.duration_min / df.distance_km,
    elevation_gain_per_km=df.elevation_gain_m / df.distance_km,
)
drop(~df.pace_min_km.between(2.5, 15), "pace di luar 2.5-15 menit/km")
drop(df.elevation_gain_per_km > 300, "elevasi/km > 300 m/km")

df = df.drop(columns=["cadence_avg"])  # kosong 100%
df = df.sort_values(["athlete_id", "timestamp"]).reset_index(drop=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"Awal: {n0} | Akhir: {len(df)} | Pelari: {df.athlete_id.nunique()}")
for reason, n in log:
    print(f"  dibuang {n:>6}  <- {reason}")