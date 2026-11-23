import pandas as pd

u = pd.read_csv("data/raw/standard_utmb_races.csv")
u = u[(u.quality_flag == "ok") & (u.distance_km > 0) & (u.elevation_gain_m >= 0)].copy()
u["epk"] = u.elevation_gain_m / u.distance_km
u["km_effort"] = u.distance_km + u.elevation_gain_m / 100
tr = pd.read_csv("data/processed/features.csv")

P = [.05, .10, .25, .50, .75, .90, .95, .99]
print(f"UTMB (bersih): {len(u)} event\n")
print("Elevasi per km, UTMB:\n", u.epk.describe(percentiles=P).round(1))
print("\nElevasi per km, lomba proxy dataset 1:\n", tr.race_elev_per_km.describe(percentiles=P).round(1))
print("\nMedian per kategori UTMB:")
print(u.groupby("race_category")[["distance_km", "elevation_gain_m", "epk", "km_effort"]].median().round(1))
print("\nkm_effort: UTMB vs dataset 1")
print(pd.DataFrame({"utmb": u.km_effort.describe(percentiles=P), "ds1": tr.km_effort.describe(percentiles=P)}).round(1))

for thr in (15, 20, 25, 30):
    print(f"ambang {thr} m/km | UTMB >= ambang: {100*(u.epk >= thr).mean():5.1f}% | "
          f"ds1 >= ambang: {100*(tr.race_elev_per_km >= thr).mean():5.1f}%")