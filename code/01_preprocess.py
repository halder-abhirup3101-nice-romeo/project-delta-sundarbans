import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime, timezone

print("=" * 50)
print("PROJECT DELTA — Step 1: Data Loading")
print("=" * 50)

# ── Base path ─────────────────────────────────────────
base = r"C:\Users\Hp\OneDrive\Desktop\PROJECT DELTA\data\raw"

# ── 1. IBTrACS Cyclone Data ───────────────────────────
print("\n[1] Loading IBTrACS cyclone data...")
df_ibtracs = pd.read_csv(base + r"\ibtracs.NI.list.v04r00.csv", skiprows=[1])
cyclones = df_ibtracs[df_ibtracs["NAME"].isin(["AILA", "AMPHAN"])]
cyclones_clean = cyclones[[
    "NAME", "SEASON", "ISO_TIME",
    "LAT", "LON",
    "WMO_WIND", "WMO_PRES",
    "NEWDELHI_WIND", "NEWDELHI_PRES",
    "DIST2LAND", "LANDFALL"
]].copy()
print(f"    Cyclone records loaded: {cyclones_clean.shape[0]} rows")

# ── 2. ERA5 Rainfall Data ─────────────────────────────
print("\n[2] Loading ERA5 rainfall data...")
ds = nc.Dataset(base + r"\era5_rainfall.nc")
times = ds.variables['valid_time'][:]
dates = [datetime.fromtimestamp(int(t), tz=timezone.utc).strftime('%Y-%m') for t in times]
tp = ds.variables['tp'][:] * 1000
era5_summary = pd.DataFrame({
    "month": dates,
    "avg_rainfall_mm": [float(np.ma.mean(tp[0])), float(np.ma.mean(tp[1]))]
})
print(f"    ERA5 time steps: {dates}")
print(f"    Aila rainfall:   {era5_summary.iloc[0]['avg_rainfall_mm']:.2f} mm")
print(f"    Amphan rainfall: {era5_summary.iloc[1]['avg_rainfall_mm']:.2f} mm")

# ── 3. Census 2011 Data ───────────────────────────────
print("\n[3] Loading Census 2011 block data...")
df_census = pd.read_csv(base + r"\census_sundarbans_2011.csv")
print(f"    Blocks loaded: {df_census.shape[0]}")
print(f"    Columns: {df_census.columns.tolist()}")

# ── 4. Displacement + Feature Data ───────────────────
print("\n[4] Loading displacement and feature data...")
df_disp = pd.read_csv(base + r"\displacement_sundarbans.csv")
print(f"    Block-event rows: {df_disp.shape[0]}")
print(f"    Events: {df_disp['event'].unique()}")

# ── 5. Merge Census into Displacement Data ────────────
print("\n[5] Merging datasets...")
df = df_disp.merge(df_census, on="block_name", how="left")
print(f"    Merged shape: {df.shape}")
print(f"    Columns: {df.columns.tolist()}")

# ── 6. Check for missing values ───────────────────────
print("\n[6] Checking for missing values...")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("    No missing values found!")
else:
    print(missing)

# ── 7. Save merged dataset ────────────────────────────
save_path = r"C:\Users\Hp\OneDrive\Desktop\PROJECT DELTA\data\raw\merged_dataset.csv"
df.to_csv(save_path, index=False)
print(f"\n[7] Merged dataset saved!")
print(f"    Path: {save_path}")
print(f"    Shape: {df.shape}")

print("\n" + "=" * 50)
print("Step 1 Complete — Ready for Feature Engineering")
print("=" * 50)
