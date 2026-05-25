import pandas as pd
import numpy as np

print("=" * 50)
print("PROJECT DELTA — Step 2: Feature Engineering")
print("=" * 50)

# ── Load merged dataset ───────────────────────────────
df = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\PROJECT DELTA\data\raw\merged_dataset.csv")
print(f"\n[1] Loaded merged dataset: {df.shape}")

# ── 1. Rainfall Anomaly Index ─────────────────────────
print("\n[2] Calculating Rainfall Anomaly Index...")
df["rainfall_anomaly"] = (
    (df["rainfall_event_mm"] - df["rainfall_30yr_mean_mm"])
    / df["rainfall_30yr_std_mm"]
)
print(f"    Aila avg anomaly:   {df[df['event']=='aila_2009']['rainfall_anomaly'].mean():.2f}")
print(f"    Amphan avg anomaly: {df[df['event']=='amphan_2020']['rainfall_anomaly'].mean():.2f}")

# ── 2. Poverty Index ──────────────────────────────────
print("\n[3] Calculating Poverty Index...")
df["poverty_index_raw"] = (
    0.40 * df["pct_kutcha_housing"] +
    0.35 * df["historical_breach_rate"] +
    0.25 * df["sc_population_pct"]
)
df["poverty_index"] = (
    (df["poverty_index_raw"] - df["poverty_index_raw"].min()) /
    (df["poverty_index_raw"].max() - df["poverty_index_raw"].min())
)
print(f"    Poverty index range: {df['poverty_index'].min():.2f} to {df['poverty_index'].max():.2f}")

# ── 3. Population Density Normalised ─────────────────
print("\n[4] Normalising population density...")
df["pop_density_norm"] = (
    (df["population_density"] - df["population_density"].min()) /
    (df["population_density"].max() - df["population_density"].min())
)

# ── 4. Final Feature Matrix ───────────────────────────
print("\n[5] Building final feature matrix...")
feature_cols = [
    "max_wind_speed_knots",
    "min_pressure_hpa",
    "storm_surge_height_m",
    "rainfall_anomaly",
    "flood_inundation_pct",
    "mean_elevation_m",
    "mangrove_cover_pct",
    "pop_density_norm",
    "poverty_index",
    "pct_kutcha_housing",
    "distance_to_coast_km",
    "historical_breach_rate",
    "literacy_rate",
    "cyclone_category"
]

df_features = df[["block_name", "event", "displacement_class"] + feature_cols].copy()

print(f"    Features selected: {len(feature_cols)}")
print(f"    Final shape: {df_features.shape}")

# ── 5. Class Distribution ─────────────────────────────
print("\n[6] Displacement class distribution:")
class_counts = df_features["displacement_class"].value_counts().sort_index()
labels = {0: "Low", 1: "Medium", 2: "High"}
for cls, count in class_counts.items():
    print(f"    Class {cls} ({labels[cls]}): {count} blocks")

# ── 6. Save feature matrix ────────────────────────────
save_path = r"C:\Users\Hp\OneDrive\Desktop\PROJECT DELTA\data\raw\features.csv"
df_features.to_csv(save_path, index=False)
print(f"\n[7] Feature matrix saved!")
print(f"    Path: {save_path}")

print("\n" + "=" * 50)
print("Step 2 Complete — Ready for Model Training")
print("=" * 50)
