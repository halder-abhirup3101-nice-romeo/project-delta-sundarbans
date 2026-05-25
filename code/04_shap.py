import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("PROJECT DELTA — Step 4: SHAP Analysis")
print("=" * 50)

# ── Load data and retrain block-level model ───────────
df = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\PROJECT DELTA\data\raw\features.csv")

block_features = [
    "flood_inundation_pct",
    "mean_elevation_m",
    "mangrove_cover_pct",
    "pop_density_norm",
    "poverty_index",
    "pct_kutcha_housing",
    "distance_to_coast_km",
    "historical_breach_rate",
    "literacy_rate"
]

train = df[df["event"] == "aila_2009"]
test  = df[df["event"] == "amphan_2020"]

X_train = train[block_features]
y_train = train["displacement_class"]
X_test  = test[block_features]

model = XGBClassifier(
    objective        = "multi:softprob",
    num_class        = 3,
    eval_metric      = "mlogloss",
    n_estimators     = 100,
    max_depth        = 3,
    learning_rate    = 0.1,
    subsample        = 0.9,
    colsample_bytree = 0.9,
    random_state     = 42,
    verbosity        = 0
)
model.fit(X_train, y_train)
print("\n[1] Model retrained successfully")

# ── SHAP Explainer ────────────────────────────────────
print("\n[2] Computing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Handle both 3D array and list-of-arrays formats
if isinstance(shap_values, list):
    shap_high = shap_values[2]
else:
    shap_high = shap_values[:, :, 2]

print("    SHAP values computed!")
print(f"    shap_high shape: {shap_high.shape}")

# ── Plot 1: Global Feature Importance (High class) ────
print("\n[3] Generating Plot 1 — Global Feature Importance...")
mean_abs_shap = pd.Series(
    np.abs(shap_high).mean(axis=0),
    index=block_features
).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
colors = ["#185FA5" if v > mean_abs_shap.mean() else "#B5D4F4"
          for v in mean_abs_shap.values]
bars = plt.barh(mean_abs_shap.index, mean_abs_shap.values, color=colors)
plt.xlabel("Mean |SHAP value| — impact on High displacement prediction")
plt.title("PROJECT DELTA — Feature Importance for High Displacement Risk")
plt.tight_layout()
plt.savefig(r"C:\Users\Hp\OneDrive\Desktop\PROJECT DELTA\outputs\shap_importance.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("    Saved: shap_importance.png")

# ── Plot 2: Block-by-block SHAP for High class ────────
print("\n[4] Generating Plot 2 — Block SHAP Heatmap...")
shap_df = pd.DataFrame(shap_high, columns=block_features)
shap_df.index = test["block_name"].values

plt.figure(figsize=(10, 6))
import matplotlib.cm as cm
im = plt.imshow(shap_df.values, aspect="auto", cmap="RdBu_r")
plt.colorbar(im, label="SHAP value (positive = pushes toward High risk)")
plt.xticks(range(len(block_features)), block_features, rotation=45, ha="right")
plt.yticks(range(len(test)), test["block_name"].values)
plt.title("PROJECT DELTA — SHAP Values per Block (High Displacement Class)")
plt.tight_layout()
plt.savefig(r"C:\Users\Hp\OneDrive\Desktop\PROJECT DELTA\outputs\shap_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("    Saved: shap_heatmap.png")

# ── Print SHAP values for top 3 high-risk blocks ──────
print("\n[5] SHAP breakdown for top 3 high-risk blocks:")
top_blocks = ["Gosaba", "Patharpratima", "Kultali"]
for block in top_blocks:
    idx = list(test["block_name"].values).index(block)
    print(f"\n    {block}:")
    block_shap = pd.Series(shap_high[idx], index=block_features)
    block_shap = block_shap.sort_values(ascending=False)
    for feat, val in block_shap.items():
        direction = "↑ High" if val > 0 else "↓ Low"
        print(f"      {feat:<30} {val:+.3f}  {direction}")

print("\n" + "=" * 50)
print("Step 4 Complete — SHAP Analysis Done")
print("=" * 50)
