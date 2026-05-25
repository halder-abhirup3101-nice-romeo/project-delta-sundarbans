import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("PROJECT DELTA — Step 3: Model Training")
print("=" * 50)

# ── Load feature matrix ───────────────────────────────
df = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\PROJECT DELTA\data\raw\features.csv")
print(f"\n[1] Loaded feature matrix: {df.shape}")

feature_cols = [
    "max_wind_speed_knots", "min_pressure_hpa", "storm_surge_height_m",
    "rainfall_anomaly", "flood_inundation_pct", "mean_elevation_m",
    "mangrove_cover_pct", "pop_density_norm", "poverty_index",
    "pct_kutcha_housing", "distance_to_coast_km", "historical_breach_rate",
    "literacy_rate", "cyclone_category"
]

# ── Temporal split ────────────────────────────────────
print("\n[2] Splitting data temporally...")
train = df[df["event"] == "aila_2009"]
test  = df[df["event"] == "amphan_2020"]

X_train = train[feature_cols]
y_train = train["displacement_class"]
X_test  = test[feature_cols]
y_test  = test["displacement_class"]

print(f"    Train (Aila 2009):   {X_train.shape[0]} blocks")
print(f"    Test  (Amphan 2020): {X_test.shape[0]} blocks")
print(f"    Train classes: {dict(y_train.value_counts().sort_index())}")
print(f"    Test classes:  {dict(y_test.value_counts().sort_index())}")

# ── Train XGBoost ─────────────────────────────────────
print("\n[3] Training XGBoost model...")
model = XGBClassifier(
    objective        = "multi:softprob",
    num_class        = 3,
    eval_metric      = "mlogloss",
    n_estimators     = 200,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    verbosity        = 0
)

model.fit(X_train, y_train)
print("    Model trained successfully!")

# ── Evaluate ──────────────────────────────────────────
print("\n[4] Evaluating on Amphan 2020 test set...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("\n    Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Low", "Medium", "High"]))

print("    Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"              Pred Low  Pred Med  Pred High")
for i, row in enumerate(cm):
    label = ["Actual Low ", "Actual Med ", "Actual High"][i]
    print(f"    {label}  {row[0]}         {row[1]}         {row[2]}")

# ── AUC-ROC ───────────────────────────────────────────
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
print(f"\n    AUC-ROC (macro): {auc:.3f}")

# ── Feature Importance ────────────────────────────────
print("\n[5] Feature Importance (top 10):")
importance = pd.Series(model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False)
for feat, score in importance.head(10).items():
    bar = "█" * int(score * 100)
    print(f"    {feat:<30} {bar} {score:.3f}")

# ── Save predictions ──────────────────────────────────
results = test[["block_name", "event", "displacement_class"]].copy()
results["predicted_class"] = y_pred
results["prob_low"]    = y_prob[:, 0].round(2)
results["prob_medium"] = y_prob[:, 1].round(2)
results["prob_high"]   = y_prob[:, 2].round(2)
results["correct"]     = (results["displacement_class"] == results["predicted_class"])

save_path = r"C:\Users\Hp\OneDrive\Desktop\PROJECT DELTA\data\raw\predictions.csv"
results.to_csv(save_path, index=False)

print("\n[6] Predictions saved!")
print(results[["block_name", "displacement_class", "predicted_class", "prob_high", "correct"]].to_string(index=False))

print("\n" + "=" * 50)
# ── Retrain with block-level features only ────────────
print("\n[7] Retraining with block-level features only...")

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

X_train_b = train[block_features]
X_test_b  = test[block_features]

model2 = XGBClassifier(
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

model2.fit(X_train_b, y_train)
y_pred2 = model2.predict(X_test_b)
y_prob2 = model2.predict_proba(X_test_b)

print("\n    Classification Report (block features only):")
print(classification_report(y_test, y_pred2,
      target_names=["Low", "Medium", "High"]))

auc2 = roc_auc_score(y_test_bin, y_prob2, multi_class="ovr", average="macro")
print(f"    AUC-ROC (macro): {auc2:.3f}")

print("\n    Predictions vs Actual:")
results2 = test[["block_name", "displacement_class"]].copy()
results2["predicted"] = y_pred2
results2["prob_high"] = y_prob2[:, 2].round(2)
results2["correct"]   = (results2["displacement_class"] == results2["predicted"])
print(results2.to_string(index=False))

print("\n    Feature Importance (block-level model):")
importance2 = pd.Series(model2.feature_importances_, index=block_features)
importance2 = importance2.sort_values(ascending=False)
for feat, score in importance2.items():
    bar = "█" * int(score * 100)
    print(f"    {feat:<30} {bar} {score:.3f}")
print("Step 3 Complete — Model Training Done")
print("=" * 50)
