import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("PROJECT DELTA — Step 5: Risk Map")
print("=" * 50)

# ── Approximate block centroids (lat/lon) ─────────────
# Based on published Sundarbans block geography
blocks = {
    "Gosaba":           (22.17, 88.80),
    "Patharpratima":    (21.83, 88.43),
    "Kultali":          (22.03, 88.57),
    "Sagar":            (21.73, 88.07),
    "Namkhana":         (21.77, 88.23),
    "Kakdwip":          (21.88, 88.18),
    "Mathurapur I":     (22.08, 88.28),
    "Mathurapur II":    (22.18, 88.33),
    "Joynagar I":       (22.17, 88.43),
    "Joynagar II":      (22.25, 88.52),
    "Basanti":          (22.27, 88.73),
    "Canning I":        (22.32, 88.67),
    "Canning II":       (22.38, 88.62),
    "Sandeshkhali I":   (22.48, 88.77),
    "Sandeshkhali II":  (22.55, 88.85),
    "Hingalganj":       (22.63, 88.83),
    "Minakhan":         (22.68, 88.73),
    "Haroa":            (22.72, 88.63),
    "Hasnabad":         (22.78, 88.78),
}

# ── Prediction results ────────────────────────────────
predictions = {
    "Gosaba":           2,
    "Patharpratima":    2,
    "Kultali":          2,
    "Sagar":            1,
    "Namkhana":         1,
    "Kakdwip":          1,
    "Mathurapur I":     1,
    "Mathurapur II":    1,
    "Joynagar I":       0,
    "Joynagar II":      0,
    "Basanti":          2,
    "Canning I":        1,
    "Canning II":       1,
    "Sandeshkhali I":   2,
    "Sandeshkhali II":  1,
    "Hingalganj":       1,
    "Minakhan":         0,
    "Haroa":            0,
    "Hasnabad":         0,
}

# ── Actual Amphan displacement ────────────────────────
actual = {
    "Gosaba":           2,
    "Patharpratima":    2,
    "Kultali":          2,
    "Sagar":            1,
    "Namkhana":         2,
    "Kakdwip":          1,
    "Mathurapur I":     1,
    "Mathurapur II":    1,
    "Joynagar I":       0,
    "Joynagar II":      0,
    "Basanti":          1,
    "Canning I":        1,
    "Canning II":       1,
    "Sandeshkhali I":   2,
    "Sandeshkhali II":  1,
    "Hingalganj":       1,
    "Minakhan":         0,
    "Haroa":            0,
    "Hasnabad":         0,
}

# ── Color mapping ─────────────────────────────────────
colors = {0: "#2B7A4B", 1: "#D4730A", 2: "#A32D2D"}
labels = {0: "Low risk", 1: "Medium risk", 2: "High risk"}

# ── Figure with two side-by-side maps ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.patch.set_facecolor("#F8F7F4")

for ax, data, title in zip(
    axes,
    [predictions, actual],
    ["Predicted (XGBoost Model)", "Actual (Amphan 2020)"]
):
    ax.set_facecolor("#D6EAF8")
    ax.set_xlim(87.9, 89.1)
    ax.set_ylim(21.6, 22.9)

    for block, (lat, lon) in blocks.items():
        risk = data[block]
        color = colors[risk]

        # Draw block as circle
        circle = plt.Circle((lon, lat), 0.06,
                             color=color, alpha=0.85, zorder=3)
        ax.add_patch(circle)

        # Block name label
        ax.annotate(block, (lon, lat),
                    fontsize=6.5, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points",
                    color="#2C2C2A", fontweight="500", zorder=4)

    # Legend
    patches = [mpatches.Patch(color=colors[i], label=labels[i])
               for i in [0, 1, 2]]
    ax.legend(handles=patches, loc="lower left",
              fontsize=9, framealpha=0.9)

    # Labels
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Bay of Bengal label
    ax.text(88.95, 21.75, "Bay of\nBengal",
            fontsize=9, color="#185FA5",
            ha="right", style="italic")

    # North arrow
    ax.annotate("N", xy=(87.95, 22.8), fontsize=12,
                fontweight="bold", ha="center")
    ax.annotate("↑", xy=(87.95, 22.72), fontsize=16,
                ha="center", color="#2C2C2A")

fig.suptitle(
    "PROJECT DELTA — Post-Cyclone Displacement Risk Map\n"
    "Indian Sundarbans | Cyclone Amphan 2020",
    fontsize=13, fontweight="bold", y=1.01
)

plt.tight_layout()

save_path = r"C:\Users\Hp\OneDrive\Desktop\PROJECT DELTA\outputs\risk_map.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()

print("\nRisk map saved → outputs/risk_map.png")
print("\nMap summary:")
pred_counts = pd.Series(predictions.values()).value_counts().sort_index()
for cls, count in pred_counts.items():
    print(f"  {labels[cls]}: {count} blocks")

print("\n" + "=" * 50)
print("Step 5 Complete — Risk Map Generated")
print("=" * 50)
