"""
Lab # 03 - Lab Task 2
Title: Feature scaling on image pixel data (StandardScaler and MinMaxScaler),
with before/after pixel-value distribution visualization.

Uses the tabular pixel dataset produced by task1_image_preprocessing.py
(outputs/pixel_table.csv), where each row is one pixel and R, G, B are the
raw 0-255 pixel-value features to be scaled - the same way Age and Salary
were scaled as numeric features in the lab handout's CSV example.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

OUTPUT_DIR = "outputs"
TABLE_PATH = os.path.join(OUTPUT_DIR, "pixel_table.csv")

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


# ---------------------------------------------------------------------------
# Step 1: Load the tabular pixel dataset created in Task 1
# ---------------------------------------------------------------------------
log("=" * 70)
log("STEP 1: LOAD TABULAR PIXEL DATASET")
log("=" * 70)

if not os.path.exists(TABLE_PATH):
    raise SystemExit(
        f"'{TABLE_PATH}' not found. Run task1_image_preprocessing.py first."
    )

df = pd.read_csv(TABLE_PATH)
log(f"Loaded '{TABLE_PATH}' with shape {df.shape}")
log(f"Columns: {df.columns.tolist()}")

# Feature columns we scale: the raw pixel-intensity columns (0-255).
feature_cols = ["R", "G", "B"]
X = df[feature_cols].values.astype(np.float64)
log(f"Feature matrix X (raw pixel values) shape: {X.shape}")
log(f"Before scaling -> min={X.min():.2f} max={X.max():.2f} "
    f"mean={X.mean():.2f} std={X.std():.2f}")

# ---------------------------------------------------------------------------
# Step 2: Apply StandardScaler  ->  X' = (X - mean) / std
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("STEP 2: STANDARDIZATION (StandardScaler)")
log("=" * 70)

standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)

log(f"Learned per-column mean : {standard_scaler.mean_}")
log(f"Learned per-column std  : {standard_scaler.scale_}")
log(f"After StandardScaler -> min={X_standardized.min():.3f} "
    f"max={X_standardized.max():.3f} mean={X_standardized.mean():.3f} "
    f"std={X_standardized.std():.3f}")

# ---------------------------------------------------------------------------
# Step 3: Apply MinMaxScaler  ->  X' = (X - Xmin) / (Xmax - Xmin)
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("STEP 3: NORMALIZATION (MinMaxScaler)")
log("=" * 70)

minmax_scaler = MinMaxScaler()  # default feature_range=(0, 1)
X_minmax = minmax_scaler.fit_transform(X)

log(f"Learned per-column min : {minmax_scaler.data_min_}")
log(f"Learned per-column max : {minmax_scaler.data_max_}")
log(f"After MinMaxScaler -> min={X_minmax.min():.3f} max={X_minmax.max():.3f} "
    f"mean={X_minmax.mean():.3f} std={X_minmax.std():.3f}")

# ---------------------------------------------------------------------------
# Step 4: Save the scaled tabular datasets
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("STEP 4: SAVE SCALED TABULAR DATASETS")
log("=" * 70)

df_standardized = df.copy()
df_standardized[["R_standard", "G_standard", "B_standard"]] = X_standardized
std_csv = os.path.join(OUTPUT_DIR, "pixel_table_standardscaled.csv")
df_standardized.to_csv(std_csv, index=False)
log(f"Saved -> {std_csv}")

df_minmax = df.copy()
df_minmax[["R_minmax", "G_minmax", "B_minmax"]] = X_minmax
minmax_csv = os.path.join(OUTPUT_DIR, "pixel_table_minmaxscaled.csv")
df_minmax.to_csv(minmax_csv, index=False)
log(f"Saved -> {minmax_csv}")

# ---------------------------------------------------------------------------
# Step 5: Visualize pixel-value distributions before vs after scaling
# ---------------------------------------------------------------------------
log("")
log("=" * 70)
log("STEP 5: VISUALIZE PIXEL-VALUE DISTRIBUTIONS BEFORE / AFTER SCALING")
log("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].hist(X[:, 0], bins=50, color="red", alpha=0.5, label="R")
axes[0].hist(X[:, 1], bins=50, color="green", alpha=0.5, label="G")
axes[0].hist(X[:, 2], bins=50, color="blue", alpha=0.5, label="B")
axes[0].set_title("Original pixel values\n(raw, 0-255)")
axes[0].set_xlabel("pixel value")
axes[0].set_ylabel("pixel count")
axes[0].legend()

axes[1].hist(X_standardized[:, 0], bins=50, color="red", alpha=0.5, label="R")
axes[1].hist(X_standardized[:, 1], bins=50, color="green", alpha=0.5, label="G")
axes[1].hist(X_standardized[:, 2], bins=50, color="blue", alpha=0.5, label="B")
axes[1].set_title("After StandardScaler\n(mean=0, std=1)")
axes[1].set_xlabel("standardized value")
axes[1].legend()

axes[2].hist(X_minmax[:, 0], bins=50, color="red", alpha=0.5, label="R")
axes[2].hist(X_minmax[:, 1], bins=50, color="green", alpha=0.5, label="G")
axes[2].hist(X_minmax[:, 2], bins=50, color="blue", alpha=0.5, label="B")
axes[2].set_title("After MinMaxScaler\n(range 0-1)")
axes[2].set_xlabel("normalized value")
axes[2].legend()

plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, "scaling_distributions.png")
plt.savefig(dist_path, dpi=130)
plt.close(fig)
log(f"Saved distribution comparison figure -> {dist_path}")

# ---------------------------------------------------------------------------
# Step 6: Discussion - when to use MinMaxScaler vs StandardScaler
# ---------------------------------------------------------------------------
discussion = """
STEP 6: WHEN TO USE MinMaxScaler vs StandardScaler ON IMAGE DATA
======================================================================
MinMaxScaler (Normalization, X' = (X - Xmin) / (Xmax - Xmin)):
  - Rescales values into a fixed, known range (default [0, 1]).
  - Ideal for image pixel data because pixel intensities already have a
    known, fixed range (0-255 for 8-bit images), so the min/max are known
    and consistent across images.
  - Commonly used before feeding images into neural networks / CNNs, since
    most activation functions and image-display conventions expect inputs
    in [0, 1] (or [-1, 1] with a shifted variant).
  - Sensitive to outliers: a single unusually bright/dark pixel can stretch
    the scale and compress the rest of the values.

StandardScaler (Standardization, X' = (X - mean) / std):
  - Centers values around 0 with unit standard deviation; does NOT bound
    values to a fixed range.
  - Useful when the model assumes/benefits from normally-distributed,
    zero-centered features (e.g., PCA, linear/logistic regression, SVM,
    k-means), or when different features have very different natural scales
    that need to be put on comparable footing.
  - Less sensitive to a fixed min/max, more robust when the data's range
    isn't naturally bounded or when outliers should have a smaller relative
    effect on the extremes of the scaled data (though it is still affected
    by outliers through the mean/std).

For this image-preprocessing task specifically: MinMaxScaler is generally
preferred for raw pixel values because the physical range (0-255) is fixed
and known in advance, so mapping it to [0, 1] is simple, interpretable, and
matches what most image-based ML/DL pipelines expect. StandardScaler becomes
more relevant later in the pipeline, e.g., when pixel data has already been
turned into extracted numeric features (like edge counts, color histograms,
or CNN embedding vectors) with unknown/unbounded ranges and varying scales,
where zero-centering helps optimization algorithms converge faster.
"""
log(discussion)

log("")
log("Task 2 complete. See outputs/ for the scaled CSVs and the distribution")
log("comparison figure (scaling_distributions.png).")

with open(os.path.join(OUTPUT_DIR, "console_log_task2.txt"), "w") as f:
    f.write("\n".join(log_lines))
