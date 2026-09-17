"""
Lab # 03 - "Show Sir" pipeline viewer.

This script does NOT introduce any new preprocessing technique. It simply
re-runs the same steps as task1_image_preprocessing.py / task2_feature_
scaling.py and draws every stage side-by-side in ONE picture, so it is easy
to demo the whole pipeline live to the teacher: run this file, then open
outputs/preprocessing_pipeline_view.png and talk through it panel by panel.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

IMAGE_PATH = os.path.join("dataset", "panda_original.jpg")
OUTPUT_DIR = "outputs"
RESIZE_TO = (128, 128)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Re-run the same pipeline (kept short - full explanations are in
# task1_image_preprocessing.py and task2_feature_scaling.py)
# ---------------------------------------------------------------------------
image = Image.open(IMAGE_PATH)
image.verify()
image = Image.open(IMAGE_PATH).convert("RGB")

resized_image = image.resize(RESIZE_TO)
resized_arr = np.array(resized_image)

normalized_arr = resized_arr.astype(np.float32) / 255.0

h, w, _ = resized_arr.shape
R = resized_arr[:, :, 0].flatten()
G = resized_arr[:, :, 1].flatten()
B = resized_arr[:, :, 2].flatten()
gray = (0.299 * R + 0.587 * G + 0.114 * B)

brightness = np.where(gray < 85, 0, np.where(gray < 170, 1, 2))  # 0=dark,1=mid,2=light
brightness_map = brightness.reshape(h, w)

# missing-value simulation + mean imputation (same as Task 1, Step 6)
rng = np.random.default_rng(42)
pixel_features = np.stack([R, G, B], axis=1).astype(np.float64)
missing_mask = rng.random(pixel_features.shape) < 0.02
X_missing = pixel_features.copy()
X_missing[missing_mask] = np.nan
imputa = SimpleImputer(missing_values=np.nan, strategy="mean")
X_imputed = imputa.fit_transform(X_missing)

missing_any_pixel = missing_mask.any(axis=1).reshape(h, w)  # True where a pixel had a missing channel
imputed_image = X_imputed.reshape(h, w, 3).astype(np.uint8)

# feature scaling (same as Task 2)
X = pixel_features
X_std = StandardScaler().fit_transform(X)
X_mm = MinMaxScaler().fit_transform(X)

# ---------------------------------------------------------------------------
# Build ONE figure with every stage, in the same order you explain it
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
fig.suptitle("Image Preprocessing Pipeline - Step by Step (128x128 throughout)", fontsize=16, fontweight="bold")

# 1. Resized (this is our only "starting" image - always 128x128, no 720x720 shown)
axes[0, 0].imshow(resized_arr)
axes[0, 0].set_title(f"1. Image (128x128)\n{resized_arr.shape}")
axes[0, 0].axis("off")

# 2. Normalized [0,1]
axes[0, 1].imshow(normalized_arr)
axes[0, 1].set_title("2. Normalized\npixel values 0-255 -> 0-1")
axes[0, 1].axis("off")

# 3. Missing pixels highlighted in red
highlight = resized_arr.copy()
highlight[missing_any_pixel] = [255, 0, 0]
axes[0, 2].imshow(highlight)
axes[0, 2].set_title(f"3. Simulated missing pixels\n(red = {missing_any_pixel.sum()} pixels marked NaN)")
axes[0, 2].axis("off")

# 4. After mean imputation
axes[0, 3].imshow(imputed_image)
axes[0, 3].set_title("4. After Mean Imputation\n(missing pixels filled with column mean)")
axes[0, 3].axis("off")

# 5. Brightness category map (the column we one-hot encode)
cmap = matplotlib.colors.ListedColormap(["black", "gray", "white"])
axes[1, 0].imshow(brightness_map, cmap=cmap, vmin=0, vmax=2)
axes[1, 0].set_title("5. Brightness category per pixel\n(dark / mid / light -> one-hot encoded)")
axes[1, 0].axis("off")

# 6. Pixel-value histogram: original vs MinMax (overlaid, R channel only, simplified)
axes[1, 1].hist(X[:, 0], bins=40, color="gray", alpha=0.6, label="original (0-255)")
axes[1, 1].hist(X_mm[:, 0] * 255, bins=40, color="green", alpha=0.5, label="MinMax (rescaled x255 to compare shape)")
axes[1, 1].set_title("6. MinMaxScaler\nsame shape, values now in [0,1]")
axes[1, 1].legend(fontsize=7)
axes[1, 1].set_xlabel("pixel value")

axes[1, 2].hist(X_std[:, 0], bins=40, color="purple", alpha=0.6)
axes[1, 2].axvline(0, color="black", linestyle="--", linewidth=1)
axes[1, 2].set_title("7. StandardScaler\ncentered at 0 (mean=0, std=1)")
axes[1, 2].set_xlabel("standardized value")

# 8. Zoomed pixel grid - shows exactly how each pixel becomes one row in the
#    tabular dataset (pixel_table.csv): one grid cell = one row = (R, G, B).
GRID_R0, GRID_C0, GRID_N = 60, 96, 4  # a 4x4 patch with good color variety
patch = resized_arr[GRID_R0:GRID_R0 + GRID_N, GRID_C0:GRID_C0 + GRID_N]

ax = axes[1, 3]
ax.imshow(patch, extent=(0, GRID_N, GRID_N, 0), interpolation="nearest")
ax.set_xticks(range(GRID_N + 1))
ax.set_yticks(range(GRID_N + 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, color="cyan", linewidth=1.5)
ax.set_title(f"8. Zoomed pixel grid ({GRID_N}x{GRID_N})\neach cell = 1 row in pixel_table.csv")

for gr in range(GRID_N):
    for gc in range(GRID_N):
        rr, gg, bb = patch[gr, gc]
        text_color = "white" if (int(rr) + int(gg) + int(bb)) < 380 else "black"
        ax.text(gc + 0.5, gr + 0.5, f"{rr},{gg},{bb}",
                ha="center", va="center", fontsize=6.5, color=text_color)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(OUTPUT_DIR, "preprocessing_pipeline_view.png")
plt.savefig(out_path, dpi=140)
plt.close(fig)

print(f"Saved -> {out_path}")
print("Open this single image and walk through the 8 panels, left to right, top to bottom.")
