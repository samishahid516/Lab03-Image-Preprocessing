import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

IMAGE_PATH = os.path.join("dataset", "panda_original.jpg")
OUTPUT_DIR = "outputs"
RESIZE_TO = (128, 128)

os.makedirs(OUTPUT_DIR, exist_ok=True)

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


log("=" * 70)
log("STEP 1: LOAD & EXPLORE THE IMAGE")
log("=" * 70)

image = None
try:
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"'{IMAGE_PATH}' does not exist (missing image).")
    image = Image.open(IMAGE_PATH)
    image.verify()  # check corrupted
    image = Image.open(IMAGE_PATH).convert("RGB")  # reopen
    log(f"Loaded image : {IMAGE_PATH}")
    log(f"Original size (W,H) : {image.size}")
    log(f"Mode : {image.mode}")
except Exception as e:
    log(f"ERROR: could not load '{IMAGE_PATH}' -> {e}")
    log("Handling technique: since the image is missing/corrupted and it is our")
    log("only observation, we stop here rather than processing a broken image.")
    raise SystemExit(1)

original_arr = np.array(image)
log(f"Original pixel array shape : {original_arr.shape} dtype={original_arr.dtype}")
log(f"Original pixel value range : min={original_arr.min()} max={original_arr.max()} "
    f"mean={original_arr.mean():.2f}")

log("")
log("=" * 70)
log("STEP 2: HANDLE MISSING / CORRUPTED IMAGES")
log("=" * 70)
log(
    "The try/except block above already covers this: if the image file did not "
    "exist, or Image.verify() found it corrupted, the script would report the "
    "problem and stop instead of crashing on broken pixel data. Since the image "
    "loaded successfully here, no rows need to be dropped."
)

log("")
log("=" * 70)
log(f"STEP 3: RESIZE IMAGE TO {RESIZE_TO}")
log("=" * 70)

resized_image = image.resize(RESIZE_TO)
resized_arr = np.array(resized_image)
resized_image.save(os.path.join(OUTPUT_DIR, "panda_resized_128x128.jpg"))
log(f"Resized pixel array shape : {resized_arr.shape}")
log(f"Saved resized image -> outputs/panda_resized_128x128.jpg")

log("")
log("=" * 70)
log("STEP 4: PREPROCESS PIXEL VALUES (NORMALIZE EACH PIXEL TO [0, 1])")
log("=" * 70)

normalized_arr = resized_arr.astype(np.float32) / 255.0
log(f"Pixel range before normalization : {resized_arr.min()} - {resized_arr.max()}")
log(f"Pixel range after  normalization : {normalized_arr.min():.3f} - {normalized_arr.max():.3f}")

log("")
log("=" * 70)
log("STEP 5: CONVERT IMAGE PIXELS INTO A TABULAR (ROW/COLUMN) DATASET")
log("=" * 70)

h, w, _ = resized_arr.shape
rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

R = resized_arr[:, :, 0].flatten()
G = resized_arr[:, :, 1].flatten()
B = resized_arr[:, :, 2].flatten()
R_norm = normalized_arr[:, :, 0].flatten()
G_norm = normalized_arr[:, :, 1].flatten()
B_norm = normalized_arr[:, :, 2].flatten()

gray = (0.299 * R + 0.587 * G + 0.114 * B)  # grayscale


def brightness_label(v):
    if v < 85:
        return "dark"
    elif v < 170:
        return "mid"
    else:
        return "light"


brightness = np.array([brightness_label(v) for v in gray])

pixel_df = pd.DataFrame({
    "row": rows.flatten(),
    "col": cols.flatten(),
    "R": R,
    "G": G,
    "B": B,
    "R_norm": R_norm,
    "G_norm": G_norm,
    "B_norm": B_norm,
    "brightness": brightness,
})

log(f"Tabular dataset shape (rows=pixels, columns=features) : {pixel_df.shape}")
log("First 5 rows of the tabular pixel dataset:")
log(pixel_df.head().to_string(index=False))

tabular_csv_path = os.path.join(OUTPUT_DIR, "pixel_table.csv")
pixel_df.to_csv(tabular_csv_path, index=False)
log(f"Saved full tabular pixel dataset -> {tabular_csv_path}")

log("")
log("=" * 70)
log("STEP 6: MEAN IMPUTATION ON SIMULATED MISSING PIXEL VALUES")
log("=" * 70)
log(
    "Our single image has no missing pixel values on its own, so to "
    "demonstrate the handout's Mean Imputation technique (SimpleImputer, "
    "strategy='mean'), we randomly mark ~2% of R/G/B pixel values as "
    "missing (NaN) - simulating sensor noise / corrupted pixel readings - "
    "then fill them back in using the mean of each column, exactly like "
    "imputa = SimpleImputer(missing_values=np.nan, strategy='mean') did "
    "for the Age and Salary columns in the handout."
)

rng = np.random.default_rng(42)
pixel_features = pixel_df[["R", "G", "B"]].values.astype(np.float64)
missing_mask = rng.random(pixel_features.shape) < 0.02  # 2% missing
X_missing = pixel_features.copy()
X_missing[missing_mask] = np.nan
log(f"Simulated missing values : {int(missing_mask.sum())} out of "
    f"{missing_mask.size} pixel-channel entries ({missing_mask.mean() * 100:.2f}%)")

imputa = SimpleImputer(missing_values=np.nan, strategy="mean")
imputa.fit(X_missing)
X_imputed = imputa.transform(X_missing)
log(f"Column means learned by SimpleImputer (used to fill NaNs) : "
    f"R={imputa.statistics_[0]:.2f} G={imputa.statistics_[1]:.2f} B={imputa.statistics_[2]:.2f}")

pixel_df["R_missing"] = X_missing[:, 0]
pixel_df["G_missing"] = X_missing[:, 1]
pixel_df["B_missing"] = X_missing[:, 2]
pixel_df["R_mean_imputed"] = X_imputed[:, 0]
pixel_df["G_mean_imputed"] = X_imputed[:, 1]
pixel_df["B_mean_imputed"] = X_imputed[:, 2]

sample_missing_rows = pixel_df[missing_mask.any(axis=1)].head()
log("Sample rows that had a missing value, before vs after mean imputation:")
log(sample_missing_rows[
    ["row", "col", "R_missing", "G_missing", "B_missing",
     "R_mean_imputed", "G_mean_imputed", "B_mean_imputed"]
].to_string(index=False))

imputed_csv_path = os.path.join(OUTPUT_DIR, "pixel_table_mean_imputed.csv")
pixel_df.to_csv(imputed_csv_path, index=False)
log(f"Saved tabular dataset with missing-value + mean-imputed columns -> {imputed_csv_path}")

log("")
log("=" * 70)
log("STEP 7: ONE-HOT ENCODE THE CATEGORICAL 'brightness' COLUMN")
log("=" * 70)

encoder = OneHotEncoder()
brightness_ohe = encoder.fit_transform(pixel_df[["brightness"]]).toarray()
ohe_columns = [f"brightness_{c}" for c in encoder.categories_[0]]
ohe_df = pd.DataFrame(brightness_ohe, columns=ohe_columns)

pixel_df_encoded = pd.concat([pixel_df, ohe_df], axis=1)
log(f"Categories found : {encoder.categories_[0].tolist()}")
log("First 5 rows after one-hot encoding:")
log(pixel_df_encoded.head().to_string(index=False))

encoded_csv_path = os.path.join(OUTPUT_DIR, "pixel_table_encoded.csv")
pixel_df_encoded.to_csv(encoded_csv_path, index=False)
log(f"Saved one-hot encoded tabular dataset -> {encoded_csv_path}")

log("")
log("=" * 70)
log("STEP 8: VISUALIZE ORIGINAL (128x128) VS PREPROCESSED (128x128) IMAGE")
log("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
axes[0].imshow(resized_arr)
axes[0].set_title(f"Resized image (raw pixels)\n{resized_arr.shape}", fontsize=10)
axes[0].axis("off")

axes[1].imshow(normalized_arr)
axes[1].set_title(f"Preprocessed\n{RESIZE_TO}, normalized [0,1]", fontsize=10)
axes[1].axis("off")

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, "original_vs_preprocessed.png")
plt.savefig(viz_path, dpi=130)
plt.close(fig)
log(f"Saved comparison figure -> {viz_path}")

log("")
log("Task 1 complete. See outputs/ for the resized image, the visualization,")
log("and the tabular pixel CSVs (pixel_table.csv, pixel_table_mean_imputed.csv,")
log("pixel_table_encoded.csv).")

with open(os.path.join(OUTPUT_DIR, "console_log_task1.txt"), "w") as f:
    f.write("\n".join(log_lines))
