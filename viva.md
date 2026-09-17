# Viva Guide (Simple) — Lab 03 Task 1 & Task 2

Use this as your speaking script for the viva. It covers:
1. What to say (one line) for every `import`
2. The exact error you get if you remove that import
3. How to explain the whole project to your teacher, step by step
4. A side-by-side demo of MinMaxScaler vs StandardScaler
5. **A one-picture live demo you can show sir directly** (`view_preprocessing_pipeline.py`)

---

## SIMPLE EXPLANATION — read this first (easy words, step by step)

This section explains everything in the simplest way possible, with no
confusing terms. Read this before anything else in the file.

### What is PIL, and why do we import it?

```python
from PIL import Image
```

- **PIL** = "Python Imaging Library" (the installed package is called
  `Pillow`, but you still write `from PIL import Image` — that's just its
  name in code, don't worry about the difference for viva).
- It is a ready-made toolbox for working with image files. We don't write
  our own code to open a `.jpg` file byte-by-byte — PIL already knows how.
- From PIL, we only use ONE thing in this project: the `Image` class. It
  gives us 4 tools:
  1. `Image.open(path)` — opens the image file.
  2. `.verify()` — checks the file is not broken/corrupted.
  3. `.convert("RGB")` — makes sure the image has exactly 3 color channels
     (Red, Green, Blue), even if the original was black-and-white or had a
     4th transparency channel.
  4. `.resize((128, 128))` — changes the image's width and height to
     exactly 128 by 128 pixels.
- **If you remove `from PIL import Image`:** the very first line that uses
  `Image.open(...)` will crash with:
  ```
  NameError: name 'Image' is not defined
  ```
  because Python has no idea what the word "Image" means without that
  import — it is not a built-in Python word, it only exists because PIL
  defines it.

### How do you know resize actually worked? (proof, not just trust)

You don't just "believe" the resize happened — the code **proves** it by
printing the shape before and after:

```python
print(resized_arr.shape)   # (128, 128, 3)
```

- Before resizing: the array shape is `(720, 720, 3)` — meaning 720 rows,
  720 columns, 3 color channels.
- After `image.resize((128, 128))`: the new array shape is `(128, 128, 3)`
  — meaning now only 128 rows and 128 columns.
- **This printed shape IS the proof.** If resize did not work, the shape
  would still say 720×720. Since it says 128×128, resize definitely worked.
- You can also **see it** — open `outputs/panda_resized_128x128.jpg` and
  compare its file size / zoom level to the original; it's visibly a
  smaller image.

### The whole pipeline, in the simplest words possible

Think of it like a factory line — the image enters on one side, and comes
out the other side ready for a machine learning model:

1. **Open the image** — like opening a photo in a photo viewer app.
2. **Check it's not broken** — like checking a file isn't corrupted before
   using it (this uses `.verify()`).
3. **Resize to 128×128** — like shrinking a big photo to a small, fixed
   size, so every photo in a dataset is the same size.
4. **Normalize (divide by 255)** — every pixel's color number (0 to 255)
   gets turned into a small decimal between 0 and 1. The picture looks
   exactly the same to your eyes — only the numbers behind it changed.
5. **Turn the image into a table** — every single pixel becomes one row in
   a spreadsheet-like table, with columns for its position and its Red,
   Green, Blue color numbers. A 128×128 image = 16,384 rows (one per
   pixel).
6. **Fix missing pixel values (Mean Imputation)** — pretend some pixel
   values got lost/corrupted; fill each missing one back in using the
   average (mean) value of that column, so no gaps are left in the table.
7. **One-hot encode a category column** — label each pixel as dark, mid, or
   light, then turn that single text column into 3 separate 0/1 columns
   (one column per category) — because machine learning models can't
   understand text like "dark" directly, only numbers.
8. **Visualize before vs after** — draw the images side by side so you can
   literally see that resize + normalize happened.
9. **(Task 2) Scale the numbers two different ways** — `MinMaxScaler`
   squeezes all pixel numbers into 0–1. `StandardScaler` re-centers them
   around 0. Both are drawn as histograms so you can see the difference.

That's the entire project. Every section below this one just repeats these
same 9 steps in more technical detail, with code and viva questions.

---

## 0. Live demo for sir — one picture, every step

There is a dedicated script, `view_preprocessing_pipeline.py`, whose only
job is to **show** the pipeline, not explain the code. Run it and open one
image — that's the whole demo.

**How to run it:**
```
python view_preprocessing_pipeline.py
```
This prints:
```
Saved -> outputs\preprocessing_pipeline_view.png
```
Then open `outputs/preprocessing_pipeline_view.png`. It's a single figure
with 8 panels (2 rows × 4 columns) laid out in the exact order you explain
the pipeline. **Everything is shown at 128×128 — there is no 720×720 panel
anywhere**, to keep the demo focused on the actual preprocessed size used
for the tabular data. Point at each panel while you talk:

| Panel | What sir sees | What you say |
|---|---|---|
| **1. Image (128×128)** | The panda, already at the working size | "This is my image at 128×128 — the size I resized it to and the size every later step works with." |
| **2. Normalized** | Looks the same to the eye, but values are now 0–1 instead of 0–255 | "Same picture, but now every pixel value is scaled between 0 and 1 — normalization doesn't change how it looks, only the numbers behind it." |
| **3. Simulated missing pixels** | Same panda with red dots scattered on it | "These red dots are pixels I randomly marked as missing (NaN), just to demonstrate how missing data is handled." |
| **4. After Mean Imputation** | The panda again, with faint colored speckles where the red dots were | "SimpleImputer filled each missing pixel back in using the average (mean) value of that color channel — that's why you see faint color noise instead of red now; each R, G, B channel was filled independently, so occasionally the color at that exact spot looks slightly off, but the image is fully reconstructed." |
| **5. Brightness category map** | The panda shown in black / gray / white | "This is the categorical column I created — each pixel labeled dark, mid, or light based on brightness — and this is exactly the column I then one-hot encoded, the same way the Country column was one-hot encoded in the handout." |
| **6. MinMaxScaler** | Green histogram, same shape as the original | "This shows MinMaxScaler — it squeezes pixel values into 0 to 1 but keeps the exact same distribution shape, just a straight rescale." |
| **7. StandardScaler** | Purple histogram, centered on the dashed line at 0 | "This shows StandardScaler — it re-centers the values around a mean of 0, and now some values are negative, unlike MinMaxScaler." |
| **8. Zoomed pixel grid (4×4)** | A small 4×4 patch of the image, blown up big, with grid lines, and each cell labeled with its actual R,G,B numbers | "This is a zoomed-in 4×4 patch of the image with grid lines drawn on it. Each grid cell is one pixel, and each pixel is exactly one row in my tabular dataset — so this box right here is literally what `pixel_table.csv` looks like, just drawn as a picture instead of a table." |

**One line to open with, and one to close with:**
- Opening: *"Sir, instead of explaining the code first, let me just show you the whole pipeline in one picture — everything at the 128×128 size I actually work with."*
- Closing: *"So panels 1–2 are resizing/normalizing, panels 3–4 are handling missing data, panel 5 is the categorical/one-hot part, panel 6–7 are the two feature-scaling techniques from Task 2, and panel 8 shows exactly how a pixel becomes a table row."*

---

## 1b. How mean is calculated — do this by hand, don't just say "the mean function"

If sir asks "how did you calculate the mean" for the Mean Imputation step,
don't just say `imputa.statistics_` or `.mean()` — show the actual formula
and a small worked example with real numbers from this project.

### The formula

> **Mean = (sum of all known values) ÷ (count of known values)**

In symbols: `mean = (x1 + x2 + x3 + ... + xn) / n` — and importantly, the
`NaN` (missing) entries are **skipped**, not counted as 0 and not counted in
`n`. If a missing value were wrongly treated as 0, it would drag the mean
down incorrectly — that's why we must exclude missing entries from both the
sum and the count.

### Tiny example first (say this out loud)

> "Suppose just 5 pixels' Red values are: 200, 210, missing, 195, 205.
> I ignore the missing one, add up the rest: 200 + 210 + 195 + 205 = 810.
> I divide by how many real values I added: 4.
> Mean = 810 ÷ 4 = 202.5.
> Then I put 202.5 in place of the missing value."

### Then show the REAL numbers from your project (Red channel)

These are the actual numbers from this run — pull them up if sir wants proof:

| Quantity | Value |
|---|---|
| Total R-channel pixel entries | 16,384 |
| Entries randomly marked missing (simulated) | 321 |
| Entries actually used to compute the mean | 16,384 − 321 = **16,063** |
| Sum of those 16,063 known R values | **3,231,671** |
| Mean = 3,231,671 ÷ 16,063 | **201.1872...  ≈ 201.19** |

> "This 201.19 is exactly the number `SimpleImputer` printed as the learned
> mean for the R column, and it's exactly the value it used to fill every
> missing R pixel — you can see it in `pixel_table_mean_imputed.csv`, e.g.
> row 17 of column 0 has `R_missing = NaN` and `R_mean_imputed = 201.19`."

### If sir asks "why divide by count of known values, not total count (16,384)?"

> "Because dividing by the total including missing entries would incorrectly
> treat every missing pixel as if it contributed 0 to the sum, which would
> pull the mean down artificially. We only want the average of the pixels we
> actually know the value of."

---

## 1. Imports — what each one does + error if removed

### `task1_image_preprocessing.py`

| Import | Say this to sir | If you delete this line, you get |
|---|---|---|
| `import os` | "Used to work with file paths and folders — like joining `dataset` + filename, and creating the `outputs` folder." | `NameError: name 'os' is not defined` |
| `import numpy as np` | "Used for all the math on pixel arrays — creating arrays, dividing, flattening." | `NameError: name 'np' is not defined` |
| `import pandas as pd` | "Used to build the tabular (row/column) table of pixels and save it as CSV." | `NameError: name 'pd' is not defined` |
| `from PIL import Image` | "Used to open, verify, resize, and convert the image file." | `NameError: name 'Image' is not defined` |
| `import matplotlib` + `matplotlib.use("Agg")` | "Tells matplotlib to save plots to a file instead of opening a window (no screen/display here)." | Without `Agg`, on a machine with no display, `plt.savefig()` can crash with a display error |
| `import matplotlib.pyplot as plt` | "Used to draw and save the before/after image comparison." | `NameError: name 'plt' is not defined` |
| `from sklearn.preprocessing import OneHotEncoder` | "Used to convert the categorical `brightness` column (dark/mid/light) into binary 0/1 columns." | `NameError: name 'OneHotEncoder' is not defined` |
| `from sklearn.impute import SimpleImputer` | "Used to fill missing pixel values with the column mean (Mean Imputation)." | `NameError: name 'SimpleImputer' is not defined` |

### `task2_feature_scaling.py`

| Import | Say this to sir | If you delete this line, you get |
|---|---|---|
| `import os` | "File paths again — to load `pixel_table.csv` and save results." | `NameError: name 'os' is not defined` |
| `import pandas as pd` | "To load the tabular pixel CSV and save the scaled versions." | `NameError: name 'pd' is not defined` |
| `import numpy as np` | "For array math when computing min/max/mean/std of pixel values." | `NameError: name 'np' is not defined` |
| `import matplotlib` / `matplotlib.pyplot as plt` | "To draw the before/after scaling histograms." | `NameError: name 'plt' is not defined` |
| `from sklearn.preprocessing import StandardScaler, MinMaxScaler` | "The two scaling techniques the lab asks for — Standardization and Normalization." | `NameError: name 'StandardScaler' is not defined` (and same for `MinMaxScaler`) |

**One rule to remember:** almost every import removal gives the same kind of
error — `NameError: name 'X' is not defined` — because Python simply doesn't
know what that word means anymore without the import. Sir will likely ask
this as a trick question, so just say:
> "If I remove an import, Python throws a `NameError` the moment that name is
> used, because the import is what tells Python where that function/class
> comes from."

---

## 2. How to explain the whole thing to sir (talking script)

Say it in this order, it matches how the code actually runs:

1. **"I picked one image as my dataset"** — a 720×720 panda image
   (`dataset/panda_original.jpg`).
2. **"I load it and check it's not missing or corrupted"** — I check the
   file exists, then use `Image.verify()` inside a `try/except`. If the file
   were missing or broken, my code would print a clear message and stop
   instead of crashing.
3. **"I resize it to 128×128"** — using `image.resize((128,128))`, so every
   image in a bigger dataset would end up the same size (required for ML
   models).
4. **"I normalize every pixel to 0–1"** — dividing every pixel value by 255,
   because raw 0–255 values are too large for models to train well on.
5. **"I convert the image into a table"** — every pixel becomes one row, with
   columns for its position (`row`, `col`) and color (`R`, `G`, `B`). A
   128×128 image gives 16,384 rows. This is exactly like a CSV file, just
   built from image pixels.
6. **"I simulate missing pixel values and fix them with Mean Imputation"** —
   since my real image has no missing pixels, I randomly mark ~2% of them as
   missing (`NaN`) to demonstrate the technique, then use
   `SimpleImputer(strategy='mean')` to fill them back in with the column's
   average value — same method sir taught with `Age`/`Salary`.
7. **"I one-hot encode a categorical column"** — I calculate each pixel's
   brightness and label it dark/mid/light, then one-hot encode that column
   into 3 binary columns, same idea as the Country column in the handout.
8. **"I visualize original vs preprocessed"** — a side-by-side image so you
   can see the resize + normalization worked.
9. **"In Task 2, I scale the pixel features"** — I take the R/G/B columns
   from my tabular data and apply both `StandardScaler` and `MinMaxScaler`,
   then plot histograms to show how each one changes the pixel-value
   distribution.

---

## 3. MinMaxScaler vs StandardScaler — show the difference, one by one

Say: **"Let me show you the same pixel values scaled two different ways."**

### Formulas

- **MinMaxScaler (Normalization):** `X' = (X - Xmin) / (Xmax - Xmin)` → squeezes values into a fixed `[0, 1]` range.
- **StandardScaler (Standardization):** `X' = (X - mean) / std` → centers values around 0, with no fixed upper/lower bound.

### Real numbers from this project's run (Red channel)

| Stage | Min | Max | Mean | Std |
|---|---|---|---|---|
| Original (raw pixel values) | 0 | 255 | 201.21 | 87.56 |
| After **StandardScaler** | ≈ -2.41 | ≈ 0.66 | 0.00 | 1.00 |
| After **MinMaxScaler** | 0.00 | 1.00 | 0.78 | 0.34 |

### One-by-one, pick one pixel and walk through it

Say: **"Take one pixel with R = 44."**

- **StandardScaler:** `(44 - 201.21) / 87.56 = -1.795` → a negative number,
  because 44 is below the average (201.21).
- **MinMaxScaler:** `(44 - 0) / (255 - 0) = 0.173` → a small positive
  fraction, because 44 is closer to the minimum (0) than the maximum (255).

Say: **"Notice StandardScaler can go negative, MinMaxScaler never can — it's
always between 0 and 1."**

### Show the picture

Open `outputs/scaling_distributions.png` — 3 panels side by side:
1. Original pixel values (0–255)
2. After StandardScaler (centered at 0, some negative)
3. After MinMaxScaler (squeezed into 0–1)

Say: **"The shape of the histogram is exactly the same in panel 1 and panel
3 — MinMaxScaler is just a straight-line rescale, it doesn't distort the
distribution. StandardScaler shifts the bars around 0 instead."**

### When to use which (your answer if sir asks)

- **Use MinMaxScaler for raw image pixels** — because pixel values already
  have a known fixed range (0–255), so scaling to [0,1] is simple and it's
  what most image models (CNNs) expect.
- **Use StandardScaler for other numeric features** — e.g. if you later
  extract features like edge count or color histogram values that don't
  have a fixed range, or for algorithms like PCA/SVM/k-means that assume
  data centered around 0.

---

## 4. Quick-reference: files to open during viva

| Show sir this file | To prove |
|---|---|
| `outputs/preprocessing_pipeline_view.png` | **The whole pipeline in one picture — open this first** |
| `outputs/panda_resized_128x128.jpg` | Image was resized to 128×128 |
| `outputs/original_vs_preprocessed.png` | Original vs normalized comparison |
| `outputs/pixel_table.csv` | Image converted into a table (16,384 rows) |
| `outputs/pixel_table_mean_imputed.csv` | Mean imputation before/after columns |
| `outputs/pixel_table_encoded.csv` | One-hot encoded brightness columns |
| `outputs/pixel_table_standardscaled.csv` | StandardScaler output columns |
| `outputs/pixel_table_minmaxscaled.csv` | MinMaxScaler output columns |
| `outputs/scaling_distributions.png` | Visual before/after comparison of both scalers |
