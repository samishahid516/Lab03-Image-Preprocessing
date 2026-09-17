"""
Generates a Word (.docx) lab journal for Lab # 03 - Task 1 & Task 2,
combining the objective, code, and actual outputs/screenshots from this
project into one document ready to upload/submit.

Run this AFTER task1_image_preprocessing.py and task2_feature_scaling.py
(and view_preprocessing_pipeline.py) have been run at least once, since it
reads their generated files from outputs/.
"""

import os
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

OUTPUT_DIR = "outputs"
JOURNAL_PATH = os.path.join(OUTPUT_DIR, "Lab03_Journal_Image_Preprocessing.docx")

STUDENT_NAME = "____________________"   # fill in before submitting
ROLL_NO = "____________________"        # fill in before submitting


def set_cell_font(cell, text, size=9, bold=False, mono=True):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    if mono:
        run.font.name = "Consolas"
        r = run._element.rPr.rFonts
        r.set(qn("w:eastAsia"), "Consolas")


def add_code_block(doc, code_text, font_size=8.5):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(code_text)
    run.font.name = "Consolas"
    run.font.size = Pt(font_size)
    r = run._element.rPr.rFonts
    r.set(qn("w:eastAsia"), "Consolas")
    # light shading behind the code block
    shd = doc.add_paragraph
    return p


def add_shaded_paragraph_border(paragraph):
    pPr = paragraph._p.get_or_add_pPr()
    shd = pPr.makeelement(qn("w:shd"), {
        qn("w:val"): "clear", qn("w:color"): "auto", qn("w:fill"): "F2F2F2"
    })
    pPr.append(shd)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    return h


def add_dataframe_table(doc, df, max_rows=8, float_fmt="{:.2f}"):
    df_show = df.head(max_rows)
    table = doc.add_table(rows=1, cols=len(df_show.columns))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df_show.columns):
        set_cell_font(hdr_cells[i], str(col), size=8, bold=True, mono=False)
    for _, row in df_show.iterrows():
        cells = table.add_row().cells
        for i, col in enumerate(df_show.columns):
            val = row[col]
            if isinstance(val, float):
                text = float_fmt.format(val)
            else:
                text = str(val)
            set_cell_font(cells[i], text, size=8, mono=True)
    return table


doc = Document()

# Default document font
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

# ---------------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------------
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Lab # 03")
run.bold = True
run.font.size = Pt(28)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run("Image Data Handling & Preprocessing")
run.bold = True
run.font.size = Pt(18)

subtitle2 = doc.add_paragraph()
subtitle2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle2.add_run("Task 1: Load, Resize, Clean & Encode Image Data\nTask 2: Feature Scaling (StandardScaler & MinMaxScaler)")
run.font.size = Pt(13)
run.italic = True

doc.add_paragraph()
info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
info.add_run(f"Student Name: {STUDENT_NAME}\n").font.size = Pt(12)
info.add_run(f"Roll No: {ROLL_NO}\n").font.size = Pt(12)
info.add_run("Course: Data Preprocessing / Machine Learning Lab\n").font.size = Pt(12)

doc.add_page_break()

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
add_heading(doc, "1. Objective", level=1)
doc.add_paragraph(
    "To implement various image data handling and preprocessing techniques "
    "using Python: loading, exploring, resizing, normalizing, handling "
    "missing/corrupted image data, converting image pixels into a tabular "
    "(row/column) dataset, one-hot encoding a categorical column, and "
    "visualizing original vs preprocessed images (Task 1). Then, applying "
    "and comparing two feature scaling techniques - StandardScaler and "
    "MinMaxScaler - on the tabular pixel data, and visualizing the pixel "
    "value distributions before and after scaling (Task 2)."
)

add_heading(doc, "2. Tools & Libraries Used", level=1)
tools = [
    ("os", "File and folder path handling."),
    ("numpy", "Numeric array operations on pixel data."),
    ("pandas", "Building and saving the tabular pixel dataset (DataFrame/CSV)."),
    ("PIL (Pillow) - Image", "Opening, verifying, resizing, and converting the image."),
    ("matplotlib.pyplot", "Plotting and saving all visualizations."),
    ("sklearn.preprocessing.OneHotEncoder", "One-hot encoding the categorical brightness column."),
    ("sklearn.impute.SimpleImputer", "Mean imputation of simulated missing pixel values."),
    ("sklearn.preprocessing.StandardScaler", "Standardization: (X - mean) / std."),
    ("sklearn.preprocessing.MinMaxScaler", "Normalization: (X - min) / (max - min)."),
]
table = doc.add_table(rows=1, cols=2)
table.style = "Light Grid Accent 1"
hdr = table.rows[0].cells
set_cell_font(hdr[0], "Library / Import", bold=True, mono=False, size=10)
set_cell_font(hdr[1], "Purpose in this project", bold=True, mono=False, size=10)
for name, purpose in tools:
    cells = table.add_row().cells
    set_cell_font(cells[0], name, mono=True, size=9)
    set_cell_font(cells[1], purpose, mono=False, size=9)

add_heading(doc, "3. Dataset", level=1)
doc.add_paragraph(
    "A single image (dataset/panda_original.jpg, 720x720 pixels) is used as "
    "the dataset. Every pixel of this image, after resizing to 128x128, is "
    "treated as one row of a tabular dataset - exactly like each row of "
    "purchases.csv represented one customer observation in the lab handout, "
    "except here each row is one pixel observation with columns for its "
    "position (row, col) and its Red/Green/Blue color values."
)

doc.add_page_break()

# ---------------------------------------------------------------------------
# TASK 1
# ---------------------------------------------------------------------------
add_heading(doc, "Task 1: Image Loading & Preprocessing", level=1)

add_heading(doc, "4.1 Code", level=2)
with open("task1_image_preprocessing.py") as f:
    code1 = f.read()
add_code_block(doc, code1)

doc.add_page_break()

add_heading(doc, "4.2 Output - Console Log (Task 1)", level=2)
with open(os.path.join(OUTPUT_DIR, "console_log_task1.txt")) as f:
    log1 = f.read()
add_code_block(doc, log1, font_size=7.5)

doc.add_page_break()

add_heading(doc, "4.3 Output - Full Pipeline View (all steps in one picture)", level=2)
doc.add_picture(os.path.join(OUTPUT_DIR, "preprocessing_pipeline_view.png"), width=Inches(6.3))

add_heading(doc, "4.4 Output - Original (128x128) vs Preprocessed (Normalized)", level=2)
doc.add_picture(os.path.join(OUTPUT_DIR, "original_vs_preprocessed.png"), width=Inches(6.0))

doc.add_page_break()

add_heading(doc, "4.5 Output - Tabular Pixel Dataset (pixel_table.csv, first 8 rows)", level=2)
df1 = pd.read_csv(os.path.join(OUTPUT_DIR, "pixel_table.csv"))
add_dataframe_table(doc, df1, max_rows=8)

add_heading(doc, "4.6 Output - Mean Imputation Sample (missing vs imputed pixel values)", level=2)
df2 = pd.read_csv(os.path.join(OUTPUT_DIR, "pixel_table_mean_imputed.csv"))
missing_rows = df2[df2[["R_missing", "G_missing", "B_missing"]].isna().any(axis=1)]
cols_show = ["row", "col", "R_missing", "G_missing", "B_missing",
             "R_mean_imputed", "G_mean_imputed", "B_mean_imputed"]
add_dataframe_table(doc, missing_rows[cols_show], max_rows=6)

doc.add_page_break()

add_heading(doc, "4.7 Output - One-Hot Encoded Brightness Column (pixel_table_encoded.csv)", level=2)
df3 = pd.read_csv(os.path.join(OUTPUT_DIR, "pixel_table_encoded.csv"))
cols_show2 = ["row", "col", "R", "G", "B", "brightness",
              "brightness_dark", "brightness_light", "brightness_mid"]
add_dataframe_table(doc, df3[cols_show2], max_rows=8)

doc.add_page_break()

# ---------------------------------------------------------------------------
# TASK 2
# ---------------------------------------------------------------------------
add_heading(doc, "Task 2: Feature Scaling (StandardScaler & MinMaxScaler)", level=1)

add_heading(doc, "5.1 Code", level=2)
with open("task2_feature_scaling.py") as f:
    code2 = f.read()
add_code_block(doc, code2)

doc.add_page_break()

add_heading(doc, "5.2 Output - Console Log (Task 2)", level=2)
with open(os.path.join(OUTPUT_DIR, "console_log_task2.txt")) as f:
    log2 = f.read()
add_code_block(doc, log2, font_size=7.5)

doc.add_page_break()

add_heading(doc, "5.3 Output - Pixel Value Distributions Before/After Scaling", level=2)
doc.add_picture(os.path.join(OUTPUT_DIR, "scaling_distributions.png"), width=Inches(6.3))

add_heading(doc, "5.4 Output - Sample Scaled Values", level=2)
df_std = pd.read_csv(os.path.join(OUTPUT_DIR, "pixel_table_standardscaled.csv"))
df_mm = pd.read_csv(os.path.join(OUTPUT_DIR, "pixel_table_minmaxscaled.csv"))

doc.add_paragraph("After StandardScaler:").runs[0].bold = True
add_dataframe_table(doc, df_std[["row", "col", "R", "G", "B", "R_standard", "G_standard", "B_standard"]], max_rows=6)

doc.add_paragraph("After MinMaxScaler:").runs[0].bold = True
add_dataframe_table(doc, df_mm[["row", "col", "R", "G", "B", "R_minmax", "G_minmax", "B_minmax"]], max_rows=6)

doc.add_page_break()

add_heading(doc, "5.5 Discussion: MinMaxScaler vs StandardScaler", level=2)
discussion_points = [
    ("MinMaxScaler (Normalization)",
     "Formula: X' = (X - Xmin) / (Xmax - Xmin). Rescales values into a fixed "
     "[0, 1] range. Ideal for image pixel data because pixel intensities "
     "already have a known, fixed range (0-255). Commonly used before "
     "feeding images into neural networks / CNNs. Sensitive to outliers."),
    ("StandardScaler (Standardization)",
     "Formula: X' = (X - mean) / std. Centers values around 0 with unit "
     "standard deviation; does not bound values to a fixed range. Useful "
     "for algorithms like PCA, linear/logistic regression, SVM, and k-means "
     "that assume zero-centered features, or when features have very "
     "different natural scales."),
    ("Recommendation for this project",
     "MinMaxScaler is generally preferred for raw image pixel values, since "
     "the physical range (0-255) is fixed and known, making [0, 1] scaling "
     "simple and consistent with what most image-based ML/DL pipelines "
     "expect. StandardScaler becomes more relevant for derived numeric "
     "features (e.g., extracted color histograms or embeddings) with "
     "unknown/unbounded ranges."),
]
for heading_text, body_text in discussion_points:
    p = doc.add_paragraph()
    p.add_run(heading_text + ": ").bold = True
    p.add_run(body_text)

doc.add_page_break()

# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------
add_heading(doc, "6. Conclusion", level=1)
doc.add_paragraph(
    "In this lab, a single image dataset was successfully loaded, verified "
    "for corruption, resized to 128x128, and normalized to a [0, 1] pixel "
    "range. The image was converted into a tabular dataset of 16,384 rows "
    "(one per pixel), missing pixel values were simulated and repaired "
    "using Mean Imputation (SimpleImputer), a categorical brightness column "
    "was one-hot encoded, and the pipeline was visualized end-to-end. In "
    "Task 2, StandardScaler and MinMaxScaler were both applied to the pixel "
    "features and compared visually and numerically, showing that "
    "MinMaxScaler preserves the original distribution shape while bounding "
    "values to [0, 1], whereas StandardScaler re-centers values around a "
    "mean of 0 with unit variance. These preprocessing steps mirror the "
    "same techniques taught for tabular CSV data in the lab handout, "
    "applied instead to image pixel data."
)

doc.save(JOURNAL_PATH)
print(f"Saved -> {JOURNAL_PATH}")
