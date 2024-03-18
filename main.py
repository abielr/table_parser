import base64
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import pandas as pd
import io
import fitz
from pdf2image import convert_from_path
import copy
from transformers import TableTransformerForObjectDetection, AutoModelForObjectDetection
from PIL import ImageDraw
import utils
import ocr_utils
device = "cuda" if torch.cuda.is_available() else "cpu"

PATH = Path(__file__).parent.resolve()


################################
# Load models
################################

detection_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
detection_model.to(device)

structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-fin")
structure_model.to(device)

id2label = detection_model.config.id2label
id2label[len(detection_model.config.id2label)] = "no object"

structure_id2label = structure_model.config.id2label
structure_id2label[len(structure_id2label)] = "no object"

style = """
<style>
        table, th, td {
                border: 1px solid black
        }
</style>
"""

################################
# Convert PDF to images
################################

input_file = PATH / "data/The Supplemental Information_3Q23_ADA.pdf"
doc = fitz.open(input_file)
images = convert_from_path(input_file)
page_idx = 5
table_idx = 1

################################
# Table detection
################################

page_image = images[page_idx]
pixel_values = utils.detection_transform(page_image).unsqueeze(0)
pixel_values = pixel_values.to(device)

with torch.no_grad():
     table_outputs = detection_model(pixel_values)
table_objects = utils.outputs_to_objects(table_outputs, page_image.size, id2label)

# fig = utils.visualize_detected_tables(page_image, table_objects)

tokens = []
crop_padding = 10
tables_crops = utils.objects_to_crops(page_image, tokens, table_objects, utils.detection_class_thresholds, padding=crop_padding)
cropped_tables = [t['image'].convert("RGB") for t in tables_crops]
cropped_table = cropped_tables[table_idx]

################################
# Table structure recognition
################################

pixel_values = utils.structure_transform(cropped_table).unsqueeze(0)
pixel_values = pixel_values.to(device)

with torch.no_grad():
  structure_outputs = structure_model(pixel_values)

structure_outputs = utils.outputs_to_objects(structure_outputs, cropped_table.size, structure_id2label)

# cropped_table_visualized = cropped_table.copy()
# draw = ImageDraw.Draw(cropped_table_visualized)

# for cell in structure_outputs:
#     draw.rectangle(cell["bbox"], outline="red")

# cropped_table_visualized

################################
# Use OCR or PDF reader to get tokens
################################

# tokens = ocr_utils.get_ocr_layout_tesseract(cropped_table)
# tokens_paddle = ocr_utils.get_ocr_layout_paddle(cropped_table)
tokens = ocr_utils.get_layout_pymupdf(doc[page_idx], page_image, table_objects, crop_padding=crop_padding, padding=10)[table_idx]

selected_tokens = tokens
tables_structure = utils.objects_to_structures(copy.deepcopy(structure_outputs), selected_tokens, utils.structure_class_thresholds)
tables_cells = [utils.structure_to_cells(structure, selected_tokens)[0] for structure in tables_structure]
tables_htmls = [utils.cells_to_html(cells, use_rowspans=False) for cells in tables_cells]
tables_csvs = [utils.cells_to_csv(cells) for cells in tables_cells]

# Write out HTML
with open(PATH / "output/test.html", "w") as fh: fh.write(f'<head><meta charset="UTF-8">{style}</head>' + tables_htmls[0])

# Convert to DataFrame
pd.read_csv(io.StringIO(tables_csvs[0]))

# cropped_table_visualized = cropped_table.copy()
# draw = ImageDraw.Draw(cropped_table_visualized)

# for cell in [x for x in structure_outputs if x['label'] == 'table projected row header']:
#     draw.rectangle(cell["bbox"], outline="green")
# for cell in [x for x in structure_outputs if x['label'] == 'table row']:
#     draw.rectangle(cell["bbox"], outline="green")
# for cell in tables_structure[0]['spanning cells']:
#     draw.rectangle(cell["bbox"], outline="red")
# for item in tables_structure[0].values():
#     for cell in item:
#         draw.rectangle(cell["bbox"], outline="red")
# for cell in tables_cells[0]:
#         draw.rectangle(cell["bbox"], outline="red")

# cropped_table_visualized