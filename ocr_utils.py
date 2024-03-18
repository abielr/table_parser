import pandas as pd
from paddleocr import PaddleOCR
import pytesseract
from pytesseract import Output
from PIL import Image
import PIL
import numpy as np
import fitz

# Some functions in this file are borrowed from the unstructured_inference library

def parse_ocr_data_paddle(ocr_data: list) -> list:
    text_regions = []
    for idx in range(len(ocr_data)):
        res = ocr_data[idx]
        if not res:
            continue

        for line in res:
            x1 = min([i[0] for i in line[0]])
            y1 = min([i[1] for i in line[0]])
            x2 = max([i[0] for i in line[0]])
            y2 = max([i[1] for i in line[0]])
            text = line[1][0]
            if not text:
                continue
            cleaned_text = text.strip()
            if cleaned_text:
                text_region = dict(
                    bbox=[x1, y1, x2, y2],
                    text=cleaned_text
                )
                text_regions.append(text_region)

    return text_regions

def parse_ocr_data_tesseract(ocr_data: pd.DataFrame) -> list:
    zoom = 1

    text_regions = []
    for idtx in ocr_data.itertuples():
        text = idtx.text
        if not text:
            continue

        cleaned_text = str(text) if not isinstance(text, str) else text.strip()

        if cleaned_text:
            x1 = idtx.left / zoom
            y1 = idtx.top / zoom
            x2 = (idtx.left + idtx.width) / zoom
            y2 = (idtx.top + idtx.height) / zoom
            text_region = dict(
                bbox=[x1, y1, x2, y2],
                text=cleaned_text
            )
            text_regions.append(text_region)

    return text_regions

def get_ocr_layout_paddle(image: Image) -> list[dict]:
    # TODO: Make it so PaddleOCR() is only called once
    paddle_ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        enable_mkldnn=False
    )

    tokens = parse_ocr_data_paddle(paddle_ocr.ocr(np.array(image), cls=True))
    for idx, token in enumerate(tokens):
        if "span_num" not in token:
            token["span_num"] = idx
        if "line_num" not in token:
            token["line_num"] = 0
        if "block_num" not in token:
            token["block_num"] = 0
    return tokens

def get_ocr_layout_tesseract(image: Image) -> list[dict]:
    tokens = pytesseract.image_to_data(image, output_type=Output.DATAFRAME, lang='eng')
    tokens = tokens.dropna()
    tokens = parse_ocr_data_tesseract(tokens)
    for idx, token in enumerate(tokens):
        if "span_num" not in token:
            token["span_num"] = idx
        if "line_num" not in token:
            token["line_num"] = 0
        if "block_num" not in token:
            token["block_num"] = 0
    return tokens

def get_layout_pymupdf(page: fitz.Page, page_image: PIL.Image, table_objects: list[dict], crop_padding: int = 0, padding: int = 0) -> list[list[dict]]:
    """
    Extract text from a PDF page and translate the text bounding box coordinates
    to align with the image coordinates of the page, then filter tokens that are
    part of detected tables and update the bounding boxes to be relative to the
    table image coordinates

    Args:
        page (fitz.Page): PyMuPDF PDF page object
        page_image (PIL.Image): Image of the PDF page
        table_objects (list[dict]): A list of detected tables and their bounding boxes, as output by the table-transformers detection model
        crop_padding (int): The amount of additional pixel space that was added around the bounding boxes specified in table_objects when the table was cropped
        padding (int): The pixel distance beyond the table that we allow for text to spill into and still be included in the table
    Returns:
        list: List of words in each table with bounding boxes whose coordinates are relative to the table image
    """
    words = [list(x) for x in page.get_text('words')]
    # words.sort(key=lambda x: (x[1], x[0])) # Sort in reading order
    # Rescale PDF coordinates to image coordinates
    x_scaling = page_image.size[0] / page.rect[2]
    y_scaling = page_image.size[1] / page.rect[3]
    for idx in range(len(words)):
        words[idx][0] = words[idx][0] * x_scaling
        words[idx][1] = words[idx][1] * y_scaling
        words[idx][2] = words[idx][2] * x_scaling
        words[idx][3] = words[idx][3] * y_scaling
    token_list = []
    for table_object in table_objects:
        tokens = []
        x1_, y1_, x2_, y2_ = table_object['bbox']
        x1_, y1_, x2_, y2_ = x1_-crop_padding, y1_-crop_padding, x2_+crop_padding, y2_+crop_padding
        for idx, rec in enumerate(words):
            x1, y1, x2, y2, text = rec[:5]
            if x1+padding >= x1_ and x2-padding <= x2_ and y1+padding >= y1_ and y2-padding <= y2_: # Filter only for words inside the table rectangle
                tokens.append(dict(
                    bbox=[x1-x1_, y1-y1_, x2-x1_, y2-y1_], # Shift from page image coordinates to table image coordinates
                    text=text.strip(),
                    span_num=idx,
                    line_num=0,
                    block_num=0
                ))
        token_list.append(tokens)
    return token_list