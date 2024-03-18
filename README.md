# Table parsing experiments
This repo contains experiments in automated table parsing. See `main.py` for an example.

Basic approach:

1. If parsing from a PDF, convert the PDF pages to images. 
1. Run the table detection model to locate one or more tables on a page and crop out the tables as separate images.
1. Run the table structure model on each table image to identify likely rows and columns of data.
1. Use OCR to get the all text strings in the table image along with the location of the text. Alternatively, if the table is from a PDF and a PDF library can extract the table content as text strings we can use the PDF library to get the same information with more accuracy and speed.
1. Allocate the text strings into the table cells bsaed on which cell they have the most overlap with, and finally refine the table.

# Credits

This repo draws on work from the following sources:

* Microsoft [Table Transformer](https://github.com/microsoft/table-transformer) library
* The [Unstructured](https://github.com/Unstructured-IO/unstructured) library
* [Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Table%20Transformer) from Niels Rogge