import pandas as pd
import docx
import pptx
import fitz  # PyMuPDF

from odf import text as odf_text
from odf.opendocument import load as odf_load
from odf.table import Table, TableRow, TableCell
from pydocx import PyDocX
from bs4 import BeautifulSoup
import mimetypes
import os

from src.state_models import FileProcessingState
from src.utils.logger import get_logger
from src.utils.image_converter import (
    convert_to_standard_jpeg, cleanup_temp_file
)


logger = get_logger("process_and_extract")


def get_file_type(state: FileProcessingState) -> FileProcessingState:
    """Identifies the file's MIME type to enable correct routing."""
    file_path = state.file_path
    if not file_path:
        state.file_type = "unknown"
        state.extracted_text = ""
        state.extracted_images = []
        logger.warning("No file path provided, setting file type to unknown")
        return state

    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        ext = os.path.splitext(file_path)[-1].lower()
        # A more comprehensive MIME map
        mime_map = {
            '.docx': ('application/vnd.openxmlformats-officedocument.'
                      'wordprocessingml.document'),
            '.doc': 'application/msword',
            '.pptx': ('application/vnd.openxmlformats-officedocument.'
                      'presentationml.presentation'),
            '.xlsx': ('application/vnd.openxmlformats-officedocument.'
                      'spreadsheetml.sheet'),
            '.xls': 'application/vnd.ms-excel',
            '.odt': 'application/vnd.oasis.opendocument.text',
            '.ods': 'application/vnd.oasis.opendocument.spreadsheet',
            '.odp': 'application/vnd.oasis.opendocument.presentation',
            '.csv': 'text/csv',
            '.tsv': 'text/tab-separated-values',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.pdf': 'application/pdf',
        }
        mime_type = mime_map.get(ext, 'application/octet-stream')

    state.file_type = mime_type
    state.extracted_text = ""
    state.extracted_images = []
    logger.info(f"Identified file type: {state.file_type} " +
                f"for {state.file_path}")
    return state


def extract_doc_content(state: FileProcessingState) -> FileProcessingState:
    """Extracts text from legacy .doc files using pydocx."""
    logger.info(f"Extracting content from .doc file: {state.file_path}")

    try:
        html = PyDocX(state.file_path).to_html()
        soup = BeautifulSoup(html, 'html.parser')
        state.extracted_text = soup.get_text()
        logger.info(f"Extracted text length: {len(state.extracted_text)}")
    except Exception as e:
        state.extracted_text = f"Error processing .doc file: {e}"
        logger.error("Error processing .doc file: %s", e)
    return state


def extract_text_content(state: FileProcessingState) -> FileProcessingState:
    """Extracts text from text files."""
    logger.info(f"Extracting content from text file: {state.file_path}")

    try:
        with open(state.file_path, 'r', encoding='utf-8') as f:
            state.extracted_text = f.read()
        logger.info(f"Extracted text length: {len(state.extracted_text)}")
        logger.debug("Extracted text snippet: " +
                     f"{state.extracted_text[:200]}...")
    except FileNotFoundError:
        state.extracted_text = f"Error: File not found at {state.file_path}"
        logger.error("File not found: %s", state.file_path)
    except Exception as e:
        state.extracted_text = f"Error processing text file: {e}"
        logger.error("Error processing text file: %s", e)
    return state


def extract_csv_tsv_content(state: FileProcessingState) -> FileProcessingState:
    """Extracts text from CSV/TSV files."""
    logger.info(f"Extracting content from CSV/TSV file: {state.file_path}")

    try:
        separator = '\t' if 'tab-separated-values' in state.file_type else ','
        df = pd.read_csv(state.file_path, sep=separator,
                         on_bad_lines='skip',
                         encoding='utf-8')
        state.extracted_text = df.to_string()
        logger.info(f"Extracted text length: {len(state.extracted_text)}")
    except Exception as e:
        state.extracted_text = f"Error processing CSV/TSV file: {e}"
        logger.error("Error processing CSV/TSV file: %s", e)
    return state


def extract_excel_content(state: FileProcessingState) -> FileProcessingState:
    """Extracts text from Excel files."""
    logger.info(f"Extracting content from Excel file: {state.file_path}")

    try:
        xls = pd.ExcelFile(state.file_path)
        text_content = ["--- Sheet: " +
                        f"{name} ---\n{xls.parse(name).to_string()}" for name
                        in xls.sheet_names]
        state.extracted_text = "\n\n".join(text_content)
        logger.info(f"Extracted text length: {len(state.extracted_text)}")
    except Exception as e:
        state.extracted_text = f"Error processing Excel file: {e}"
        logger.error("Error processing Excel file: %s", e)
    return state


def extract_docx_content(state: FileProcessingState) -> FileProcessingState:
    """Extracts text from DOCX files."""
    logger.info(f"Extracting content from DOCX file: {state.file_path}")

    try:
        doc = docx.Document(state.file_path)
        state.extracted_text = "\n".join([para.text for para in
                                          doc.paragraphs])
        logger.debug("Extracted text snippet: " +
                     f"{state.extracted_text[:200]}...")
        logger.info(f"Extracted text length: {len(state.extracted_text)}")
    except Exception as e:
        state.extracted_text = f"Error processing DOCX file: {e}"
        logger.error("Error processing DOCX file: %s", e)
    return state


def extract_pptx_content(state: FileProcessingState) -> FileProcessingState:
    """Extracts text from PowerPoint files."""
    logger.info(f"Extracting content from PPTX file: {state.file_path}")

    try:
        prs = pptx.Presentation(state.file_path)
        state.extracted_text = "\n".join(
            shape.text for slide in prs.slides for shape
            in slide.shapes if hasattr(shape, "text")
        )
        logger.info(f"Extracted text length: {len(state.extracted_text)}")
    except Exception as e:
        state.extracted_text = f"Error processing PPTX file: {e}"
        logger.error("Error processing PPTX file: %s", e)
    return state


def extract_pdf_content(state: FileProcessingState) -> FileProcessingState:
    logger.info(f"Extracting content from PDF file: {state.file_path}")

    try:
        doc = fitz.open(state.file_path)
        state.extracted_text = "\n".join([page.get_text() for page in doc])

        # Extract and convert images from PDF
        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_image = doc.extract_image(xref)

                # Save extracted image to temporary file for conversion
                import tempfile
                temp_img_path = tempfile.mktemp(suffix='.png')
                with open(temp_img_path, 'wb') as f:
                    f.write(base_image["image"])

                # Convert to standard JPEG format
                converted_path = convert_to_standard_jpeg(
                    temp_img_path,
                    max_resolution=512,
                    quality=70
                )

                if converted_path:
                    # Read the converted JPEG image
                    with open(converted_path, "rb") as f:
                        state.extracted_images.append(f.read())

                    # Clean up temporary files
                    cleanup_temp_file(converted_path)
                    cleanup_temp_file(temp_img_path)
                else:
                    # Fallback to original image if conversion fails
                    state.extracted_images.append(base_image["image"])
                    cleanup_temp_file(temp_img_path)

        logger.info(f"Extracted text length: {len(state.extracted_text)}")
        logger.info(f"Extracted image count: {len(state.extracted_images)}")
    except Exception as e:
        state.extracted_text = f"Error processing PDF file: {e}"
        logger.error("Error processing PDF file: %s", e)
    return state


def extract_image_content(state: FileProcessingState) -> FileProcessingState:
    logger.info(f"Extracting content from image file: {state.file_path}")

    try:
        # Convert image to standard JPEG format
        converted_path = convert_to_standard_jpeg(
            state.file_path,
            max_resolution=512,
            quality=70
        )

        if converted_path:
            # Read the converted JPEG image
            with open(converted_path, "rb") as f:
                state.extracted_images.append(f.read())

            # Clean up the temporary converted file
            cleanup_temp_file(converted_path)

            logger.info(
                "Successfully converted and extracted image to JPEG format"
            )
        else:
            # Fallback to original image if conversion fails
            logger.warning(
                "Failed to convert image to JPEG, using original format"
            )
            with open(state.file_path, "rb") as f:
                state.extracted_images.append(f.read())

        logger.info(f"Extracted image count: {len(state.extracted_images)}")
    except Exception as e:
        logger.error("Error processing image file: %s", e)
        state.extracted_text = f"Error processing image file: {e}"
    return state


def extract_odt_content(state: FileProcessingState) -> FileProcessingState:
    logger.info(f"Extracting content from ODT file: {state.file_path}")

    try:
        doc = odf_load(state.file_path)
        text_elements = doc.getElementsByType(odf_text.P)
        state.extracted_text = "\n".join([elem.text for elem in text_elements])
        logger.info(f"Extracted text length: {len(state.extracted_text)}")
    except Exception as e:
        state.extracted_text = f"Error processing ODT file: {e}"
        logger.error("Error processing ODT file: %s", e)
    return state


def extract_ods_content(state: FileProcessingState) -> FileProcessingState:
    logger.info(f"Extracting content from ODS file: {state.file_path}")

    try:
        doc = odf_load(state.file_path)
        tables = doc.getElementsByType(Table)
        text_content = []
        for table in tables:
            rows = table.getElementsByType(TableRow)
            for row in rows:
                cells = row.getElementsByType(TableCell)
                row_text = " | ".join([cell.text for cell in cells])
                text_content.append(row_text)
        state.extracted_text = "\n".join(text_content)
        logger.info(f"Extracted text length: {len(state.extracted_text)}")
    except Exception as e:
        state.extracted_text = f"Error processing ODS file: {e}"
        logger.error("Error processing ODS file: %s", e)
    return state


def extract_odp_content(state: FileProcessingState) -> FileProcessingState:
    logger.info(f"Extracting content from ODP file: {state.file_path}")

    try:
        doc = odf_load(state.file_path)
        text_elements = doc.getElementsByType(odf_text.P)
        state.extracted_text = "\n".join([elem.text for elem in text_elements])
        logger.info(f"Extracted text length: {len(state.extracted_text)}")
    except Exception as e:
        state.extracted_text = f"Error processing ODP file: {e}"
        logger.error("Error processing ODP file: %s", e)
    return state
