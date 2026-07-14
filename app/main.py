import json
import sys
from pathlib import Path
from src.ingestion.pdf_loader import classify_pdf
from src.extraction.digital_extractor import extract_pdf_text
from src.extraction.table_extractor import extract_tables
from src.extraction.image_extractor import extract_images
from src.extraction.docx_extractor import extract_docx_text, extract_docx_tables
from src.extraction.docx_image_extractor import extract_docx_images
from src.config import PDF_OUTPUT_DIR, DOCX_OUTPUT_DIR


def process_pdf(pdf_path: str):
    result = classify_pdf(pdf_path)
    print(f"Classified as: {result['type']}")

    if result["type"] != "digital":
        print("This PDF needs OCR - not implemented yet.")
        return

    text_data = extract_pdf_text(pdf_path)
    table_data = extract_tables(pdf_path)

    image_output_dir = PDF_OUTPUT_DIR / "images" / Path(pdf_path).stem
    image_data = extract_images(pdf_path, str(image_output_dir))

    save_output(pdf_path, text_data, table_data, image_data, PDF_OUTPUT_DIR)


def process_docx(docx_path: str):
    text_data = extract_docx_text(docx_path)
    table_data = extract_docx_tables(docx_path)

    image_output_dir = DOCX_OUTPUT_DIR / "images" / Path(docx_path).stem
    image_data = extract_docx_images(docx_path, str(image_output_dir))

    save_output(docx_path, text_data, table_data, image_data, DOCX_OUTPUT_DIR)


def save_output(file_path, text_data, table_data, image_data, output_dir):
    output = {
        "text": text_data,
        "tables": table_data,
        "images": image_data
    }

    output_file = output_dir / (Path(file_path).stem + ".json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Extracted data saved to: {output_file}")
    print(f"Tables found: {len(table_data)}")
    print(f"Images found: {len(image_data)}")


def process_file(file_path: str):
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        process_pdf(file_path)
    elif ext == ".docx":
        process_docx(file_path)
    else:
        print(f"Unsupported file type: {ext}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_file>")
        sys.exit(1)

    process_file(sys.argv[1])
