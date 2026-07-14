import fitz
from src.config import MIN_TEXT_LAYER_CHARS


def has_text_layer(pdf_path: str, min_chars: int = MIN_TEXT_LAYER_CHARS) -> bool:
    doc = fitz.open(pdf_path)
    total_chars = 0

    for page in doc:
        total_chars += len(page.get_text().strip())
        if total_chars > min_chars:
            doc.close()
            return True

    doc.close()
    return False


def classify_pdf(pdf_path: str) -> dict:
    is_digital = has_text_layer(pdf_path)
    return {
        "file": pdf_path,
        "type": "digital" if is_digital else "scanned",
    }
