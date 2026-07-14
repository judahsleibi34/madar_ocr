import fitz


def extract_pdf_text(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        page_lines = []
        for b in blocks:
            text = b[4].strip()
            if text:
                page_lines.append({
                    "text": text,
                    "bbox": [b[0], b[1], b[2], b[3]]
                })
        pages_data.append({"page": page_num + 1, "blocks": page_lines})

    doc.close()
    return pages_data
