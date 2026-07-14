import pdfplumber


def extract_tables(pdf_path: str) -> list:
    tables_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table_index, table in enumerate(tables):
                tables_data.append({
                    "page": page_num + 1,
                    "table_index": table_index,
                    "rows": table
                })

    return tables_data
