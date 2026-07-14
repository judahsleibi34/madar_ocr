import fitz
from pathlib import Path


def extract_images(pdf_path: str, output_dir: str) -> list:
    doc = fitz.open(pdf_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_data = []

    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            filename = f"page{page_num + 1}_img{img_index}.{ext}"
            filepath = output_path / filename

            with open(filepath, "wb") as f:
                f.write(image_bytes)

            images_data.append({
                "page": page_num + 1,
                "file": str(filepath)
            })

    doc.close()
    return images_data
