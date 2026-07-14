import zipfile
from pathlib import Path


def extract_docx_images(docx_path: str, output_dir: str) -> list:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_data = []

    with zipfile.ZipFile(docx_path) as z:
        media_files = [f for f in z.namelist() if f.startswith("word/media/")]
        for media_file in media_files:
            filename = Path(media_file).name
            filepath = output_path / filename
            with z.open(media_file) as source, open(filepath, "wb") as target:
                target.write(source.read())
            images_data.append({"file": str(filepath)})

    return images_data
