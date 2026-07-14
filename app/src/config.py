from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT_DIR / "data" / "input"
OUTPUT_DIR = ROOT_DIR / "data" / "output"

PDF_OUTPUT_DIR = OUTPUT_DIR / "pdf"
DOCX_OUTPUT_DIR = OUTPUT_DIR / "docx"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEXT_LAYER_CHARS = 20
