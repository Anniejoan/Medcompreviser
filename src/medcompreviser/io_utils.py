from pathlib import Path
from typing import Dict, Any
from pypdf import PdfReader


def read_pdf_text(pdf_path: str) -> str:
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts).strip()


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)