import pymupdf
from pathlib import Path

def load_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF page by page.

    Returns a list of dictionaries containing:
    - text: extracted page text
    - page: page number (starting from 1)
    - source: PDF filename
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The provided file is not a PDF.")

    extracted_pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()

            if not text:
                continue

            extracted_pages.append({
                "text": text,
                "page": page_number,
                "source": pdf_path.name
            })

    return extracted_pages