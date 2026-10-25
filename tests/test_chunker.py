from src.document_loader import load_pdf
from src.text_chunker import chunk_pages

pages = load_pdf("data/documents/sample.pdf")

chunks = chunk_pages(
    pages,
    chunk_size=500,
    chunk_overlap=100
)

print(f"Extracted pages: {len(pages)}")
print(f"Generated chunks: {len(chunks)}")

for index, chunk in enumerate(chunks[:5], start=1):
    print(f"\nChunk {index}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print(f"Text: {chunk['text'][:200]}")
