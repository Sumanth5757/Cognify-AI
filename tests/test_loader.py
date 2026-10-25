from src.document_loader import load_pdf

pages = load_pdf("data/documents/sample.pdf")
print(f"Extracted {len(pages)} pages")

for page in pages:
    print(f"\nSource: {page['source']}")
    print(f"Page: {page['page']}")
    print(page["text"][:300])
