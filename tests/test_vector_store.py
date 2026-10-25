from src.document_loader import load_pdf
from src.text_chunker import chunk_pages
from src.embeddings import generate_embeddings
from src.vector_store import create_faiss_index

# Load the PDF
pages = load_pdf("data/documents/sample.pdf")

# Split pages into chunks
chunks = chunk_pages(pages)

# Extract chunk text
chunk_texts = [chunk["text"] for chunk in chunks]

# Generate embeddings
embeddings = generate_embeddings(chunk_texts)

# Create FAISS index
index = create_faiss_index(embeddings)

# Display index information
print("Number of chunks:", len(chunks))
print("Number of vectors in FAISS:", index.ntotal)
print("Embedding dimension:", index.d)

assert index.ntotal == len(chunks)
assert index.d == 384

print("\nFAISS index created successfully!")
