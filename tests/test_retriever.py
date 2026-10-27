
from src.document_loader import load_pdf
from src.text_chunker import chunk_pages
from src.embeddings import generate_embeddings
from src.vector_store import create_faiss_index, search_faiss


# Load the PDF
pages = load_pdf("data/documents/sample.pdf")

# Split the pages into chunks
chunks = chunk_pages(pages)

# Generate embeddings for all document chunks
chunk_texts = [chunk["text"] for chunk in chunks]
embeddings = generate_embeddings(chunk_texts)

# Create the FAISS index
index = create_faiss_index(embeddings)

# Ask a test question
question = "What is the main topic discussed in this document?"

# Retrieve the three most relevant chunks
results = search_faiss(
    question=question,
    index=index,
    chunks=chunks,
    top_k=3
)

# Display the results
print("\nQuestion:", question)
print("\nRetrieved chunks:", len(results))

for rank, result in enumerate(results, start=1):
    print(f"\n--- Result {rank} ---")
    print("Similarity:", round(result["similarity"], 4))
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Text:", result["text"][:500])

