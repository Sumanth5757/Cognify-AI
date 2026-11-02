from src.document_loader import load_pdf
from src.text_chunker import chunk_pages
from src.embeddings import generate_embeddings
from src.vector_store import create_faiss_index, search_faiss
from src.rag_pipeline import ask_question


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
question = "What was the GDP of India in 2025?"

# Execute the RAG pipeline
result = ask_question(
    question=question,
    index=index,
    chunks=chunks
)

# Display the results
print("\nQuestion:", result["question"])

print("\nRetrieved chunks:", len(result["retrieved_chunks"]))

print("\n--- Evidence Check ---")
print(result["evidence_status"])

print("\n--- Gemini Generated Answer ---")
print(result["answer"])

print("\n--- Answer Verification ---")
print(result["verification"])

if result["citations"]:
    for citation in result["citations"]:
        print(
            f"[{citation['source']}, "
            f"Page {citation['page']}]"
        )
else:
    print("No supporting citations available.")