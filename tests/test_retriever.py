from src.llm import generate_answer
from src.document_loader import load_pdf
from src.text_chunker import chunk_pages
from src.embeddings import generate_embeddings
from src.vector_store import create_faiss_index, search_faiss
from src.answer_verifier import verify_answer
from src.evidence_checker import check_evidence


def get_source_citations(results: list[dict]) -> list[dict]:
    """
    Extract unique source and page references from retrieved chunks.
    """

    citations = []

    seen = set()

    for result in results:
        source_page = (result["source"], result["page"])

        if source_page not in seen:
            citations.append({
                "source": result["source"],
                "page": result["page"]
            })

            seen.add(source_page)

    return citations

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

# Retrieve the three most relevant chunks
results = search_faiss(
    question=question,
    index=index,
    chunks=chunks,
    top_k=3
)

# Extract unique source citations
citations = get_source_citations(results)

print("\n--- Source Citations ---")

for citation in citations:
    print(f"Source: {citation['source']} | Page: {citation['page']}")

# Prepare context from the retrieved chunks
context = "\n\n".join(
    f"Source: {result['source']}\n"
    f"Page: {result['page']}\n"
    f"Content: {result['text']}"
    for result in results
)

# Check whether the retrieved context contains sufficient evidence
evidence_status = check_evidence(
    question=question,
    context=context
)

print("\n--- Evidence Check ---")
print(evidence_status)

# Generate and verify an answer only if evidence is sufficient
if evidence_status == "ANSWERABLE":

    answer = generate_answer(
        question=question,
        context=context
    )

    verification_result = verify_answer(
        answer=answer,
        context=context
    )

else:
    answer = (
        "The uploaded document does not contain "
        "sufficient information to answer this question."
    )

    verification_result = "Verification skipped due to insufficient evidence."

# Display the results
print("\nQuestion:", question)
print("\nRetrieved chunks:", len(results))

for rank, result in enumerate(results, start=1):
    print(f"\n--- Result {rank} ---")
    print("Similarity:", round(result["similarity"], 4))
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Text:", result["text"][:500])

print("\n--- Gemini Generated Answer ---")
print(answer)

print("\n--- Answer Verification Result ---")
print(verification_result)

print("\n--- Supporting Sources ---")

if evidence_status == "ANSWERABLE":
    for citation in citations:
        print(f"[{citation['source']}, Page {citation['page']}]")
else:
    print("No supporting citations available.")