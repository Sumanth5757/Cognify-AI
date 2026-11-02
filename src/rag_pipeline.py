from src.vector_store import search_faiss
from src.evidence_checker import check_evidence
from src.llm import generate_answer
from src.answer_verifier import verify_answer

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


def ask_question(
    question: str,
    index,
    chunks: list[dict],
    top_k: int = 3
):
    """
    Execute the RAG question-answering pipeline.
    """

    # Retrieve relevant document chunks
    results = search_faiss(
        question=question,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    # Prepare context from retrieved chunks
    context = "\n\n".join(
        f"Source: {result['source']}\n"
        f"Page: {result['page']}\n"
        f"Content: {result['text']}"
        for result in results
    )

    # Check whether the retrieved context contains sufficient evidence
    if not results:
        evidence_status = "INSUFFICIENT"
    else:
        evidence_status = check_evidence(
            question=question,
            context=context
        )

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

        verification_result = (
            "Verification skipped due to insufficient evidence."
        )

    # Generate citations only when evidence is sufficient
    if evidence_status == "ANSWERABLE":
        citations = get_source_citations(results)
    else:
        citations = []

    return {
        "question": question,
        "retrieved_chunks": results,
        "context": context,
        "evidence_status": evidence_status,
        "answer": answer,
        "verification": verification_result,
        "citations": citations
    }