from src.llm import client, MODEL_NAME


def check_evidence(question: str, context: str) -> str:
    """
    Check whether the retrieved document context contains
    sufficient evidence to answer the user's question.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context.strip():
        raise ValueError("Context cannot be empty.")

    prompt = f"""
    You are an evidence sufficiency checker for Cognify AI.

    Your task is to determine whether the provided document
    context contains enough relevant information to answer
    the user's question.

    Follow these rules:
    1. Evaluate the question against the provided context.
    2. Do not use external knowledge.
    3. Do not assume information that is not present.
    4. If the context directly or sufficiently answers the
       question, return ANSWERABLE.
    5. If the context does not contain enough information,
       return INSUFFICIENT.

    User Question:
    {question}

    Document Context:
    {context}

    Return exactly one of these labels:
    ANSWERABLE
    INSUFFICIENT
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty evidence check.")

    verdict = response.text.strip().upper()

    if verdict not in {"ANSWERABLE", "INSUFFICIENT"}:
        raise RuntimeError(
            f"Unexpected evidence checker response: {response.text}"
        )

    return verdict