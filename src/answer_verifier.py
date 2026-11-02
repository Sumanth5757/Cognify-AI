from src.llm import client, MODEL_NAME


def verify_answer(answer: str, context: str) -> str:
    """
    Verify whether the generated answer is supported by the document context.
    """

    if not answer.strip():
        raise ValueError("Answer cannot be empty.")

    if not context.strip():
        raise ValueError("Context cannot be empty.")

    prompt = f"""
    You are an answer verification system for Cognify AI.

    Your task is to check whether the generated answer is supported
    by the provided document context.

    Follow these rules:
    1. Check every factual claim in the answer.
    2. Identify claims that are supported by the context.
    3. Identify claims that are not supported by the context.
    4. Do not use external knowledge.
    5. If the context is insufficient, clearly mention it.

    Document Context:
    {context}

    Generated Answer:
    {answer}

    Return your evaluation with:
    - Overall verdict: SUPPORTED, PARTIALLY SUPPORTED, or UNSUPPORTED.
    - Supported claims.
    - Unsupported claims.
    - Explanation.
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty verification response.")

    return response.text.strip()