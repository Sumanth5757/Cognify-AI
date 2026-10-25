def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[dict]:
    """
    Split extracted PDF pages into smaller text chunks.

    Each chunk preserves its source filename and page number.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"],
                    "source": page["source"]
                })

            start += chunk_size - chunk_overlap

    return chunks
