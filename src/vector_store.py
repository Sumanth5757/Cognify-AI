import faiss
import numpy as np

def create_faiss_index(embeddings: np.ndarray):
    """
    Create a FAISS index using normalized embeddings.

    Normalized vectors allow inner-product search
    to behave like cosine similarity search.
    """

    if embeddings.size == 0:
        raise ValueError("Embeddings cannot be empty.")

    # FAISS expects 32-bit floating-point vectors
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # Get the embedding dimension
    dimension = embeddings.shape[1]

    # Create an exact inner-product index
    index = faiss.IndexFlatIP(dimension)

    # Add embeddings to the index
    index.add(embeddings)

    return index


import faiss
import numpy as np

from src.embeddings import generate_embeddings


def create_faiss_index(embeddings: np.ndarray):
    """
    Create a FAISS index using normalized embeddings.
    """

    if embeddings.size == 0:
        raise ValueError("Embeddings cannot be empty.")

    # Convert embeddings to float32 and make a copy
    embeddings = np.array(embeddings, dtype=np.float32, copy=True)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # Get the dimension of each vector
    dimension = embeddings.shape[1]

    # Create an exact inner-product index
    index = faiss.IndexFlatIP(dimension)

    # Add document vectors to the index
    index.add(embeddings)

    return index


def search_faiss(
    question: str,
    index,
    chunks: list[dict],
    top_k: int = 3
) -> list[dict]:
    """
    Search the FAISS index and return the most relevant chunks.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if index.ntotal != len(chunks):
        raise ValueError("The index and chunks must have matching lengths.")

    if index.ntotal == 0:
        return []

    # Convert the user's question into an embedding
    question_embedding = generate_embeddings([question])

    # FAISS expects float32 vectors
    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32
    )

    # Normalize the question vector
    faiss.normalize_L2(question_embedding)

    # Limit the number of results to available vectors
    top_k = min(top_k, index.ntotal)

    # Search the index
    scores, indices = index.search(question_embedding, top_k)

    results = []

    # Map each returned vector ID to its original chunk
    for score, chunk_id in zip(scores[0], indices[0]):

        # FAISS uses -1 when no matching vector is available
        if chunk_id == -1:
            continue

        result = chunks[int(chunk_id)].copy()

        # Add the similarity score to the returned chunk
        result["similarity"] = float(score)

        results.append(result)

    return results
