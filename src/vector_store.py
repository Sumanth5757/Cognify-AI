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