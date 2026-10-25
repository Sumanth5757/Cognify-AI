
from sentence_transformers import SentenceTransformer
# Load the pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts: list[str]):
    """
    Convert a list of text strings into embedding vectors.
    """

    if not texts:
        return []

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings
