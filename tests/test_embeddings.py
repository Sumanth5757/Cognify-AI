
from src.document_loader import load_pdf
from src.text_chunker import chunk_pages
from src.embeddings import generate_embeddings

# Load the PDF
pages = load_pdf("data/documents/sample.pdf")

# Split the pages into chunks
chunks = chunk_pages(pages)

# Extract the text from each chunk
chunk_texts = [chunk["text"] for chunk in chunks]

# Generate embeddings
embeddings = generate_embeddings(chunk_texts)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

# Display the first embedding
print("\nFirst embedding:")
print(embeddings[0])

# Display the embedding dimension
print("\nVector dimensions:", len(embeddings[0]))
