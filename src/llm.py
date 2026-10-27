
import os

from dotenv import load_dotenv
from google import genai


# Load environment variables from the .env file
load_dotenv()

# Read the Gemini API key securely
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Create the Gemini API client
client = genai.Client(
    api_key=api_key,
    http_options={"timeout": 20000}
)

# Select the Gemini model
MODEL_NAME = "gemini-3.5-flash-lite"


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer using the question and retrieved document context.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context.strip():
        raise ValueError("Context cannot be empty.")

    prompt = f"""
    You are Cognify AI, a document question-answering assistant.

    Answer the user's question using only the provided document context.
    Do not invent facts or introduce unsupported information.
    If the context does not contain enough information, say so clearly.

    Document context:
    {context}

    User question:
    {question}

    Provide a clear and concise answer based on the context.
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()
