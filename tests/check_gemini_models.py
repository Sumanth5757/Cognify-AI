
from src.llm import client

print("Checking Gemini models available to your API key...")

for model in client.models.list():
    actions = model.supported_actions or []

    if "generateContent" in actions:
        print(model.name)
