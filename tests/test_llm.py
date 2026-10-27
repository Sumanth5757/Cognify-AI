
from src.llm import generate_answer


question = "How does energy efficiency help the environment?"

context = """
Improving energy efficiency reduces the amount of energy needed.
This can reduce the amount of fossil fuels burned and greenhouse
gases emitted.
"""

answer = generate_answer(
    question=question,
    context=context
)

print("\nQuestion:", question)
print("\nGenerated answer:")
print(answer)
