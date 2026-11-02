from src.evidence_checker import check_evidence


context = """
Improving energy efficiency reduces the amount of energy needed.
This reduces the amount of fossil fuels burned and greenhouse gases emitted.
"""


# Test 1: Answerable question
question1 = "How does energy efficiency help the environment?"

result1 = check_evidence(
    question=question1,
    context=context
)

print("\n--- Test 1: Answerable Question ---")
print("Question:", question1)
print("Evidence Check:", result1)


# Test 2: Insufficient evidence
question2 = "What was the GDP of India in 2025?"

result2 = check_evidence(
    question=question2,
    context=context
)

print("\n--- Test 2: Insufficient Evidence ---")
print("Question:", question2)
print("Evidence Check:", result2)