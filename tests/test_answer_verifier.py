from src.answer_verifier import verify_answer

context = """
Improving energy efficiency reduces the amount of energy needed.
This reduces the amount of fossil fuels burned and greenhouse gases emitted.
"""


# Test 1: Supported answer
supported_answer = """
Energy efficiency reduces fossil fuel consumption
and greenhouse gas emissions.
"""

print("\n========== TEST 1: SUPPORTED ANSWER ==========")

result1 = verify_answer(
    answer=supported_answer,
    context=context
)

print(result1)


# Test 2: Unsupported answer
unsupported_answer = """
Energy efficiency reduces greenhouse gas emissions by exactly 40%.
It also increases global electricity production by 25%.
"""

print("\n========== TEST 2: UNSUPPORTED ANSWER ==========")

result2 = verify_answer(
    answer=unsupported_answer,
    context=context
)

print(result2)

# Test 3: Partially supported answer
partial_answer = """
Energy efficiency reduces fossil fuel consumption.
It reduces greenhouse gas emissions by exactly 50%.
"""

print("\n========== TEST 3: PARTIALLY SUPPORTED ANSWER ==========")

result3 = verify_answer(
    answer=partial_answer,
    context=context
)

print(result3)