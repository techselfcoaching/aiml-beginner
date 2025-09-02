"""
Multi-Shot Prompting Demo: Zero-Shot vs Few-Shot Examples
"""

from openai import OpenAI

client = OpenAI()


def get_response(prompt):
    """Get AI response with error handling"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Poor prompt
# ZERO-SHOT (No Examples) - Less reliable
print("❌ ZERO-SHOT PROMPTING:")
zero_shot_prompt = """Classify the sentiment of this text as Positive, Negative, or Neutral:

"The delivery was delayed but the customer service team handled it well."
Sentiment: ?"""

print(f'"{zero_shot_prompt}"')
print("\nIssue: No examples provided, AI might be inconsistent or unclear")
print("\nResponse:")
print(get_response(zero_shot_prompt))

print("\n" + "=" * 60 + "\n")

# Good prompt
# FEW-SHOT (Multiple Examples) - More reliable and consistent
print("✅ FEW-SHOT PROMPTING:")
few_shot_prompt = """Classify the sentiment of these texts as Positive, Negative, or Neutral:

Text: "I absolutely love this product! It works perfectly."
Sentiment: Positive

Text: "This is the worst purchase I've ever made."
Sentiment: Negative

Text: "The product arrived on time and matches the description."
Sentiment: Neutral

Text: "The delivery was delayed but the customer service team handled it well."
Sentiment: ?"""

print(f'"{few_shot_prompt}"')
print("\nStrengths: Clear examples show desired format and reasoning")
print("\nResponse:")
print(get_response(few_shot_prompt))

print("\n" + "=" * 60 + "\n")

# Test multiple examples to show consistency
print("🔄 TESTING CONSISTENCY:")
test_cases = [
    "The food was okay, nothing special.",
    "Amazing quality and fast shipping!",
    "Product broke after one day. Terrible!",
    "Standard packaging, received as expected."
]

for i, test_text in enumerate(test_cases, 1):
    test_prompt = f"""Classify sentiment as Positive, Negative, or Neutral:

Examples:
"I love it!" → Positive
"It's terrible!" → Negative  
"It's fine." → Neutral

Text: "{test_text}"
Sentiment: ?"""

    print(f"Test {i}: {test_text}")
    print(f"Result: {get_response(test_prompt)}\n")

print("=" * 60)
print("KEY BENEFITS OF FEW-SHOT PROMPTING:")
print("• More consistent and predictable results")
print("• Shows desired output format clearly")
print("• Reduces ambiguity in complex tasks")
print("• Better performance on nuanced classifications")