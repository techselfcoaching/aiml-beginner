"""
Prompt Engineering Demo: Bad vs Good Examples
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

# BAD PROMPT - Vague and lacks context
print("❌ BAD PROMPT:")
bad_prompt = "Plan a trip for a family."
print(f'"{bad_prompt}"')
print("\nIssues: Too vague, no context, no constraints")
print("\nResponse:")
print(get_response(bad_prompt))

print("\n" + "="*50 + "\n")

# GOOD PROMPT - Specific, detailed, structured
print("✅ GOOD PROMPT:")
good_prompt = """You are a travel planner. Help plan a weekend trip for:
- Family of 4 (2 adults, 2 kids aged 8 & 12)
- From NYC, max 3 hours drive
- Budget: $1,200
- Loves outdoor activities
- Needs family-friendly options

Provide exactly 3 destinations with:
1. Name and brief description
2. Top 2 activities for families
3. Accommodation suggestion
4. Estimated cost"""

print(f'"{good_prompt}"')
print("\nStrengths: Clear context, specific constraints, structured output")
print("\nResponse:")
print(get_response(good_prompt))

print("\n" + "="*50)
print("KEY TIPS:")
print("• Be specific about context and constraints")
print("• Ask for structured output")
print("• Include all relevant details")
print("• Test and refine your prompts")