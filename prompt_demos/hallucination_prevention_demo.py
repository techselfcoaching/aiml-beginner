"""
hallucination_prevention_demo.py

This script demonstrates both BAD and GOOD approaches
to prevent hallucinations when using OpenAI models.
"""

from openai import OpenAI
import re

client = OpenAI()

# -------------------------------
# BAD EXAMPLE: Hallucination Risk
# -------------------------------
def bad_example():
    print("\n--- BAD EXAMPLE (Hallucination Risk) ---")
    user_question = "What is the capital of Atlantis?"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": user_question}
        ]
    )

    print("Model Answer (likely hallucinated):", response.choices[0].message.content)


# -------------------------------
# GOOD EXAMPLE 1: Grounding
# -------------------------------
def grounding_example():
    print("\n--- GOOD EXAMPLE 1 (Grounding with Knowledge Base) ---")
    knowledge_base = {
        "Paris": "Paris is the capital of France.",
        "Tokyo": "Tokyo is the capital of Japan."
    }

    user_question = "What is the capital of Atlantis?"
    city = user_question.replace("What is the capital of ", "").replace("?", "")

    if city in knowledge_base:
        print("Grounded Answer:", knowledge_base[city])
    else:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "If the answer is unknown, say 'I don’t know'."},
                {"role": "user", "content": user_question}
            ]
        )
        print("Safe Answer:", response.choices[0].message.content)


# -------------------------------
# GOOD EXAMPLE 2: Citations
# -------------------------------
def citation_example():
    print("\n--- GOOD EXAMPLE 2 (Require Citations) ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Always provide a reliable source (URL, book, or paper). If no source, say 'No reliable source available'."},
            {"role": "user", "content": "Who discovered Atlantis?"}
        ]
    )
    print("Answer with Citation:", response.choices[0].message.content)


# -------------------------------
# GOOD EXAMPLE 3: Refusal Policy
# -------------------------------
def refusal_policy_example():
    print("\n--- GOOD EXAMPLE 3 (Refusal Policy) ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You must never make up facts. If you don’t know, answer 'I don’t know'."},
            {"role": "user", "content": "What is the population of Mars in 2025?"}
        ]
    )
    print("Refusal Policy Answer:", response.choices[0].message.content)


# -------------------------------
# GOOD EXAMPLE 4: Validation Layer
# -------------------------------
def validation_example():
    print("\n--- GOOD EXAMPLE 4 (Validation Layer) ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer with a number only if you are certain. If not, respond 'Unknown'."},
            {"role": "user", "content": "How many moons does Earth have?"}
        ]
    )

    answer = response.choices[0].message.content.strip()

    if re.fullmatch(r"\d+", answer):
        print("Validated Answer:", answer)
    else:
        print("Uncertain Answer:", answer)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    bad_example()
    grounding_example()
    citation_example()
    refusal_policy_example()
    validation_example()
