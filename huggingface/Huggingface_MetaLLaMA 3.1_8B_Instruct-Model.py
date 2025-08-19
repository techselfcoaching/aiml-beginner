import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Translate 'Good morning, how are you?' into Spanish"
        }
    ],
)

print(completion.choices[0].message)