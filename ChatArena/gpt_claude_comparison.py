from openai import OpenAI
import gradio as gr
import requests

# ====== Set your API keys here ======
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
CLAUDE_API_KEY = "YOUR_CLAUDE_API_KEY"

# ====== Conversation histories ======
conversation_history_gpt = []
conversation_history_claude = []

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ====== GPT Chat function ======
def chat_with_gpt(user_input):
    global conversation_history_gpt
    conversation_history_gpt.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history_gpt
    )
    assistant_message = response.choices[0].message.content
    conversation_history_gpt.append({"role": "assistant", "content": assistant_message})

    # Prepare chat display
    chat_display = ""
    for msg in conversation_history_gpt:
        role = msg['role'].capitalize()
        chat_display += f"{role}: {msg['content']}\n\n"
    return chat_display

# ====== Claude Chat function ======
def chat_with_claude(user_input):
    global conversation_history_claude

    # Convert conversation into Anthropic’s text prompt format
    conversation_text = ""
    for msg in conversation_history_claude:
        role = "Human" if msg['role'] == "user" else "Assistant"
        conversation_text += f"{role}: {msg['content']}\n"
    conversation_text += f"Human: {user_input}\nAssistant:"

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": "claude-2.1",   # or latest model you have access to
        "prompt": conversation_text,
        "max_tokens_to_sample": 500,
        "temperature": 0.7
    }
    response = requests.post("https://api.anthropic.com/v1/complete", headers=headers, json=data)
    response_json = response.json()
    assistant_message = response_json["completion"].strip()

    conversation_history_claude.append({"role": "user", "content": user_input})
    conversation_history_claude.append({"role": "assistant", "content": assistant_message})

    # Prepare chat display
    chat_display = ""
    for msg in conversation_history_claude:
        role = msg['role'].capitalize()
        chat_display += f"{role}: {msg['content']}\n\n"
    return chat_display

# ====== Clear history functions ======
def clear_history_gpt():
    global conversation_history_gpt
    conversation_history_gpt = []
    return ""

def clear_history_claude():
    global conversation_history_claude
    conversation_history_claude = []
    return ""

# ====== Gradio UI ======
with gr.Blocks() as demo:
    gr.Markdown("## Chat with GPT and Claude Side by Side")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### GPT Chat")
            gpt_input = gr.Textbox(lines=2, placeholder="Type your message for GPT...")
            gpt_output = gr.Textbox(label="GPT Chat History")
            gpt_send = gr.Button("Send")
            gpt_clear = gr.Button("Clear History")
        with gr.Column():
            gr.Markdown("### Claude Chat")
            claude_input = gr.Textbox(lines=2, placeholder="Type your message for Claude...")
            claude_output = gr.Textbox(label="Claude Chat History")
            claude_send = gr.Button("Send")
            claude_clear = gr.Button("Clear History")

    gpt_send.click(chat_with_gpt, inputs=gpt_input, outputs=gpt_output)
    gpt_clear.click(clear_history_gpt, outputs=gpt_output)

    claude_send.click(chat_with_claude, inputs=claude_input, outputs=claude_output)
    claude_clear.click(clear_history_claude, outputs=claude_output)

demo.launch()
