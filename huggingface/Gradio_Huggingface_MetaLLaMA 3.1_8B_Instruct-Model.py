import os
import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the client
client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.environ["HF_TOKEN"],
)


def get_llama_response(message, history):
    """
    Get response from Llama model and update chat history
    """
    try:
        # Prepare messages for the API (include conversation history)
        messages = []

        # Add conversation history
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        # Add current user message
        messages.append({"role": "user", "content": message})

        # Get completion from Llama
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )

        response = completion.choices[0].message.content

        # Update history
        history.append([message, response])

        return history, history, ""

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append([message, error_msg])
        return history, history, ""


def clear_chat():
    """
    Clear the chat history
    """
    return [], [], ""


# Create Gradio interface
with gr.Blocks(title="Llama 3.1 Chat Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦙 Llama 3.1 Chat Assistant")
    gr.Markdown("Chat with Meta's Llama 3.1 8B Instruct model")

    with gr.Row():
        with gr.Column(scale=2):
            # Main chat interface
            chatbot = gr.Chatbot(
                value=[],
                label="Chat",
                height=400,
                show_label=False,
                container=True,
                bubble_full_width=False
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your message here...",
                    label="Message",
                    show_label=False,
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            clear_btn = gr.Button("Clear Chat", variant="secondary")

        with gr.Column(scale=1):
            # Chat history sidebar
            gr.Markdown("### 📜 Chat History")
            history_display = gr.Chatbot(
                value=[],
                label="History",
                height=400,
                show_label=False,
                container=True,
                show_copy_button=False,
                bubble_full_width=False
            )

    # Store chat history
    chat_history = gr.State([])


    # Event handlers
    def handle_send(message, history):
        if message.strip():
            return get_llama_response(message, history)
        return history, history, message


    # Send message on button click
    send_btn.click(
        fn=handle_send,
        inputs=[msg_input, chat_history],
        outputs=[chatbot, history_display, msg_input]
    )

    # Send message on Enter key
    msg_input.submit(
        fn=handle_send,
        inputs=[msg_input, chat_history],
        outputs=[chatbot, history_display, msg_input]
    )

    # Clear chat
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, history_display, chat_history]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Set to False if you don't want a public link
        show_error=True
    )