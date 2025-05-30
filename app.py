import gradio as gr
from transformers import pipeline

# Load Mistral model from Hugging Face
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")

def chat_with_ai(message):
    response = chatbot(message, max_new_tokens=150, do_sample=True)[0]["generated_text"]
    return response

# Gradio interface
iface = gr.Interface(
    fn=chat_with_ai,
    inputs="text",
    outputs="text",
    title="Meta AI Chatbot",
    description="A free Meta AI-style chatbot using open-source LLM."
)

iface.launch()
