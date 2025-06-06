import os
import gradio as gr
from groq import Groq

# Initialize Groq Client with API key
GROQ_API_KEY = os.getenv("groq_api_key")
client = Groq(api_key=GROQ_API_KEY)

# 1. Tamil Audio to Tamil Text (Whisper)
def transcribe_audio(audio_path):
    if not audio_path:
        return "Please upload an audio file."
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        transcription = client.audio.transcriptions.create(
            file=("audio.m4a", audio_bytes),
            model="whisper-large-v3",
            language="ta",
            response_format="verbose_json"
        )
        return transcription.text
    except Exception as e:
        return f"Error in transcription: {str(e)}"

# 2. Tamil Text to English Text (Gemma)
def translate_tamil_to_english(tamil_text):
    if not tamil_text:
        return "Please enter Tamil text."

    prompt = f"""Translate the below Tamil text to English:\nTamil Text: {tamil_text}\nGive only the translated part as the output without any extra words."""
    try:
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in translation: {str(e)}"

# 3. English Text to English Content (DeepSeek)
def generate_text(english_text):
    if not english_text:
        return "Please enter a prompt."
    try:
        prompt = f"Brief on {english_text}"
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=False,
            stop=None,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in text generation: {str(e)}"

# Pipeline
def process_audio(audio_path):
    # 1. Transcription
    tamil_text = transcribe_audio(audio_path)
    if "Error" in tamil_text:
        return tamil_text, None, None, None

    # 2. Translation
    english_text = translate_tamil_to_english(tamil_text)
    if "Error" in english_text:
        return tamil_text, english_text, None, None

    # 3. Text Generation
    generated_text = generate_text(english_text)
    return tamil_text, english_text, generated_text


# Gradio Interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload Tamil Audio"),
    outputs=[
        gr.Textbox(label="Transcribed Tamil Text"),
        gr.Textbox(label="Translated English Text"),
        gr.Textbox(label="Generated Text from English Prompt")
    ],
    title="TransArt: A Multimodal Application for Vernacular Language Translation and Text Generation",
    description="""Upload a Tamil audio file or live voice record Tamil audio and
    get transcription, translation, and further text generation."""
)

# Launch the Gradio app
iface.launch()
