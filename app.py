import os
import gradio as gr
import requests
import io
from PIL import Image
from groq import Groq


# Set your API keys
GROQ_API_KEY = os.getenv("groq_api")

# Initialize Groq API client
client = Groq(api_key=GROQ_API_KEY)

# Hugging Face API for Image Generation
HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"



# Function 1: Tamil Audio to Tamil Text (Transcription)
def transcribe_audio(audio_path):
    if not audio_path:
        return "Please upload an audio file."

    try:
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model="whisper-large-v3",
                language="ta",  # Tamil
                response_format="verbose_json",
            )
        return transcription.text
    except Exception as e:
        return f"Error in transcription: {str(e)}"



# Function 2: Tamil Text to English Translation
def translate_tamil_to_english(tamil_text):
    if not tamil_text:
        return "Please enter Tamil text for translation."

    prompt = f"Translate the below Tamil text to English:\nTamil Text: {tamil_text}\nGive only the translated part as the output without any extra words."

    try:
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error in translation: {str(e)}"



# Function 3: English Text to Image Generation
def generate_image(english_text):
    if not english_text:
        return "Please enter a description for image generation."

    try:
        payload = {"inputs": english_text}
        response = requests.post(f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}", json=payload)
        response.raise_for_status()
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
        return image

    except Exception as e:
        return f"Error in image generation: {str(e)}"



# Function 4: English Text to Further Text Generation
def generate_text(english_text):
    if not english_text:
        return "Please enter a prompt."

    try:
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": english_text}],
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error in text generation: {str(e)}"



# Combined Function to Process All Steps Sequentially
def process_audio(audio_path):
    # Step 1: Tamil Audio → Tamil Text
    tamil_text = transcribe_audio(audio_path)
    if "Error" in tamil_text:
        return tamil_text, None, None, None

    # Step 2: Tamil Text → English Text
    english_text = translate_tamil_to_english(tamil_text)
    if "Error" in english_text:
        return tamil_text, english_text, None, None

    # Step 3: English Text → Image
    image = generate_image(english_text)
    if "Error" in str(image):
        return tamil_text, english_text, None, None

    # Step 4: English Text → Generated Text
    generated_text = generate_text(english_text)
    return tamil_text, english_text, image, generated_text



# Create Gradio Interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload Tamil Audio"),
    outputs=[
        gr.Textbox(label="Transcribed Tamil Text"),
        gr.Textbox(label="Translated English Text"),
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Generated Text from English Prompt"),
    ],
    title="Tamil Audio to AI Processing Pipeline",
    description="Upload a Tamil audio file and get transcription, translation, image generation, and further text generation.",
)



# Launch the Gradio app
iface.launch()
