import os
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

    prompt = f"""Translate the below Tamil text to English:\n
    Tamil Text: {tamil_text}\n
    Give only the translated part as the output without any extra words."""
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
        response = requests.post(f"https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell", json=payload)
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

    prompt = f"Give me a brief paragraph on the topic '{english_text}'"
    try:
        response = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
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
    title="TransArt: A Multimodal Application for Vernacular Language Translation and Image Synthesis",
    description="""Upload a Tamil audio file or live voice record Tamil audio and
    get transcription, translation, image generation, and further text generation."""
)

# Launch the Gradio app
iface.launch()
