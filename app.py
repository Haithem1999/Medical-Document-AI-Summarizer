import streamlit as st
from PyPDF2 import PdfReader
import openai
from PIL import Image
import io
import fitz  # PyMuPDF
import os
import base64

# Configure OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_text_from_pdf(pdf_file):
    # Create PDF reader object
    pdf_reader = PdfReader(pdf_file)
    text = ""
    
    # Extract text from each page
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
        
        # Convert PDF page to image to handle handwritten content
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert image to base64 for OpenAI Vision API
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
            
            # Extract text from image using OpenAI
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that transcribes text from images."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please transcribe the text from this image."},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,", {base64_image}}
                        ]
                    }
                ],
            )
            handwritten_text = response.choices[0].message.content
            text += handwritten_text + "\n"
    
    return text

def summarize_text(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical expert. Summarize the following medical document, highlighting key diagnoses, treatments, and important medical findings."},
                {"role": "user", "content": text}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in summarization: {str(e)}"

# Streamlit UI
st.title("Medical Document Summarizer")
st.write("Upload a medical PDF document to get an AI-powered summary")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner('Processing document...'):
        # Extract text from PDF
        text = extract_text_from_pdf(uploaded_file)
        
        # Get summary
        summary = summarize_text(text)
        
        # Display summary
        st.subheader("Document Summary")
        st.write(summary)


