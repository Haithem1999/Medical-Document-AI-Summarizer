import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
from PIL import Image
import io
import fitz  # PyMuPDF
import os
import base64

# Configure OpenAI API
api_key = st.secrets["OPENAI_API_KEY"]

def extract_text_from_pdf(pdf_file):
    # Create PDF reader object
    pdf_reader = PdfReader(pdf_file)
    text = ""
    
    # Extract text from each page
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    # Convert PDF page to image to handle handwritten content
    # (Note: we do it once here rather than inside the loop of PyPDF2)
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
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an assistant that transcribes text from images",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        ),

        # If response is a tuple, unpack it
        if isinstance(response, tuple):
            response = response[0]

        # Now access choices
        content = response.choices[0].message.content
        text += content + "\n"
    
    return text

def summarize_text(text):
    try:
        client = OpenAI(api_key=api_key)
        response_ = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical expert. Summarize the following medical document, highlighting key diagnoses, treatments, and important medical findings."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3
        )
        return response_.choices[0].message.content
    except Exception as e:
        return f"Error in summarization: {str(e)}"

# ----------------------
# Session State Setup
# ----------------------
# 1) Track the name (or another unique identifier) of the last processed file.
# 2) Track the summary text.
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None

if "summary" not in st.session_state:
    st.session_state.summary = None

# ----------------------
# Streamlit UI
# ----------------------
st.title("Medical Document Summarizer")
st.write("Upload a medical PDF document to get an AI-powered summary")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# If a file is uploaded, check if it's new or the same file.
if uploaded_file is not None:
    # If the uploaded file name is different from the last processed file name,
    # we know it's a new file => we must re-process it.
    if uploaded_file.name != st.session_state.processed_file_name:
        st.session_state.summary = None  # clear any old summary
        st.session_state.processed_file_name = uploaded_file.name

    # If there's no summary for this file in session state, process it now.
    if st.session_state.summary is None:
        with st.spinner('Processing document...'):
            text = extract_text_from_pdf(uploaded_file)
            summary = summarize_text(text)
            st.session_state.summary = summary

# If we have a summary in session state, display it and allow downloading
if st.session_state.summary:
    st.subheader("Document Summary")
    st.write(st.session_state.summary)

    st.download_button(
        label="Download Summary",
        data=st.session_state.summary,
        file_name="summary.txt",
        mime="text/plain"
    )
