import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
from PIL import Image
import io
import fitz  # PyMuPDF
import base64


# Add custom CSS to hide the GitHub icon
hide_github_icon = """
#GithubIcon {
  visibility: hidden;
}
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

# Configure OpenAI API
api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_data
def extract_text_from_pdf(pdf_bytes):
    # Create PDF reader object
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    
    # Extract text from each page
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
        
        # Convert PDF page to image to handle handwritten content
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
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
            )
            
            # If response is a tuple, unpack it
            if isinstance(response, tuple):
                response = response[0]
            
            # Now access choices
            content = response.choices[0].message.content
            handwritten_text = content
            text += handwritten_text + "\n"
    
    return text

@st.cache_data
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

# Streamlit UI
st.title("Medical Document Summarizer")
st.write("Upload a medical PDF document to get an AI-powered summary")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()  # Read the PDF file as bytes
    
    with st.spinner('Processing document...'):
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_bytes)
        
        # Get summary
        summary = summarize_text(text)
        
        # Display summary
        st.subheader("Document Summary")
        st.write(summary)
        
        # Provide a download button for the summary
        st.download_button(
            label="Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )
