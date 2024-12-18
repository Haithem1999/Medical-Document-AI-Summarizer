import streamlit as st
from PyPDF2 import PdfReader
import base64
from PIL import Image
import io
from openai import OpenAI
import tempfile

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("Medical Document AI Summarizer")
st.write("Upload your medical documents (PDF) for AI-powered summarization")

def extract_images_from_pdf(pdf_path):
    """Extract images from PDF pages for OCR processing"""
    images = []
    pdf = PdfReader(pdf_path)
    
    for page in pdf.pages:
        if '/XObject' in page['/Resources']:
            xObject = page['/Resources']['/XObject'].get_object()
            
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    data = xObject[obj].get_data()
                    try:
                        img = Image.open(io.BytesIO(data))
                        images.append(img)
                    except:
                        continue
    return images

def encode_image(image):
    """Encode image to base64 for OpenAI Vision API"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_document(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file.getvalue())
        pdf_path = tmp_file.name
        
    # Extract text
    pdf = PdfReader(pdf_path)
    text_content = ""
    for page in pdf.pages:
        text_content += page.extract_text() + "\n"
    
    # Extract and process images for handwritten content
    images = extract_images_from_pdf(pdf_path)
    
    # Process text content
    text_summary = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical document summarizer. Provide a clear, concise summary of the medical document."},
            {"role": "user", "content": f"Summarize this medical document:\n\n{text_content[:15000]}"}  # Limit text to avoid token limits
        ]
    )
    
    # Process images for handwritten content
    handwritten_content = []
    for img in images:
        base64_image = encode_image(img)
        
        vision_response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read and transcribe any handwritten medical content in this image. If there's no handwritten content, respond with 'No handwritten content detected'."},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{base64_image}"
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        handwritten_content.append(vision_response.choices[0].message.content)
    
    return text_summary.choices[0].message.content, handwritten_content

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner('Processing document... This may take a few minutes for large files.'):
        try:
            text_summary, handwritten_content = process_document(uploaded_file)
            
            st.subheader("Document Summary")
            st.write(text_summary)
            
            if any(content != "No handwritten content detected" for content in handwritten_content):
                st.subheader("Detected Handwritten Content")
                for content in handwritten_content:
                    if content != "No handwritten content detected":
                        st.write(content)
                        
        except Exception as e:
            st.error(f"An error occurred while processing the document: {str(e)}")
