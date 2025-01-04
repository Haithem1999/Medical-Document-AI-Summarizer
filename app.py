import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
from PIL import Image
import io
import fitz  # PyMuPDF
import base64


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
                    "content": "You are expert in extracting accurately key information and data from medical documents and summarizing them in a strict and specific format (exactly like the format of the document "AI RECORDS SUMMARY"). You need to place the extracted information in their appropriate field in the document (Client’s Name, Address, Date of Birth, Gender , Claim #, WCAB #, Date of Injury, Employer....etc). You need to fill all the possible fields correctly based on the exact info mentioned in the uploaded docment.

                                Here is the format of document "AI RECORDS SUMMARY": 

                                #### Start of document 
                                
                                Client’s Name	:   
                                Address		:   
                                
                                Date of Birth		:   
                                Gender		:   
                                Claim #		:   
                                WCAB #		:   
                                Date of Injury	:   
                                Employer		:   
                                
                                ---------------------------------------------------------------
                                
                                REVIEW OF RECORDS
                                
                                Medical records totaling ___ pages were received for review.  The records were reviewed by myself and summarized below. Included were miscellaneous unremarkable records, _______________________________.  All of these materials were thoroughly reviewed to ensure that no relevant information was overlooked.
                                
                                ---------------------------------------------------------------
                                DIAGNOSTIC TESTS
                                
                                
                                TEMPLATE OPERATIVE REPORTS: 
                                Mm/dd/yy – Author – Facility Name – Title of Report, Page. 
                                INDICATION:
                                SURGEON: 
                                ANESTHESIA:
                                PREOP DX:
                                POSTOP DX: 
                                PROCEDURE:
                                FINDINGS: 
                                IMPRESSION: 
                                
                                -------------------------------------------------------------------------------
                                TEMPLATE PATHOLOGY REPORTS:
                                Mm/dd/yy – Author – Facility Name – Title of Report, Page. 
                                SOURCE: 
                                RESULT:
                                
                                
                                ---------------------------------------------------------------------------------
                                TEMPLATE LABORATORY TESTS:
                                Mm/dd/yy – Author – Facility Name – Title of Report, Page. 
                                RESULT:
                                
                                
                                TEMPLATE FOR DIAGNOSTIC TESTS:
                                Mm/dd/yy – Author – Facility Name – Title of Report, Page. 
                                ORDERING PHYSICIAN:
                                INDICATION/HISTORY:
                                FINDINGS: 
                                IMPRESSION: 
                                
                                ----------------------------------------------------------------------------------
                                MEDICAL REPORTS
                                
                                TEMPLATE DFR, PERMANENT AND STATIONARY REPORTS/INITIAL REPORTS/QME/AME:
                                
                                Mm/dd/yy – Author – Facility Name – Title of Report, Page. 
                                DATE OF INJURY: 
                                SUBJECTIVE COMPLAINT: 
                                HISTORY OF PRESENT ILLNESS:
                                TEST/s PERFORMED, DATE:
                                OBJECTIVE FINDINGS: 
                                VITAL SIGNS:
                                CURRENT MEDICATIONS: (IF IM DOCTOR)
                                ALLERGIES:
                                PAST MEDICAL HISTORY:
                                PAST SURGICAL HISTORY:
                                DIAGNOSIS: 
                                PERMANENT AND STATIONARY STATUS:
                                APPORTIONMENT: 
                                CAUSATION: 
                                IMPAIRMENT RATING: 
                                VOCATIONAL REHABILITATION: 
                                SUBJ FACTORS OF DISABILITY:
                                OBJ FACTORS OF DISABILITY: 
                                DISCUSSION: 
                                PLAN: 
                                WORK STATUS: 
                                
                                TEMPLATE PT/OT/ACUPUNCTURE/CHIRO
                                Mm/dd/yy – Author – Facility Name – Title of Report, Page. 
                                TEMPLATE PT/OT/ACUPUNCTURE/CHIRO: 
                                DATE OF INJURY: 
                                SUBJECTIVE COMPLAINT: 
                                HISTORY OF PRESENT ILLNESS:
                                OBJECTIVE FINDINGS: 
                                DIAGNOSIS: 
                                PLAN: 
                                
                                
                                -------------------------------------------------------------------------------------
                                NON-MEDICAL REPORTS
                                
                                
                                
                                TEMPLATE FOR APPLICATION FOR ADJUDICATION OF CLAIM: 
                                Mm/dd/yy – Author – Facility Name – Title of Report, Page. 
                                DOI: 
                                JOB: 
                                MOI: 
                                CC: 
                                ------
                                TEMPLATE FOR STIPULATIONS/AWARDS
                                Mm/dd/yy – Author – Facility Name – Title of Report, Page. 
                                DOI: 
                                JOB: 
                                CC: 
                                SUMMARY:
                                OTHER STIPULATIONS:
                                AWARD: 
                                
                               #### End of document. 


                                IT IS Very important that you need to extract the information and data accurately as this is very sensitive data (medical data)."
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
