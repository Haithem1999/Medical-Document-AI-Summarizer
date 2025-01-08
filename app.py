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
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """ You are expert in extracting accurately key information and data from medical documents and summarizing them in a strict, specific and chronological format just like the output format of MediScan.ai.  

                                    IT IS Very important that you need to extract the information and data accurately and chronologically as this is very sensitive data (medical data).
                                                                    
                                    After that you need to give a detailed report of all the aspects mentioned in the uploaded documents: treatments, diagnosis, symptoms, results, reports..etc. 
                                    You need to analyze every section of every page in the uploaded document and put it in a detailed report.  
                                                                        
                                    VERY IMPORTANT: In each uploaded document, you do the summarization based on the date and the number of pages related to that date. For example, if a document has 10 pages. The first 3 pages contain medical info for the date: 01 / 15 / 2024. You need to mention the date and the pages related to that date in the document + The title of the section and then give the summary and analysis of these 3 pages for that date and so on and so forth.  
                                    Usually the dates of each section are mentioned in the top left corner of each page. Each section (for example 3 pages) has a unique date. A summarized anlaysis will be unique to each of these dates and number of pages. 
                                    
                                    Here is an example of the output format you should return:
                                    
                                    ---------------------------------------------------------------------------------------------------
                                    08 / 27 / 2024 |  Request for Authorization | 7 pgs 
                                    HPI: Patient working on a conveyor belt fell after losing balance, resulting in injuries. Initial injury on 06/15/2024, first examined on 08/15/2024. CC: Low back pain radiating down the left leg with numbness, mild limp on the left leg, pain with walking more than 1 hour, sitting more than 1 hour, bending, difficulty washing feet and putting on socks and shoes, limited lifting capacity, sleep disruption due to back pain, neck pain with needle-like sensation around the right ear, bilateral hand pain affecting lifting, shoulder pain with more pain on the left, mid to lower rib pain affecting breathing, pain rated 4-8/10. Work History: Patient employed at Grow Smart Labor Inc, working in the Tectrol department. Previous work injury in 2020 involving the left low back, head, and right index finger, but no residual limitations. Exams: Height: 63 3/4 inch. Weight: 181 lbs. BP: 185/121. Pulse: 86 bpm. Mild limp on left lower extremity, mild paracervical muscle tenderness, tenderness in anterior and lateral lower ribs, lumbar spasm left more than right, decreased sensation to pinprick in left lower extremity, weakness in left EHL, dorsiflexion, plantar flexion 4/5. DX: Low back pain, lumbar sprain/strain with left radiculopathy, cervicalgia, right and left shoulder pain with left shoulder impingement, right and left hand pain with loss of left grip strength, ribs pain. TX: Recommended obtaining x-rays of lumbar spine, bilateral hands, ribs, and bilateral shoulders. Recommended 8 sessions of physical therapy for low back and shoulders. Additional options include chiropractic, acupuncture, possible MRI, EMG/NCS of lower extremities, specialty consult.
                                    
                                    -----------------------------------------------------------------------------------------
                                    
                                    09 / 17 / 2024 | Primary Treating Physician's Progress Report (PR-2) | 3 pgs	
                                    Primary Treating Physician's Progress Report (PR-2)
                                    Sovathana Khuong, D.C., QME
                                    CC: Left wrist and hand pain, neck and mid back pain, numbness along the lower ribs bilaterally, pain and numbness in low back extending into left leg, intermittent neck pain with limited motion, pain and weakness in both hands and wrists, shoulder pain worse with overhead use. Exams: Mild limp on left lower extremity, mild paracervical muscle tenderness, cervical AROM: 50 deg flexion, 45 deg extension, 35 deg RLF, 35 deg LLF, 70 deg RR, 70 deg LR, tender anterior and lateral lower ribs, difficulty sitting up due to rib pain, thoracic AROM: 10 deg flexion, 10 deg RR, 5 deg LR, difficulty heel and toe walking on left, spasm L>R paralumbar muscles, SLR positive left, facet loading maneuvers positive, lumbar AROM: fingertips to below knees, 15 deg extension, tender left ACJ, positive impingement signs left shoulder, left shoulder AROM: 125 deg flexion, 120 deg abduction, 75 deg ER, 40 deg IR, diffuse tenderness in both dorsal hands, grip strength: 16.5 lbs left, 57.0 lbs right, diffuse decreased sensation to pinprick left lower extremity, weakness left EHL, DF, PF 4/5. DX: Low back pain, lumbar sprain/strain with left radiculopathy, cervicalgia, right and left shoulder pain with left shoulder impingement, right and left hand pain with loss of left grip strength, ribs pain. TX: Condition unchanged, authorized treatment for thoracic spine, lumbar spine, and right wrist. Requested X-rays for lumbar and thoracic spine and right hand, PT for lumbar and thoracic spine. A QME needed for disputed body parts. Treatment includes Tylenol. PT not attended due to change of primary treating physician. No past work.
                                    
                                    ---------------------------------------------------------------------------------------------
                                    10 / 01 / 2024 | Request for Authorization and Progress Report | 6 pgs	
                                    Request for Authorization and Progress Report
                                    Sova Khuong, D.C., QME
                                    CC: Patient continues with right wrist/hand pain affecting grasping and strength activities. He remains with pain in the lower ribs, mid and low back with left leg numbness. Limitations are unchanged with prolonged walk/sit, bending and heavy lifting. Pain is rated 4-8/10. Exams: Mild limp on left lower extremity without external device. Mild paracervical muscle tenderness without midline tenderness. Tender anterior and lateral lower ribs, right upper quadrant, left upper quadrant. Difficulty heel and toe walk on left lower extremity due to pain and weakness. Spasm left greater than right paralumbar muscles. Tender left greater than right paralumbar and lumbosacral regions. Straight leg raise positive left. Facet loading maneuvers positive. Lumbar active range of motion: fingertips to below knees in flexion, 15 degrees extension. Diffuse tenderness bilateral dorsum hands and all digits without swelling or atrophy. DX: Low back pain; lumbar sprain/strain with left radiculopathy; cervicalgia; right and left shoulder pain with left shoulder impingement and left grip strength loss; ribs pain. Right wrist x-rays: no acute fracture or other acute finding. Thoracic spine x-rays: mild degenerative changes. Lumbar spine x-rays: mild degenerative changes. TX: Patient seen in follow up for his industrial injury. Unchanged condition with authorization for physical therapy for thoracic and lumbar spine pending scheduling. Course of occupational therapy for skilled hand therapy x8 for the right wrist/hand to improve grip strength recommended. Observation of response to treatment. A QME needed to address disputed parts: left wrist, neck, shoulders.
                                    
                                    -------------------------------------------------------------------------------------------------
                                """
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
