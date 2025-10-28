import base64
import io
import time
from dotenv import load_dotenv #type:ignore
load_dotenv()
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai #type :ignore
import pdf2image #type: ignore

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- CORE FUNCTIONS ---

def get_gemini_response(prompt_input, pdf_content, job_description):
    """Generates content using the Gemini model."""
    # Using the recommended model for multimodal (vision) tasks
    model=genai.GenerativeModel('gemini-2.5-flash')
    # The contents array includes the prompt text, the image part of the PDF, and the job description
    response=model.generate_content([prompt_input, pdf_content[0], job_description])
    return response.text

def input_pdf_setup(uploaded_file):
    """Converts the first page of the uploaded PDF to a base64 encoded image."""
    if uploaded_file is not None:
        try:
            # Use pdf2image to convert the first page of the PDF bytes to an image
            # Note: On Windows, you may need to have Poppler installed for this to work.
            image = pdf2image.convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1)
        except Exception as e:
            st.error(f"Error processing PDF. Ensure **Poppler** is installed and accessible if you are on Windows. Details: {e}")
            raise e
            
        first_page = image[0]
        img_bytes_arr = io.BytesIO()
        first_page.save(img_bytes_arr, format='JPEG')
        img_bytes_arr = img_bytes_arr.getvalue()
        
        pdf_parts=[
            {
                "mime_type":"image/jpeg",
                "data":base64.b64encode(img_bytes_arr).decode()
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
# --- STREAMLIT APP CONFIG ---

st.set_page_config(page_title="ATS Resume Expert", layout="wide")
st.header("ATS Tracking System")

# --- USER INPUTS ---

# Input for the Job Description
input_text = st.text_area("Job Description: ", key='input', height=200, 
    help="Paste the full job description here.")

# Multi-file uploader for Resumes
uploaded_files = st.file_uploader(
    "Upload your Resumes (PDF)...",
    type=["pdf"],
    accept_multiple_files=True, # ALLOWS MULTIPLE FILES
    key='uploaded_files_input',
    help="You can drag and drop up to 10 resumes at once."
)

if uploaded_files:
    st.info(f"**{len(uploaded_files)}** PDF(s) ready for analysis.")
elif uploaded_files is None:
    st.write("Please upload one or more resumes to begin.")


# Prompt for professional HR evaluation
input_prompt1 = """
You are an experienced HR professional. Your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements. 
Provide your response in a clear, concise format using bullet points.
"""

# Consolidated prompt for detailed ATS report (replaces old input_prompt3 and input_prompt4)
input_prompt_ats_report = """
You are a highly skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and HR requirements.
Your task is to evaluate the uploaded resume against the provided job description.

Please structure your output **strictly** into three markdown sections:
1.  **Percentage Match:** Provide ONLY the numerical percentage value indicating the match. (Example: 85%)
2.  **Missing Keywords:** List 3-5 critical keywords or skills from the Job Description that are missing or underrepresented in the Resume.
3.  **Actionable Summary:** Provide a final, brief summary (3-4 bullet points) on how the candidate can improve their resume specifically for THIS job description.
"""
input_prompt_percentage_only = """
As a highly advanced ATS, your sole function is to calculate the percentage match between the resume and the job description.
Provide **ONLY** a single numerical value representing the percentage match, followed by the '%' sign.
For example: "87%". Do not provide any explanations, keywords, or additional text.
"""

# --- SUBMISSION LOGIC ---

submit_report = st.button("Generate Full Report for All Resumes")
submit_percentage=st.button("Percentage compare")

if submit_report:
    if uploaded_files and input_text:
        # Loop through each uploaded file for batch processing
        for i, file in enumerate(uploaded_files):
            st.markdown(f"<h3 style='color: #4CAF50;'>---  Analyzing Resume {i+1}: {file.name} ---</h3>", unsafe_allow_html=True) 

            with st.spinner(f'Analyzing {file.name}...'):
                try:
                    # 1. Process the current file
                    pdf_content = input_pdf_setup(file)
                    
                    # 2. Get the Comprehensive ATS Report
                    ats_response = get_gemini_response(input_prompt_ats_report, pdf_content, input_text)
                    st.markdown("#### ATS Match Report")
                    st.write(ats_response)
                    
                    # 3. Get the HR Review (placed in an expander for cleaner output)
                    hr_response = get_gemini_response(input_prompt1, pdf_content, input_text)
                    with st.expander("Click to view HR Professional Evaluation"):
                        st.markdown("#### HR Professional Evaluation")
                        st.write(hr_response)

                except Exception as e:
                    # Handles errors during file reading or API call for a specific file
                    st.error(f"An error occurred while processing **{file.name}**. Please check the file format or the Poppler installation.")
                    # Optional: Print detailed error to console/logs: print(e)
                    
    else:
        st.warning("Please upload at least one resume and provide a Job Description before generating reports.")
elif submit_percentage:
    if uploaded_files and input_text:
        st.subheader(" Resume Match Percentage Summary")
        
        # List to store data for the final table
        summary_data = []
        
        with st.spinner('Generating summary table...'):
            for i, file in enumerate(uploaded_files):
                st.info(f"Processing Resume {i+1}: {file.name}")
                try:
                    pdf_content = input_pdf_setup(file)
                    
                    # Use the strict percentage-only prompt
                    percentage_response = get_gemini_response(input_prompt_percentage_only, pdf_content, input_text)
                    
                    # Clean up the response to ensure only the percentage string is taken
                    match_percentage = percentage_response.strip().split('\n')[0]
                    
                    summary_data.append({
                        "Resume Number": i + 1,
                        "Resume File Name": file.name,
                        "Match Percentage": match_percentage
                    })
                    
                except Exception as e:
                    st.warning(f"Could not process **{file.name}**. Skipping in summary. (Error: {e})")
                    summary_data.append({
                        "Resume Number": i + 1,
                        "Resume File Name": file.name,
                        "Match Percentage": "Error"
                    })

            # Display the final table using st.dataframe
            if summary_data:
                st.dataframe(summary_data, use_container_width=True)
                st.success("Summary table generated successfully.")
            else:
                st.error("No valid resumes were processed to generate the summary.")

    else:
        st.warning("Please upload at least one resume and provide a Job Description.")