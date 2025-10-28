import io
import streamlit as st
import os
import re
from PIL import Image
import pdf2image # For PDF to image conversion (CV pre-processing)
import pytesseract # For Optical Character Recognition (OCR/CV)
from sklearn.feature_extraction.text import TfidfVectorizer # For NLP/ML
from sklearn.metrics.pairwise import cosine_similarity # For ML match score
import nltk
from nltk.corpus import stopwords
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- NLTK & Global Setup ---
# Classical NLP needs to download resource files like stopwords
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# --- ML/NLP UTILITY FUNCTIONS (Replacing LLM) ---

def preprocess_text(text):
    """Clean text: lowercase, remove non-alphanumeric, remove stopwords."""
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    # Filter stopwords and short words
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]
    return ' '.join(tokens)

def calculate_match_percentage(resume_text, job_description):
    """Calculates match percentage using TF-IDF and Cosine Similarity (Classical ML)."""
    
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(job_description)
    
    if not processed_resume or not processed_jd:
        return 0, []
    
    documents = [processed_resume, processed_jd]
    
    # ML: Use TF-IDF to convert text into numerical vectors
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # ML: Calculate Cosine Similarity (the percentage match score)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    percentage = round(similarity_score * 100)
    
    # Extract features for keyword comparison
    feature_names = vectorizer.get_feature_names_out()
    jd_vector = tfidf_matrix[1].toarray()[0]
    
    # Get top 20 relevant words in JD based on their TF-IDF weight
    top_jd_indices = jd_vector.argsort()[-20:][::-1]
    top_jd_keywords = [feature_names[i] for i in top_jd_indices if jd_vector[i] > 0.1] # Filter for relevance
    
    return percentage, top_jd_keywords

def create_ats_report(resume_text, job_description):
    """Creates a basic ATS-style report using deterministic NLP and ML scores."""
    
    percentage, top_jd_keywords = calculate_match_percentage(resume_text, job_description)
    
    # Identify missing keywords based on top JD terms
    resume_processed_words = set(preprocess_text(resume_text).split())
    missing_keywords = [
        word for word in top_jd_keywords
        if word not in resume_processed_words and len(word) > 3
    ][:5]
    
    # --- Report Formatting (Replaces LLM Text Generation) ---
    report = f"**Percentage Match:** {percentage}%\n\n"
    
    report += "**Missing Keywords (Top 5 based on TF-IDF):**\n"
    if missing_keywords:
        report += "\n".join([f"- {k.title()}" for k in missing_keywords])
    else:
        report += "- All major keywords found or job description terms are too generic."
        
    report += "\n\n**Final Thoughts (Rule-Based):**\n"
    if percentage >= 70:
        report += "- **Strong Candidate**: Excellent structural alignment. The resume is highly relevant to the job posting."
    elif percentage >= 40:
        report += "- **Mid-level Match**: Fair overlap, but needs specific tailoring. Ensure the missing keywords are explicitly mentioned."
    else:
        report += "- **Low Match**: Significant mismatch. Major overhaul required to align the content with the JD."
        
    return report

def create_hr_review(resume_text, job_description):
    """Creates a basic HR review based on simple rules and document length."""
    
    review = "**Professional Evaluation (Rule-Based Approximation):**\n\n"
    
    # Check Alignment (uses ML score)
    percentage, _ = calculate_match_percentage(resume_text, job_description)
    
    review += f"- **Alignment Score**: {percentage}%\n"
    
    if percentage >= 70:
        review += "- **Strength**: The candidate's profile shows strong alignment with the job's core vocabulary and requirements.\n"
    elif percentage < 50:
        review += "- **Weakness**: The resume lacks specific details and terminology to pass a manual HR review easily.\n"
        
    # Check for keywords like "management", "leadership", etc.
    if re.search(r'leadership|management|senior', resume_text, re.IGNORECASE):
         review += "- **Strength**: Leadership/Management experience keywords are present.\n"
    
    # Check for length (a proxy for detail)
    if len(resume_text.split()) < 300:
        review += "- **Weakness**: Resume is short. Consider adding more measurable achievements (metrics, numbers).\n"
    
    return review

# --- CV/OCR Function (Modified for Tesseract) ---

def input_pdf_setup(uploaded_file):
    """Converts the uploaded PDF to raw text using OCR."""
    if uploaded_file is not None:
        try:
            # 1. Convert PDF page to image (CV Pre-processing)
            # The .read() function must be called inside the loop or function that processes the file
            image = pdf2image.convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1)[0]
            
            # 2. Use Tesseract (OCR) to extract text
            resume_text = pytesseract.image_to_string(image)
            return resume_text
        except Exception as e:
            # Inform user about missing dependencies
            st.error("Error: Failed to read PDF content. Please ensure **Tesseract OCR** and **Poppler** (if on Windows) are correctly installed.")
            raise FileNotFoundError(f"Failed to read PDF content: {e}")
    else:
        raise FileNotFoundError("No file uploaded")
    
# --- STREAMLIT APP ---
    
st.set_page_config(page_title="Classical ATS Expert", layout="wide")
st.header(" Classical ATS Tracking System (Non-LLM)")

# --- USER INPUTS ---
input_text = st.text_area("Job Description: ", key='input', height=200,help="Paste the full job description here.")

uploaded_files = st.file_uploader(
    "Upload your Resumes (PDF)...",
    type=["pdf"],
    accept_multiple_files=True,
    key='uploaded_files_input',
    help="You can drag and drop multiple resumes here."
)

# --- SUBMISSION BUTTONS (COLORIZED) ---
col1, col2 = st.columns(2)
with col1:
    submit_full_report = st.button("Generate Full Report for All Resumes", type="primary") 
with col2:
    submit_summary_table = st.button("Generate Percentage Summary Table", type="secondary")

# --- MAIN LOGIC ---

if submit_full_report:
    if uploaded_files and input_text:
        for i, file in enumerate(uploaded_files):
            st.markdown(f"<h3 style='color: #007BFF;'>--- 📄 Analyzing Resume {i+1}: {file.name} ---</h3>", unsafe_allow_html=True) 

            with st.spinner(f'Analyzing {file.name} using Classical ML/NLP...'):
                try:
                    # 1. Get raw text from PDF using OCR/CV
                    resume_text = input_pdf_setup(file)
                    
                    # 2. ATS Match Report (uses ML and NLP)
                    ats_report = create_ats_report(resume_text, input_text)
                    st.markdown("#### ATS Match Report (ML/NLP)")
                    st.write(ats_report)
                    
                    # 3. HR Review (uses rule-based logic)
                    hr_review = create_hr_review(resume_text, input_text)
                    with st.expander("Click to view HR Professional Evaluation (Rule-Based)"):
                        st.markdown("#### HR Professional Evaluation")
                        st.write(hr_review)

                except Exception as e:
                    st.error(f"An error occurred while processing **{file.name}**. Check dependencies: {e}")
                    
    else:
        st.warning("Please upload at least one resume and provide a Job Description.")

# --- SUMMARY TABLE LOGIC ---
elif submit_summary_table:
    if uploaded_files and input_text:
        st.subheader("Resume Match Percentage Summary")
        summary_data = []
        
        with st.spinner('Generating summary table using Cosine Similarity...'):
            for i, file in enumerate(uploaded_files):
                st.info(f"Processing Resume {i+1}: {file.name}")
                try:
                    resume_text = input_pdf_setup(file)
                    
                    # Get the percentage only (ML)
                    percentage, _ = calculate_match_percentage(resume_text, input_text)
                    
                    summary_data.append({
                        "Resume Number": i + 1,
                        "Resume File Name": file.name,
                        "Match Percentage": f"{percentage}%"
                    })
                    
                except Exception:
                    summary_data.append({
                        "Resume Number": i + 1,
                        "Resume File Name": file.name,
                        "Match Percentage": "Error (OCR/ML Failed)"
                    })

            if summary_data:
                st.dataframe(summary_data, use_container_width=True)
                st.success("Summary table generated successfully using classical ML.")
            else:
                st.error("No valid resumes were processed.")
    else:
        st.warning("Please upload at least one resume and provide a Job Description.")