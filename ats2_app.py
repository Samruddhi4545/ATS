import io
import streamlit as st
import os
import re
import uuid # For unique keys in custom button
from PIL import Image
import pdf2image
import pytesseract#type:ignore
from sklearn.feature_extraction.text import TfidfVectorizer#type:ignore
from sklearn.metrics.pairwise import cosine_similarity#type:ignore
import nltk#type:ignore
from nltk.corpus import stopwords#type:ignore
from nltk.stem import PorterStemmer # NEW IMPORT for stemming#type:ignore
import pandas as pd
import matplotlib.pyplot as plt

# Configure Tesseract path (required for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- NLTK & Global Setup (for Accuracy) ---
try:
    # Check and download necessary NLTK resources
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

STOPWORDS = set(stopwords.words('english'))
# Initialize Stemmer for linguistic normalization
STEMMER = PorterStemmer()
# Critical keywords for the Hybrid Score Boost (e.g., from your JavaFX context)
# Critical keywords for the Hybrid Score Boost (e.g., from your JavaFX context)
CRITICAL_KEYWORDS = {
    "DATA_STORAGE": ["sql", "database", "mongodb", "mysql", "nosql"],
    "AGILE_METHOD": ["scrum", "agile", "kanban", "sprint"],
    "JAVAFX_UI": ["java", "javafx", "fxml", "gui", "swing"],
    "CLOUD_OPS": ["aws", "cloud", "docker", "kubernetes", "azure"]
}
GROUP_BONUS = 50 # A major score boost per category

# --- UTILITY FUNCTIONS ---

def custom_button(label, key, color="#1E88E5", hover_color="#1565C0"):
    """Generates a custom-styled, interactive button using HTML and CSS."""
    unique_id = str(uuid.uuid4())
    st.markdown(f"""
    <style>
    #{unique_id} .stButton > button {{
        background-color: {color};
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
        transition-duration: 0.4s;
        width: 100%;
    }}
    #{unique_id} .stButton > button:hover {{
        background-color: {hover_color};
        color: white;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }}
    </style>
    <div id="{unique_id}">
    </div>
    """, unsafe_allow_html=True)
    # Streamlit requires the st.button call to manage the state
    return st.button(label, key=key, use_container_width=True)

def preprocess_text(text):
    """Clean text: lowercase, remove non-alphanumeric, remove stopwords, and ADD STEMMING (Accuracy Boost)."""
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    
    # Filter stopwords and short words
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]
    
    # --- ACCURACY BOOST: STEMMING ---
    # Normalize words to their root form (e.g., 'running' -> 'run')
    tokens = [STEMMER.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def calculate_match_percentage(resume_text, job_description):
    """Calculates match percentage using TF-IDF and Cosine Similarity (Classical ML), with Hybrid Scoring."""
    
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(job_description)
    
    if not processed_resume or not processed_jd:
        return 0, []
    
    documents = [processed_resume, processed_jd]
    
    # --- ACCURACY BOOST: N-GRAMS & FEATURE CAPACITY ---
    vectorizer = TfidfVectorizer(
        max_features=5000, # Increased capacity
        ngram_range=(1, 3) # Capture single words and two-word phrases (e.g., 'deep learning')
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # ML: Calculate Cosine Similarity (Base Percentage)
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    base_percentage = round(similarity_score * 100)
    
    # --- ACCURACY BOOST: HYBRID SCORING (KEYWORD BOOST) ---
    resume_processed_words = set(processed_resume.split())
    
    group_hits = 0
    found_groups = set()
    
    # Iterate through groups and check if any stemmed word in the resume matches any group keyword
    for group_name, keywords in CRITICAL_KEYWORDS.items():
        if group_name not in found_groups:
            for keyword in keywords:
                if STEMMER.stem(keyword.lower()) in resume_processed_words:
                    group_hits += 1
                    found_groups.add(group_name) # Ensures we only count one bonus per category
                    break

    # Now, calculate bonus based on group hits, not individual word hits
    # This rewards candidates who demonstrate skills in a required domain.
    bonus_score = group_hits * GROUP_BONUS

    # Calculate Final Percentage (max 100)
    final_percentage = min(100, base_percentage + bonus_score)

    # --- Keyword Extraction ---
    feature_names = vectorizer.get_feature_names_out()
    jd_vector = tfidf_matrix[1].toarray()[0]
    top_jd_indices = jd_vector.argsort()[-20:][::-1]
    top_jd_keywords = [
        feature_names[i] for i in top_jd_indices
        if jd_vector[i] > 0.1 and feature_names[i] in processed_jd
    ]
    
    return final_percentage, top_jd_keywords

# (Your existing create_ats_report and create_hr_review functions go here, they use calculate_match_percentage)
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
    
def input_pdf_setup(uploaded_file):
    """Converts the uploaded PDF to raw text using OCR."""
    if uploaded_file is not None:
        try:
            # 1. Convert PDF page to image (CV Pre-processing)
            uploaded_file.seek(0) # Reset file pointer to the beginning
            # Use a higher DPI for better OCR results
            image = pdf2image.convert_from_bytes(uploaded_file.read(), first_page=1, last_page=1, dpi=300)[0]
            
            # 2. Use Tesseract (OCR) to extract text
            # Use 'eng' language for English, or adjust if needed
            resume_text = pytesseract.image_to_string(image, lang='eng')
            return resume_text
        except Exception as e:
            st.error("Error: Failed to read PDF content. Please ensure **Tesseract OCR** and **Poppler** are correctly installed.")
            raise FileNotFoundError(f"Failed to read PDF content: {e}")
    else:
        raise FileNotFoundError("No file uploaded")
    
# --- STREAMLIT APP ---
    
st.set_page_config(page_title="Classical ATS Expert", layout="wide")
st.header("ATS Tracking System (Non-LLM)")

# --- USER INPUTS ---
input_text = st.text_area("Job Description: ", key='input', height=200,help="Paste the full job description here.")

uploaded_files = st.file_uploader(
    "Upload your Resumes (PDF)...",
    type=["pdf"],
    accept_multiple_files=True,
    key='uploaded_files_input',
    help="You can drag and drop multiple resumes here."
)

# --- SUBMISSION BUTTONS (3-Column Layout with Custom Style) ---
col1, col2, col3 = st.columns(3)

with col1:
    # Custom-styled button for the primary action
    submit_full_report = custom_button(
        label="1. Generate FULL Candidate Reports",
        key="full_report_btn",
        color="#1E88E5",
        hover_color="#1565C0"
    )

with col2:
    # Standard Streamlit button
    submit_summary_table = st.button(
        "2. Generate Top Candidate Table",
        type="secondary",
        use_container_width=True,
        key="summary_btn"
    )

with col3:
    # Standard Streamlit button for the new graph
    submit_graph = st.button(
        "3. Generate Applicant Pool OVERVIEW (Graph)",
        type="secondary",
        use_container_width=True,
        key="graph_btn"
    )

# --- MAIN LOGIC ---

if submit_full_report:
    if uploaded_files and input_text:
        # ... (Full Report Logic - uses the enhanced calculate_match_percentage) ...
        for i, file in enumerate(uploaded_files):
            st.markdown(f"<h3 style='color: #007BFF;'>--- Analyzing Resume {i+1}: {file.name} ---</h3>", unsafe_allow_html=True) 

            with st.spinner(f'Analyzing {file.name} using Classical ML/NLP...'):
                try:
                    resume_text = input_pdf_setup(file)
                    
                    ats_report = create_ats_report(resume_text, input_text)
                    st.markdown("#### ATS Match Report (ML/NLP)")
                    st.write(ats_report)
                    
                    hr_review = create_hr_review(resume_text, input_text)
                    with st.expander("Click to view HR Professional Evaluation (Rule-Based)"):
                        st.markdown("#### HR Professional Evaluation")
                        st.write(hr_review)

                except Exception as e:
                    st.error(f"An error occurred while processing **{file.name}**. Check dependencies: {e}")
                    
    else:
        st.warning("Please upload at least one resume and provide a Job Description")

# --- GRAPH LOGIC (New Block) ---
elif submit_graph:
    if uploaded_files and input_text:
        st.subheader(" Applicant Pool Distribution Overview")
        summary_data = []

        with st.spinner('Calculating all match scores for visualization...'):
            for i, file in enumerate(uploaded_files):
                try:
                    resume_text = input_pdf_setup(file)
                    # Uses the newly accurate calculation
                    percentage, _ = calculate_match_percentage(resume_text, input_text) 
                    summary_data.append({"Match Percentage (%)": percentage})
                except Exception:
                    summary_data.append({"Match Percentage (%)": 0})
            
            if summary_data:
                # pandas and matplotlib imports are at the top now
                df = pd.DataFrame(summary_data)
                
                # --- GRAPH GENERATION ---
                st.markdown("### Match Score Distribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                ax.hist(
                    df['Match Percentage (%)'],
                    bins=10,
                    edgecolor='black',
                    color='#32CD32'
                )
                
                ax.set_title('Distribution of Applicant Match Scores', fontsize=16)
                ax.set_xlabel('Match Percentage (%)', fontsize=14)
                ax.set_ylabel('Number of Applicants', fontsize=14)
                ax.set_xticks(range(0, 101, 10))
                ax.grid(axis='y', alpha=0.5)
                
                st.pyplot(fig)
                plt.close(fig)
                
                st.success(f"Graph generated successfully for {len(df)} applicants.")
            else:
                st.error("No valid resumes were processed.")
    else:
        st.warning("Please upload at least one resume and provide a Job Description.")


# --- SUMMARY TABLE LOGIC (Stable Filtering Fix) ---
elif submit_summary_table:
    if uploaded_files and input_text:
        st.subheader(" Resume Match Percentage Summary")
        summary_data = []

        with st.spinner('Generating summary table using Cosine Similarity...'):
            for i, file in enumerate(uploaded_files):
                st.info(f"Processing Resume {i+1}: {file.name}")
                try:
                    resume_text = input_pdf_setup(file)
                    # Uses the newly accurate calculation
                    percentage, _ = calculate_match_percentage(resume_text, input_text)
                    
                    summary_data.append({
                        "Resume Number": i + 1,
                        "Resume File Name": file.name,
                        "Match Percentage (%)": percentage # Consistent column name
                    })

                except Exception:
                    summary_data.append({
                        "Resume Number": i + 1,
                        "Resume File Name": file.name,
                        "Match Percentage (%)": 0
                    })

            # --- Data Processing for Sorting and Filtering ---
            if summary_data:
                # pandas import is at the top now
                df = pd.DataFrame(summary_data)
                
                # 1. Sort the DataFrame by percentage match (Highest to Lowest)
                df_sorted = df.sort_values(by="Match Percentage (%)", ascending=False).reset_index(drop=True)
                
                # Session State Initialization (for stable number input)
                if 'top_n_candidates' not in st.session_state:
                    st.session_state['top_n_candidates'] = min(10, len(df_sorted))
                
                # 2. Add Filter Control
                st.markdown("---")
                
                filter_col, display_col = st.columns([1, 4])
                
                # This input updates st.session_state['top_n_candidates']
                filter_col.number_input(
                    "Showing Top N Candidates:",
                    min_value=1,
                    max_value=len(df_sorted),
                    value=st.session_state['top_n_candidates'], # Initialize/read from state
                    step=1,
                    key='top_n_candidates'
                )
                
                # *** STABLE FILTER FIX ***
                # Read the stable value from the session state dictionary
                N = st.session_state['top_n_candidates']

                # 3. Apply the Top N Filter
                df_filtered = df_sorted.head(N)

                # 4. Display the Final Table
                display_col.dataframe(
                    df_filtered.style.format({"Match Percentage (%)": "{}%"}),
                    use_container_width=True,
                )
                
                st.success(f"Displaying the top {len(df_filtered)} candidates (N={N}) based on Match Percentage.")
            
            else:
                st.error("No valid resumes were processed.")
    else:
        st.warning("Please upload at least one resume and provide a Job Description.")