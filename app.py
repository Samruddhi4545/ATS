import base64
import io
from dotenv import load_dotenv #type:ignore
load_dotenv()
import streamlit as st
import os 
from PIL import Image
import pdf2image
import google.generativeai as genai #type :ignore

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,pdf_content,prompt): 
    