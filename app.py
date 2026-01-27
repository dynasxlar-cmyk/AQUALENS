import streamlit as st
from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AQUALENS | Secure Microplastic AI",
    page_icon="🔬",
    layout="wide"
)

# --- 2. LOGIN SECURITY GATE ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.markdown("""
            <div style='text-align: center; padding: 50px;'>
                <h1 style='color: #004e92; font-size: 3.5rem; letter-spacing: 5px; font-weight: 800; font-family: sans-serif;'>AQUALENS</h1>
                <p style='color: #666; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px;'>Secure Analysis Portal</p>
                <div style='margin: 20px auto; width: 100px; border-top: 3px solid #004e92;'></div>
            </div>
            """, unsafe_allow_html=True)
        
        _, col2, _ = st.columns([1, 1, 1])
        with col2:
            password = st.text_input("Enter Access Key", type="password")
            if st.button("Unlock Portal", use_container_width=True):
                if password == "aqualens2026": 
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("🚫 Access Denied")
        st.stop()

check_password()

# --- 3. THE FULL APPLICATION ---

# CSS Styling
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .header-container {
        display: flex; flex-direction: column; align-items: center;
        padding: 40px; background: linear-gradient(135deg, #004e92 0%, #000428 100%);
        border-radius: 20px; color: white; margin-bottom: 30px;
    }
    .logo-text { font-weight: 800; font-size: 4rem; letter-spacing: 10px; margin: 0; font-family: sans-serif; }
    .sub-text { font-size: 1.2rem; opacity: 0.8; letter-spacing: 3px; text-transform: uppercase; }
    div[data-testid="stMetric"] {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #eef2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔒 Security")
    if st.button("Log Out"):
        st.session_state["password_correct"] = False
        st.rerun()
    
    st.markdown("---")
    st.header("⚙️ Parameters")
    area_img = st.number_input("Area of Image (m²)", value=0.48)
    filter_area =
