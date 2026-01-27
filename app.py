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
        # Centered Login UI
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
                if password == "sirjorgepogi": # You can change this password here
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("🚫 Access Denied: Invalid Key")
        
        # Stop execution here so the rest of the app is hidden
        st.stop()

# Run the gate
check_password()

# --- 3. THE FULL APPLICATION (Runs only after login) ---

# Sleek Custom Styling (CSS)
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .header-container {
        display: flex; flex-direction: column; align-items: center;
        padding: 40px; background: linear-gradient(135deg, #004e92 0%, #000428 100%);
        border-radius: 20px; color: white; margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .logo-text { font-weight: 800; font-size: 4rem; letter-spacing: 10px; margin: 0; font-family: sans-serif; }
    .sub-text { font-size: 1.2rem; opacity: 0.8; letter-spacing: 3px; text-transform: uppercase; }
    div[data-testid="stMetric"] {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #eef2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar (Logout & Original Parameters)
with st.sidebar:
    st.markdown("### 🔒 Security")
    if st.button("Log Out of Portal"):
        st.session_state["password_correct"] = False
        st.rerun()
    
    st.markdown("---")
    st.header("⚙️ Parameters")
    area_img = st.number_input("Area of Image (m²)", value=0.48)
    filter_area = st.number_input("Filter Area (m²)", value=17.35)
    vol_l = st.number_input("Sample Volume (L)", value=0.25)
    iou_thresh = st.slider("NMS Threshold", 0.1, 1.0, 0.3)

# Header Section
st.markdown("""
    <div class="header-container">
        <p class="logo-text">AQUALENS</p>
        <p class="sub-text">Precision Microplastic Detection AI</p>
    </div>
    """, unsafe_allow_html=True)

# --- 4. CACHED MODEL LOADING ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 5. HELPER FUNCTIONS ---
def non_max_suppression(boxes, iou_thresh):
    if len(boxes) == 0: return boxes
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return boxes[keep]

# --- 6. MAIN WORKFLOW ---
uploaded_files = st.file_uploader("📤 Upload microscope images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    all_results = []
    
    with st.status("Analyzing samples with YOLO AI...", expanded=True) as status:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            
            # Detection
            results = model(opencv_image, conf=0.1)
            boxes = results[0].boxes.data.cpu().numpy() if len(results[0].boxes.data) else np.empty((0,6))
            
            num_particles = non_max_suppression(boxes[:, :5], iou_thresh).shape[0] if boxes.shape[0] > 0 else 0
            
            # Original Calculations
            particles_in_filter = (num_particles / area_img) * filter_area
            particles_per_L = particles_in_filter / vol_l
            
            all_results.append({
                "Image": uploaded_file.name,
                "Count": num_particles,
                "Filter Total": round(particles_in_filter, 2),
                "Particles/L": round(particles_per_L, 2),
                "processed_img": results[0].plot()
            })
        status.update(label="Analysis Complete!", state="complete")

    # Metrics Display
    df = pd.DataFrame(all_results)
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Particles/L", round(df["Particles/L"].mean(), 2))
    m2.metric("Total Particles Found", int(df["Count"].sum()))
    m3.metric("Max Density", f"{df['Particles/L'].max()} P/L")

    st.dataframe(df.drop(columns=['processed_img']), use_container_width=True)

    # Gallery
    with st.expander("🖼️ View Annotated Visuals"):
        cols = st.columns(3)
        for idx, res in enumerate(all_results):
            cols[idx % 3].image(res['processed_img'], caption=res['Image'], use_container_width=True)

    # Export
    csv = df.drop(columns=['processed_img']).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Analysis CSV", data=csv, file_name="aqualens_report.csv", mime="text/csv")
else:
    st.info("System Ready. Please upload images to begin.")

