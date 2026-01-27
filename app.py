import streamlit as st
from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AQUALENS | Microplastic Detection AI",
    page_icon="🔬",
    layout="wide"
)

# --- 2. PROFESSIONAL STYLING (CSS) ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Sleek Header Container */
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px;
        background: linear-gradient(135deg, #004e92 0%, #000428 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .logo-text {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 4rem;
        letter-spacing: 8px;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-text {
        font-size: 1.2rem;
        opacity: 0.9;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 10px;
    }
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eef2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CACHED MODEL LOADING ---
@st.cache_resource
def load_model():
    # Looks for best.pt in the same folder as app.py
    return YOLO("best.pt") 

model = load_model()

# --- 4. DETECTION LOGIC ---
def non_max_suppression(boxes, iou_thresh):
    if len(boxes) == 0: return boxes
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return boxes[keep]

# --- 5. HEADER ---
st.markdown("""
    <div class="header-container">
        <p class="logo-text">AQUALENS</p>
        <p class="sub-text">Precision Microplastic Analysis Portal</p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. SIDEBAR PARAMETERS ---
st.sidebar.header("🔬 Analysis Parameters")
area_img = st.sidebar.number_input("Area of Image (m²)", value=0.48)
filter_area = st.sidebar.number_input("Filter Area (m²)", value=17.35)
vol_l = st.sidebar.number_input("Sample Volume (L)", value=0.25)
iou_thresh = st.sidebar.slider("NMS Threshold", 0.1, 1.0, 0.3)

# --- 7. MAIN INTERFACE ---
uploaded_files = st.file_uploader("📤 Upload Sample Images for Analysis", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    all_results = []
    
    with st.status("Processing images with AQUALENS AI...", expanded=True) as status:
        for uploaded_file in uploaded_files:
            # Image conversion
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            
            # AI Inference
            results = model(opencv_image, conf=0.1)
            boxes = results[0].boxes.data.cpu().numpy() if len(results[0].boxes.data) else np.empty((0,6))
            
            if boxes.shape[0] > 0:
                filtered_boxes = non_max_suppression(boxes[:, :5], iou_thresh)
                num_particles = filtered_boxes.shape[0]
            else:
                num_particles = 0
            
            # Particle concentration math
            particles_in_filter = (num_particles / area_img) * filter_area
            particles_per_L = particles_in_filter / vol_l
            
            # Annotation
            annotated_frame = results[0].plot()
            
            all_results.append({
                "Image": uploaded_file.name,
                "Count": num_particles,
                "Filter Total": round(particles_in_filter, 2),
                "Particles/L": round(particles_per_L, 2),
                "processed_img": annotated_frame
            })
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # Summary Metrics
    df = pd.DataFrame(all_results)
    m1, m2, m3 = st.columns(3)
    m1.metric("Average Concentration", f"{round(df['Particles/L'].mean(), 2)} P/L")
    m2.metric("Total Particles Detected", int(df["Count"].sum()))
    m3.metric("Max Density Found", f"{df['Particles/L'].max()} P/L")

    # Data Display
    st.markdown("### 📊 Detailed Findings")
    st.dataframe(df.drop(columns=['processed_img']), use_container_width=True)

    # Gallery
    with st.expander("🖼️ View Annotated Results", expanded=True):
        cols = st.columns(3)
        for idx, res in enumerate(all_results):
            cols[idx % 3].image(res['processed_img'], caption=f"{res['Image']} ({res['Count']} particles)", use_container_width=True)

    # Export
    csv = df.drop(columns=['processed_img']).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Detailed Report (CSV)",
        data=csv,
        file_name="aqualens_report.csv",
        mime="text/csv",
    )
else:
    st.info("Please upload one or more microscope images to begin the analysis.")
