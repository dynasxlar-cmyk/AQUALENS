import streamlit as st
from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="Microplastic Detector AI", layout="wide")

# Constants from your original code
AREA_OF_IMAGE = 0.48
FILTER_AREA = 17.35
SAMPLE_VOLUME_L = 0.25
IOU_THRESHOLD = 0.3

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_model():
    # Ensure best.pt is in the same directory or provide the full path
    return YOLO("best.pt") 

model = load_model()

# --- HELPER FUNCTIONS ---
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

# --- UI DESIGN ---
st.title("🔬Microplastic Detection & Analysis")
st.markdown("Upload sample images to calculate concentrations per liter.")

# Sidebar for Parameters
st.sidebar.header("Calculation Parameters")
area_img = st.sidebar.number_input("Area of Image (m²)", value=AREA_OF_IMAGE)
filter_area = st.sidebar.number_input("Filter Area (m²)", value=FILTER_AREA)
vol_l = st.sidebar.number_input("Sample Volume (L)", value=SAMPLE_VOLUME_L)

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    all_results = []
    
    for uploaded_file in uploaded_files:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Detection
        results = model(opencv_image, conf=0.1)
        boxes = results[0].boxes.data.cpu().numpy() if len(results[0].boxes.data) else np.empty((0,6))
        
        if boxes.shape[0] > 0:
            filtered_boxes = non_max_suppression(boxes[:, :5], IOU_THRESHOLD)
            num_particles = filtered_boxes.shape[0]
        else:
            num_particles = 0
            
        # Calculations
        particles_in_filter = (num_particles / area_img) * filter_area
        particles_per_L = particles_in_filter / vol_l
        
        # Annotate
        annotated_frame = results[0].plot()
        gray_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)
        
        # Store for Table
        all_results.append({
            "Image": uploaded_file.name,
            "Count": num_particles,
            "Filter Total": round(particles_in_filter, 2),
            "Particles/L": round(particles_per_L, 2),
            "processed_img": gray_frame
        })

    # --- DISPLAY ---
    df = pd.DataFrame(all_results).drop(columns=['processed_img'])
    
    # Summary Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Particles/L", round(df["Particles/L"].mean(), 2))
    m2.metric("Total Particles Found", df["Count"].sum())
    m3.metric("Max Density", df["Particles/L"].max())

    st.dataframe(df, use_container_width=True)

    # Gallery view of processed images
    with st.expander("View Processed Annotated Images"):
        cols = st.columns(3)
        for idx, res in enumerate(all_results):
            cols[idx % 3].image(res['processed_img'], caption=res['Image'])

    # Download Results
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button("Download Analysis as CSV", data=csv, file_name="plastic_analysis.csv", mime="text/csv")
