import hashlib
import hmac
import io
import time
import zipfile
 
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
 
# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AQUALENS | Secure Microplastic AI",
    page_icon="🔬",
    layout="wide",
)
 
# --- 2. LOGIN SECURITY GATE ---
# Password now comes from st.secrets (Settings -> Secrets in Streamlit Cloud,
# or a local .streamlit/secrets.toml) instead of being hardcoded in source.
# secrets.toml:
#   APP_PASSWORD = "your-password-here"
MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60
 
 
def _get_expected_password_hash() -> str:
    raw = st.secrets.get("APP_PASSWORD", None)
    if raw is None:
        # Fallback so the app still runs if secrets aren't configured yet,
        # but this should be replaced before any real deployment.
        raw = "wheidpogi"
    return hashlib.sha256(raw.encode()).hexdigest()
 
 
def check_password() -> None:
    st.session_state.setdefault("password_correct", False)
    st.session_state.setdefault("failed_attempts", 0)
    st.session_state.setdefault("lockout_until", 0.0)
 
    if st.session_state["password_correct"]:
        return
 
    st.markdown(
        """
        <div style='text-align: center; padding: 50px;'>
            <h1 style='color: #004e92; font-size: 3.5rem; letter-spacing: 5px; font-weight: 800; font-family: sans-serif;'>AQUALENS</h1>
            <p style='color: #666; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px;'>Secure Analysis Portal</p>
            <div style='margin: 20px auto; width: 100px; border-top: 3px solid #004e92;'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    now = time.time()
    locked_out = now < st.session_state["lockout_until"]
 
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        if locked_out:
            remaining = int(st.session_state["lockout_until"] - now)
            st.error(f"🚫 Too many attempts. Try again in {remaining}s.")
        else:
            password = st.text_input("Enter Access Key", type="password")
            if st.button("Unlock Portal", use_container_width=True):
                candidate_hash = hashlib.sha256(password.encode()).hexdigest()
                expected_hash = _get_expected_password_hash()
                if hmac.compare_digest(candidate_hash, expected_hash):
                    st.session_state["password_correct"] = True
                    st.session_state["failed_attempts"] = 0
                    st.rerun()
                else:
                    st.session_state["failed_attempts"] += 1
                    if st.session_state["failed_attempts"] >= MAX_ATTEMPTS:
                        st.session_state["lockout_until"] = now + LOCKOUT_SECONDS
                        st.session_state["failed_attempts"] = 0
                        st.rerun()
                    st.error("🚫 Access Denied: Invalid Key")
 
    st.stop()
 
 
check_password()
 
# --- 3. THE FULL APPLICATION (Runs only after login) ---
 
st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)
 
with st.sidebar:
    st.markdown("### 🔒 Security")
    if st.button("Log Out of Portal"):
        st.session_state["password_correct"] = False
        st.rerun()
 
    st.markdown("---")
    st.header("⚙️ Parameters")
    area_img = st.number_input("Area of Image (m²)", value=0.48, min_value=0.0001)
    filter_area = st.number_input("Filter Area (m²)", value=17.35, min_value=0.0)
    vol_l = st.number_input("Sample Volume (L)", value=0.25, min_value=0.0001)
    conf_thresh = st.slider("Confidence Threshold", 0.01, 1.0, 0.10)
    iou_thresh = st.slider("NMS Threshold", 0.1, 1.0, 0.3)
 
st.markdown(
    """
    <div class="header-container">
        <p class="logo-text">AQUALENS</p>
        <p class="sub-text">Precise Microplastic Detection AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# --- 4. CACHED MODEL LOADING ---
@st.cache_resource(show_spinner="Loading detection model...")
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load model weights (best.pt): {exc}")
        st.stop()
 
 
model = load_model()
 
# --- 5. HELPER FUNCTIONS ---
def non_max_suppression(boxes: np.ndarray, iou_thresh: float) -> np.ndarray:
    """boxes: array of [x1, y1, x2, y2, score]."""
    if boxes.shape[0] == 0:
        return boxes
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = np.maximum(0.0, (x2 - x1)) * np.maximum(0.0, (y2 - y1))
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return boxes[keep]
 
 
@st.cache_data(show_spinner=False)
def run_detection(file_bytes: bytes, conf_thresh: float):
    """Runs YOLO inference once per (file, confidence) pair and caches the
    raw boxes plus a rendered annotation. NMS threshold is applied later so
    moving that slider doesn't re-trigger inference."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    opencv_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if opencv_image is None:
        return None, None
 
    results = model(opencv_image, conf=conf_thresh, verbose=False)
    boxes = (
        results[0].boxes.data.cpu().numpy()
        if len(results[0].boxes.data)
        else np.empty((0, 6))
    )
    # results[0].plot() returns a BGR array (OpenCV convention); convert to
    # RGB so colors render correctly in st.image / PIL.
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return boxes, annotated_rgb
 
 
# --- 6. MAIN WORKFLOW ---
uploaded_files = st.file_uploader(
    "📤 Upload microscope images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)
 
if uploaded_files:
    all_results = []
    skipped = []
 
    progress = st.progress(0.0, text="Analyzing samples with YOLO AI...")
    total = len(uploaded_files)
 
    for idx, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.getvalue()
        boxes, annotated_rgb = run_detection(file_bytes, conf_thresh)
 
        if boxes is None:
            skipped.append(uploaded_file.name)
            progress.progress((idx + 1) / total)
            continue
 
        num_particles = (
            non_max_suppression(boxes[:, :5], iou_thresh).shape[0]
            if boxes.shape[0] > 0
            else 0
        )
 
        particles_in_filter = (num_particles / area_img) * filter_area
        particles_per_L = particles_in_filter / vol_l
 
        all_results.append(
            {
                "Image": uploaded_file.name,
                "Count": num_particles,
                "Filter Total": round(particles_in_filter, 2),
                "Particles/L": round(particles_per_L, 2),
                "processed_img": annotated_rgb,
            }
        )
        progress.progress((idx + 1) / total)
 
    progress.empty()
 
    if skipped:
        st.warning(f"Could not read {len(skipped)} file(s), skipped: {', '.join(skipped)}")
 
    if not all_results:
        st.error("No images could be processed. Please check the uploaded files.")
    else:
        df = pd.DataFrame(all_results)
 
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Particles/L", round(df["Particles/L"].mean(), 2))
        m2.metric("Total Particles Found", int(df["Count"].sum()))
        m3.metric("Max Density", f"{df['Particles/L'].max()} P/L")
 
        st.dataframe(df.drop(columns=["processed_img"]), use_container_width=True)
 
        with st.expander("🖼️ View Annotated Visuals"):
            cols = st.columns(3)
            for idx, res in enumerate(all_results):
                cols[idx % 3].image(
                    res["processed_img"], caption=res["Image"], use_container_width=True
                )
 
        col_a, col_b = st.columns(2)
 
        csv = df.drop(columns=["processed_img"]).to_csv(index=False).encode("utf-8")
        col_a.download_button(
            "📥 Download Analysis CSV",
            data=csv,
            file_name="aqualens_report.csv",
            mime="text/csv",
        )
 
        # Bundle annotated images into a zip for convenient download.
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for res in all_results:
                ok, buf = cv2.imencode(
                    ".png", cv2.cvtColor(res["processed_img"], cv2.COLOR_RGB2BGR)
                )
                if ok:
                    zf.writestr(f"annotated_{res['Image']}.png", buf.tobytes())
        col_b.download_button(
            "🖼️ Download Annotated Images (.zip)",
            data=zip_buffer.getvalue(),
            file_name="aqualens_annotated_images.zip",
            mime="application/zip",
        )
else:
    st.info("System Ready. Please upload images to begin.")
