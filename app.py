"""
Cross-Sensor Image Translation: Optical-to-SAR Using Deep Learning
Master Application Entry Point (ISRO / URSC Internship Demonstration)
"""
import streamlit as st
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ui_theme import apply_theme, render_mission_banner
from src.utils.device import get_device_info
from src.datasets.dataset_registry import DatasetRegistry
from src.storage.experiment_storage import ExperimentStorage
from src.training.checkpoints import list_checkpoints

# Configure page
st.set_page_config(
    page_title="Optical-to-SAR | ISRO Demonstration",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()
render_mission_banner("Cross-Sensor Image Translation: Optical-to-SAR")

# Sidebar Mission Identity
with st.sidebar:
    st.markdown("### 🛰️ ISRO / URSC Demonstration")
    st.markdown("**Mission:** Cross-Sensor Image Translation")
    st.markdown("**Primary Model:** PyTorch Pix2Pix CGAN")
    st.markdown("**Target Sensors:** Sentinel-2 (Opt) → Sentinel-1 (SAR)")
    st.markdown("---")
    st.markdown("#### 🧭 Quick Navigation Guide")
    st.markdown(
        """
        1. **Mission Overview:** Research scope & status
        2. **Dataset Manager:** Selective retrieval (SEN1-2)
        3. **Data Exploration:** Multimodal inspection
        4. **Model Training:** Genuine PyTorch CGAN training
        5. **Optical-to-SAR:** Live file translation & GeoTIFF export
        6. **Evaluation Lab:** PSNR / SSIM & Error Maps
        7. **Experiment History:** Tracked JSON benchmark runs
        8. **System Health:** Telemetry & connectivity
        """
    )
    st.markdown("---")
    st.caption("Developed for Indian Space Research Organisation (ISRO) internship evaluation.")

# Initialize session state defaults
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    st.session_state["selected_dataset"] = "SEN1-2"

# Welcome Hero Section
st.markdown(
    """
    ### Welcome to the Cross-Sensor Remote Sensing Translation Platform
    This complete deep learning application bridges the gap between **multispectral optical satellite imagery** 
    and **synthetic aperture radar (SAR)** observations.
    
    Designed specifically for rigorous technical demonstration, this platform features genuine data retrieval, 
    real neural network training with backpropagation, full quantitative metrics (PSNR, SSIM, MAE, MSE), 
    and honest reporting of remote data accessibility.
    """
)

# Operational Status Quick Board
device_info = get_device_info()
storage = ExperimentStorage()
experiments = storage.list_experiments()
checkpoints = list_checkpoints()

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Compute Engine", f"{device_info['device'].upper()}", device_info["device_name"])
col_m2.metric("Saved Models", f"{len(checkpoints)} Checkpoints")
col_m3.metric("Completed Runs", f"{len(experiments)} Experiments")
col_m4.metric("Dataset Verification", "SEN1-2 Active", "CC BY 4.0")

st.markdown("---")
st.markdown("### 🗺️ System Architecture & Workflow Pipeline")

col_wf1, col_wf2 = st.columns(2)

with col_wf1:
    st.markdown(
        """
        ```
        [ Sentinel-2 Optical (13-Band / True Color RGB) ]
                           │
                           ▼
          [ Selective Remote HTTP Streamer / Cache ]
                           │
                           ▼
          [ Preprocessing & Calibrated Decibel Normalization ]
                           │
                           ▼
          [ Conditional U-Net 256 Generator ]
               │                          │
               │ (Adversarial + L1 Loss)  ▼
               ▼                  [ PatchGAN 70x70 Discriminator ]
        [ Synthetic SAR Output ]          ▲
               │                          │
               └───────────┬──────────────┘
                           ▼
        [ Calibrated GeoTIFF / Visual PNG / PSNR & SSIM Evaluation ]
        ```
        """
    )

with col_wf2:
    st.markdown("#### 🚀 Recommended Demonstration Walkthrough:")
    st.markdown(
        """
        1. **Step 1 — Dataset Manager:** Go to `Dataset Manager` and retrieve a smoke test subset (e.g. 10 or 25 pairs) of **SEN1-2**.
        2. **Step 2 — Data Exploration:** Inspect the raw Sentinel-1 radar backscatter (dB) and Sentinel-2 optical bands.
        3. **Step 3 — Training:** Train the baseline Pix2Pix model for 2–5 epochs in the `Model Training Laboratory` and watch loss curves converge in real time.
        4. **Step 4 — Translation:** Go to `Optical-to-SAR Translation` and test the model on any uploaded optical satellite scene.
        5. **Step 5 — Evaluation:** In the `Evaluation Laboratory`, inspect the residual error heatmap and exact PSNR / SSIM metrics.
        """
    )

st.markdown("---")
st.info("👈 Use the left sidebar to navigate across all 8 operational pages.")
