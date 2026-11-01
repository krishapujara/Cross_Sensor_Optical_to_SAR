"""
Page 1: Mission Overview
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ui_theme import apply_theme, render_mission_banner
from src.utils.device import get_device_info
from src.datasets.dataset_registry import DatasetRegistry
from src.storage.experiment_storage import ExperimentStorage
from src.training.checkpoints import list_checkpoints

st.set_page_config(page_title="Mission Overview | Optical-to-SAR", page_icon="🛰️", layout="wide")
apply_theme()
render_mission_banner("Mission Overview & Operational Status")

# High-Level Metrics Row
device_info = get_device_info()
storage = ExperimentStorage()
experiments = storage.list_experiments()
checkpoints = list_checkpoints()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div class="aerospace-card">
            <div class="card-title">Compute Hardware</div>
            <div class="card-value">{device_info['device'].upper()}</div>
            <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">{device_info['device_name']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div class="aerospace-card">
            <div class="card-title">Active Models</div>
            <div class="card-value">{len(checkpoints)}</div>
            <div style="color: #38BDF8; font-size: 0.8rem; margin-top: 4px;">Saved Checkpoints</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    latest_psnr = experiments[0].get("best_val_psnr", 0.0) if experiments else 0.0
    st.markdown(
        f"""
        <div class="aerospace-card">
            <div class="card-title">Top Benchmark PSNR</div>
            <div class="card-value">{"None" if not experiments else f"{latest_psnr:.2f} dB"}</div>
            <div style="color: #34D399; font-size: 0.8rem; margin-top: 4px;">Validation Metric</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col4:
    st.markdown(
        f"""
        <div class="aerospace-card">
            <div class="card-title">Completed Runs</div>
            <div class="card-value">{len(experiments)}</div>
            <div style="color: #A78BFA; font-size: 0.8rem; margin-top: 4px;">Experiment Records</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("### 🎯 Research Objectives & ISRO Relevance")
col_left, col_right = st.columns([1.6, 1.0])

with col_left:
    st.markdown(
        """
        This project investigates cross-sensor deep learning image translation to convert **Optical Satellite Imagery (Sentinel-2 MSI)** 
        into **Synthetic SAR-like imagery (Sentinel-1 C-Band)**.
        
        #### Key Remote Sensing Motivation:
        - **All-Weather Earth Observation:** Optical sensors are obstructed by cloud cover, haze, and darkness. SAR sensors penetrate clouds and operate independently of solar illumination.
        - **Cross-Modal Data Fusion:** Learning cross-sensor structural representations bridges the spectral gap between optical surface reflectance and microwave backscatter.
        - **Target Applications for ISRO / URSC:**
          - 🌊 Rapid flood extent estimation beneath cloudy monsoon skies
          - 🌾 Agricultural crop vigor and phenological tracking
          - 🏙️ Urban infrastructure and port monitoring
          - 🗺️ Multi-temporal land-cover and disaster response mapping
        """
    )

with col_right:
    st.markdown("#### 📡 Candidate Datasets Status")
    datasets = ["SEN1-2", "QXS-SAROPT", "SAR2Opt"]
    for d_name in datasets:
        adapter = DatasetRegistry.get_adapter(d_name)
        cached_count = len(adapter.get_cached_pairs())
        status_info = adapter.check_availability()
        
        badge_cls = "badge-verified" if status_info["verified_accessible"] else "badge-warning"
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: #F8FAFC;">{d_name}</strong>
                    <span class="{badge_cls}">{status_info['status']}</span>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">
                    Cached Pairs: <strong>{cached_count}</strong> | {status_info.get('message', '')[:65]}...
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Latest Experiment Box
st.markdown("### 🔬 Latest Model & Experiment Status")
if experiments:
    latest = experiments[0]
    st.success(
        f"**Latest Run [{latest.get('experiment_id')}]:** Model `{latest.get('model_architecture')}` trained on `{latest.get('dataset')}`. "
        f"Best Val PSNR: **{latest.get('best_val_psnr', 0.0):.2f} dB** across {latest.get('epochs_trained')} epochs. "
        f"Checkpoint: `{Path(latest.get('best_checkpoint', '')).name}`"
    )
else:
    st.info("ℹ️ No model has been trained yet. Visit the **Model Training Laboratory** after retrieving verified data pairs in the **Dataset Manager**.")
