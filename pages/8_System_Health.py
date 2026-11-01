"""
Page 8: System Health & Diagnostic Telemetry
Environment verification, hardware diagnostics, network endpoint connectivity, and storage telemetry.
"""
import streamlit as st
import sys
import platform
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ui_theme import apply_theme, render_mission_banner
from src.utils.device import get_device_info
from src.datasets.dataset_registry import DatasetRegistry
from src.storage.cache_manager import CacheManager
from src.training.checkpoints import list_checkpoints

st.set_page_config(page_title="System Health | Optical-to-SAR", page_icon="🖥️", layout="wide")
apply_theme()
render_mission_banner("System Health & Diagnostic Telemetry")

device_info = get_device_info()
cache_mgr = CacheManager()
cache_stats = cache_mgr.get_cache_stats()
checkpoints = list_checkpoints()

col1, col2, col3, col4 = st.columns(4)
col1.metric("PyTorch Engine", f"v{torch.__version__}")
col2.metric("Python Runtime", f"v{platform.python_version()}")
col3.metric("RAM Available", f"{device_info['ram_available_gb']} GB / {device_info['ram_total_gb']} GB")
col4.metric("Disk Free Space", f"{device_info['disk_free_gb']} GB")

st.markdown("### ⚙️ Compute Hardware Diagnostics")
col_hw1, col_hw2 = st.columns(2)

with col_hw1:
    st.markdown("#### Processor & Operating System")
    st.markdown(f"- **Operating System:** `{device_info['os']}`")
    st.markdown(f"- **CPU Architecture:** `{platform.processor() or 'x86_64'}`")
    st.markdown(f"- **CPU Core Count (Logical):** `{device_info['cpu_count']}`")
    st.markdown(f"- **System RAM Utilization:** `{device_info['ram_percent']}%`")

with col_hw2:
    st.markdown("#### GPU & Acceleration State")
    if device_info["cuda_available"]:
        gpu_m = device_info["gpu_memory"]
        st.success(f"**NVIDIA CUDA Active:** `{device_info['device_name']}`")
        st.markdown(f"- **Total VRAM:** `{gpu_m['total_gb']} GB`")
        st.markdown(f"- **Allocated VRAM:** `{gpu_m['allocated_gb']} GB`")
        st.markdown(f"- **Reserved VRAM:** `{gpu_m['cached_gb']} GB`")
    else:
        st.info("ℹ️ **CUDA Hardware Not Detected:** Running on Host CPU.")
        st.caption("Standard research prototyping and smoke testing run reliably on modern multi-core CPUs.")
        
st.markdown("---")
st.markdown("### 🌐 Remote Dataset Connectivity & Latency Telemetry")

test_col1, test_col2, test_col3 = st.columns(3)
datasets = ["SEN1-2", "QXS-SAROPT", "SAR2Opt"]

for idx, col in enumerate([test_col1, test_col2, test_col3]):
    d_name = datasets[idx]
    adapter = DatasetRegistry.get_adapter(d_name)
    status_info = adapter.check_availability()
    with col:
        st.markdown(f"**{d_name}**")
        if status_info["verified_accessible"]:
            st.success(f"Status: {status_info['status']}")
            st.caption(f"Latency: {status_info.get('latency_ms')} ms")
        elif status_info.get("auth_required"):
            st.warning(f"Status: {status_info['status']}")
            st.caption("Requires questionnaire application")
        else:
            st.info(f"Status: {status_info['status']}")
            st.caption(status_info.get("message", "")[:45] + "...")

st.markdown("---")
st.markdown("### 🗄️ Model Checkpoints & Storage")
st.markdown(f"- **Checkpoints Stored on Disk:** `{len(checkpoints)}`")
st.markdown(f"- **Total Cache Size:** `{cache_stats['size_mb']} MB` ({cache_stats['pairs_count']} verified pairs)")
st.caption("Notice: System metrics represent the local host machine or deployment runner, not onboard ISRO satellite payload hardware.")
