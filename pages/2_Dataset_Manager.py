"""
Page 2: Dataset Manager
Remote Dataset Selective Retrieval, Inspection, and Local Cache Storage.
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ui_theme import apply_theme, render_mission_banner
from src.datasets.dataset_registry import DatasetRegistry
from src.storage.cache_manager import CacheManager

st.set_page_config(page_title="Dataset Manager | Optical-to-SAR", page_icon="🗄️", layout="wide")
apply_theme()
render_mission_banner("Dataset Manager & Remote Access Controller")

cache_mgr = CacheManager()

# Dataset Selector Tabs
tab_select, tab_cache = st.tabs(["📡 Dataset Selection & Retrieval", "💾 Local Cache Storage Manager"])

with tab_select:
    dataset_name = st.selectbox(
        "Select Earth Observation Benchmark Dataset:",
        DatasetRegistry.list_datasets(),
        index=0,
        help="Select between SEN1-2 (Sentinel-1/2), QXS-SAROPT (GaoFen-3/Google Earth), or SAR2Opt."
    )
    
    adapter = DatasetRegistry.get_adapter(dataset_name)
    status_info = adapter.check_availability()
    meta = adapter.get_metadata()
    cache_stats = cache_mgr.get_cache_stats(dataset_name)
    
    # Status Alert
    if status_info["verified_accessible"]:
        st.success(f"✅ **{status_info['status']}** — {status_info['message']} (Probe Latency: {status_info.get('latency_ms')} ms)")
    else:
        st.warning(f"⚠️ **{status_info['status']}** — {status_info['message']}")
        
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown("#### 🛰️ Verified Sensor & Data Architecture")
        for k, v in meta.items():
            if k not in ["name", "license", "source_citation"]:
                st.markdown(f"- **{k.replace('_', ' ').title()}:** `{v}`")
                
    with col_meta2:
        st.markdown("#### 📜 License, Access & Provenance")
        st.markdown(f"- **License:** `{meta.get('license')}`")
        st.markdown(f"- **Streaming / Range Requests:** `{'Supported' if status_info.get('streaming_supported') else 'Not Supported / Monolithic'}`")
        st.markdown(f"- **Authentication Requirement:** `{'Required (Credentials/Approval)' if status_info.get('auth_required') else 'None (Anonymous Direct Access)'}`")
        st.markdown(f"- **Cached Pairs on Local Disk:** **{cache_stats['pairs_count']}** ({cache_stats['size_mb']} MB)")
        
    st.markdown("---")
    st.markdown("### 📥 Selective Sample Retrieval")
    st.caption("Avoids downloading the full multi-hundred gigabyte dataset. Retrieves only selected pairs.")
    
    col_ret1, col_ret2, col_ret3 = st.columns([1, 1, 1.5])
    with col_ret1:
        subset_option = st.selectbox(
            "Select Subset Size:",
            ["10 pairs (Smoke Test)", "25 pairs", "50 pairs", "100 pairs", "Custom"],
            index=0
        )
        if subset_option == "Custom":
            num_pairs = st.number_input("Custom Pair Count:", min_value=1, max_value=1000, value=20)
        else:
            num_pairs = int(subset_option.split()[0])
            
    with col_ret2:
        split_target = st.selectbox("Assign to Split:", ["train", "validation", "test"], index=0)
        
    with col_ret3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        retrieve_btn = st.button("🚀 Retrieve Selected Pairs", use_container_width=True)
        
    if retrieve_btn:
        if not status_info["verified_accessible"] and dataset_name != "SEN1-2":
            st.error(
                f"Cannot stream automatically: {status_info['message']} "
                f"Please place local pairs into `{adapter.cache_dir}` or use SEN1-2."
            )
        else:
            progress_bar = st.progress(0, text="Initializing selective retrieval...")
            status_text = st.empty()
            
            def on_progress(current, total):
                progress_bar.progress(current / total, text=f"Downloading & Validating Pair {current}/{total}...")
                
            with st.spinner("Downloading and validating satellite image pairs..."):
                result = adapter.retrieve_samples(count=num_pairs, split=split_target, progress_callback=on_progress)
                progress_bar.progress(1.0, text="Retrieval completed!")
                
            st.success(
                f"🎉 Successfully retrieved and verified **{result['count_retrieved']}** pairs. "
                f"Failed: {result['count_failed']}. Cached in: `{result['cached_dir']}`"
            )
            
    # Validation Button
    st.markdown("---")
    col_val1, col_val2 = st.columns([1, 2])
    with col_val1:
        val_all_btn = st.button("🔍 Validate Cached Pairs Integrity")
    with col_val2:
        if val_all_btn:
            cached_pairs = adapter.get_cached_pairs()
            if not cached_pairs:
                st.warning("No cached pairs found to validate.")
            else:
                valid_count = 0
                for p in cached_pairs:
                    res = adapter.validate_sample_pair(Path(p["optical_path"]), Path(p["sar_path"]))
                    if res.get("valid"):
                        valid_count += 1
                st.info(f"Integrity Check: **{valid_count}/{len(cached_pairs)}** cached pairs are completely valid and clean.")

with tab_cache:
    st.markdown("### 💾 Storage Allocation & Cache Controller")
    all_stats = cache_mgr.get_cache_stats()
    
    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("Total Cached Files", all_stats["file_count"])
    col_c2.metric("Total Cached Size", f"{all_stats['size_mb']} MB")
    col_c3.metric("Verified Stored Pairs", all_stats["pairs_count"])
    
    st.markdown(f"**Cache Storage Directory:** `{all_stats['path']}`")
    
    st.markdown("#### Per-Dataset Cache Breakdown")
    for d_name in DatasetRegistry.list_datasets():
        ds_stats = cache_mgr.get_cache_stats(d_name)
        st.markdown(f"- **{d_name}:** {ds_stats['pairs_count']} pairs | {ds_stats['size_mb']} MB | Path: `{ds_stats['path']}`")
        
    st.markdown("---")
    st.markdown("#### Clear Local Cache")
    clear_target = st.selectbox("Select Target Cache to Clear:", ["Entire Cache"] + DatasetRegistry.list_datasets())
    confirm_clear = st.checkbox("Confirm cache deletion (cannot be undone)")
    
    if st.button("🗑️ Clear Cache", type="secondary"):
        if not confirm_clear:
            st.warning("Please check the confirmation box before clearing cache.")
        else:
            target = None if clear_target == "Entire Cache" else clear_target
            success = cache_mgr.clear_cache(target)
            if success:
                st.success(f"Cache cleared successfully for: {clear_target}")
                st.rerun()
            else:
                st.error("Error occurred while clearing cache.")
