"""
Page 3: Data Exploration
Inspection of verified optical-SAR pairs, band extraction, and geospatial metadata.
"""
import streamlit as st
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ui_theme import apply_theme, render_mission_banner
from src.datasets.dataset_registry import DatasetRegistry
from src.preprocessing.image_loader import load_optical_image, load_sar_image
from src.preprocessing.normalization import denormalize_optical, sar_to_display_image
from src.preprocessing.geospatial_processing import extract_geospatial_metadata

st.set_page_config(page_title="Data Exploration | Optical-to-SAR", page_icon="🔍", layout="wide")
apply_theme()
render_mission_banner("Multimodal Data Exploration & Pair Inspection")

dataset_name = st.selectbox("Select Dataset to Explore:", DatasetRegistry.list_datasets(), index=0)
adapter = DatasetRegistry.get_adapter(dataset_name)
cached_pairs = adapter.get_cached_pairs()

if not cached_pairs:
    st.info(
        f"ℹ️ No cached pairs found for `{dataset_name}`. "
        f"Please navigate to the **Dataset Manager** and click 'Retrieve Selected Pairs' to fetch verified real samples."
    )
else:
    st.success(f"Loaded **{len(cached_pairs)}** verified pairs from local cache.")
    
    # Pair Selector
    sample_ids = [p["sample_id"] for p in cached_pairs]
    selected_id = st.selectbox("Select Sample Pair to Inspect:", sample_ids, index=0)
    selected_pair = next(p for p in cached_pairs if p["sample_id"] == selected_id)
    
    opt_path = Path(selected_pair["optical_path"])
    sar_path = Path(selected_pair["sar_path"])
    
    # Load Real Data
    with st.spinner("Loading multimodal imagery..."):
        opt_norm, opt_meta = load_optical_image(opt_path)
        sar_norm, sar_meta = load_sar_image(sar_path)
        geo_meta = extract_geospatial_metadata(opt_path)
        val_result = adapter.validate_sample_pair(opt_path, sar_path)
        
    opt_vis = denormalize_optical(opt_norm)
    sar_vis = sar_to_display_image(sar_norm)
    
    # Display Images Side-by-Side
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown(f"#### 🌍 Optical Image ({opt_meta['format']})")
        st.image(opt_vis, caption=f"True Color Composite RGB (Shape: {opt_vis.shape})", use_container_width=True)
        st.markdown(f"- **Filename:** `{opt_path.name}`")
        st.markdown(f"- **Original Shape:** `{opt_meta.get('raw_shape')}`")
        st.markdown(f"- **Original Dtype:** `{opt_meta.get('raw_dtype')}`")
        
    with col_img2:
        st.markdown(f"#### 📡 SAR Reference ({sar_meta['format']})")
        st.image(sar_vis, caption=f"SAR Visual Backscatter (Shape: {sar_vis.shape})", use_container_width=True)
        st.markdown(f"- **Filename:** `{sar_path.name}`")
        st.markdown(f"- **Original Shape:** `{sar_meta.get('raw_shape')}`")
        st.markdown(f"- **Original Dtype:** `{sar_meta.get('raw_dtype')}`")
        
    st.markdown("---")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("#### ✅ Pair Correspondence & Validation")
        if val_result.get("valid"):
            st.success(f"**Status:** Valid. {val_result.get('reason')}")
        else:
            st.error(f"**Status:** Validation warning: {val_result.get('reason')}")
            
        opt_info = val_result.get("optical", {})
        sar_info = val_result.get("sar", {})
        st.markdown(
            f"""
            - **Optical Value Range:** `[{opt_info.get('min_val', 0):.1f}, {opt_info.get('max_val', 0):.1f}]`
            - **SAR Value Range (dB):** `[{sar_info.get('min_val', 0):.2f}, {sar_info.get('max_val', 0):.2f}]`
            - **NaN / Inf Check:** Optical: `Clean` | SAR: `Clean`
            """
        )
        
    with col_v2:
        st.markdown("#### 🌐 Geospatial Raster Metadata")
        if geo_meta.get("is_geospatial"):
            st.info(f"**Georeferenced Image Detected**")
            st.markdown(f"- **CRS:** `{geo_meta.get('crs')}`")
            st.markdown(f"- **Resolution:** `{geo_meta.get('resolution')}`")
            st.markdown(f"- **Bounding Box:** `{geo_meta.get('bounds')}`")
        else:
            st.caption("Standard image format or local coordinate space (no embedded projection tags).")
