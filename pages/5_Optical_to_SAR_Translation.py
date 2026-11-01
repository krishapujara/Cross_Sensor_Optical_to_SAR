"""
Page 5: Optical-to-SAR Translation
Single-image and batch inference engine with geospatial metadata preservation and download capabilities.
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
from src.inference.predictor import OpticalToSARPredictor
from src.training.checkpoints import list_checkpoints
from src.datasets.dataset_registry import DatasetRegistry
from src.config.settings import SUPPORTED_EXTENSIONS

st.set_page_config(page_title="Image Translation | Optical-to-SAR", page_icon="\U0001f6f0\ufe0f", layout="wide")
apply_theme()
render_mission_banner("Optical-to-SAR Cross-Sensor Translation Engine")

checkpoints = list_checkpoints()

col_top1, col_top2 = st.columns([1.5, 1.0])
with col_top1:
    st.markdown("### 1. Select Trained Model Checkpoint")
    if not checkpoints:
        st.error("No model checkpoints found. Please train a baseline in the **Model Training Laboratory** first.")
        selected_checkpoint = None
    else:
        ckpt_options = [c["filename"] for c in checkpoints]
        selected_ckpt_name = st.selectbox("Active Checkpoint:", ckpt_options, index=0)
        selected_checkpoint = next(c for c in checkpoints if c["filename"] == selected_ckpt_name)
        st.caption(
            f"Epoch: {selected_checkpoint.get('epoch')} | "
            f"Size: {selected_checkpoint.get('size_mb')} MB | "
            f"Path: `{selected_checkpoint['path']}`"
        )

with col_top2:
    st.markdown("### 2. Choose Input Source")
    input_source = st.radio(
        "Optical Input Mode:",
        ["Upload Optical Satellite File", "Select from Cached Dataset Samples"]
    )

# Image input container
uploaded_file = None
sample_opt_path = None

if input_source == "Upload Optical Satellite File":
    uploaded_file = st.file_uploader(
        "Upload Optical Satellite Image (PNG, JPG, TIFF, GeoTIFF):",
        type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS]
    )
else:
    dataset_name = st.selectbox("Dataset Source:", DatasetRegistry.list_datasets(), index=0)
    adapter = DatasetRegistry.get_adapter(dataset_name)
    cached_pairs = adapter.get_cached_pairs()
    if not cached_pairs:
        st.warning(f"No cached pairs in `{dataset_name}`. Please retrieve pairs in Dataset Manager first.")
    else:
        sample_opt_id = st.selectbox(
            "Select Cached Sample:", [p["sample_id"] for p in cached_pairs]
        )
        selected_pair = next(p for p in cached_pairs if p["sample_id"] == sample_opt_id)
        sample_opt_path = Path(selected_pair["optical_path"])
        st.info(f"Using optical file: `{sample_opt_path.name}`")

st.markdown("---")

# Check if input is ready
has_input = (uploaded_file is not None) or (sample_opt_path is not None)
has_model = selected_checkpoint is not None

if not has_model:
    st.warning("Please train a model checkpoint in **Model Training Laboratory** before running translation.")
elif not has_input:
    st.info("Please upload an optical image or select a cached dataset sample above.")

# Run Translation Button
run_btn = st.button(
    "Run Cross-Sensor Translation",
    type="primary",
    disabled=(not has_model or not has_input),
    use_container_width=False
)

if run_btn and has_model and has_input:
    with st.spinner("Executing neural cross-sensor translation forward pass..."):
        try:
            predictor = OpticalToSARPredictor(checkpoint_path=Path(selected_checkpoint["path"]))

            if uploaded_file:
                input_payload = uploaded_file.getvalue()
            else:
                input_payload = sample_opt_path

            pred_results = predictor.predict(input_payload, img_size=256)

            st.success(
                f"Translation complete in **{pred_results['inference_time_ms']} ms** "
                f"on `{pred_results['device']}`!"
            )

            # ---- Prepare display images: always ensure uint8 RGB ----
            def to_display_rgb(arr: np.ndarray) -> np.ndarray:
                """Convert any numpy array to a displayable uint8 RGB image."""
                arr = np.asarray(arr, dtype=np.float32)
                # Handle (H, W) -> (H, W, 3)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                # Handle (H, W, 1) -> (H, W, 3)
                elif arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = np.concatenate([arr, arr, arr], axis=-1)
                # Ensure RGB (take first 3 channels if more)
                elif arr.ndim == 3 and arr.shape[-1] > 3:
                    arr = arr[..., :3]
                # Normalize to [0, 255] if values are in float range
                if arr.max() <= 1.0 and arr.min() >= -1.0 and arr.max() > 0.0:
                    arr = (arr + 1.0) / 2.0 * 255.0
                elif arr.max() <= 1.0 and arr.min() >= 0.0:
                    arr = arr * 255.0
                return np.clip(arr, 0, 255).astype(np.uint8)

            opt_vis = to_display_rgb(pred_results["optical_visual"])
            sar_vis = to_display_rgb(pred_results["synthetic_sar_visual"])

            # Display Side-by-Side Comparison
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown("#### Input Optical Satellite Image")
                st.image(opt_vis, caption="Sentinel-2 Optical (RGB)", use_container_width=True)

            with col_res2:
                st.markdown("#### Generated Synthetic SAR Image")
                st.image(sar_vis, caption="Synthetic Sentinel-1 SAR Backscatter (Visual)", use_container_width=True)

            # Stats & Geospatial Box
            col_st1, col_st2 = st.columns(2)
            with col_st1:
                st.markdown("#### Estimated SAR Radar Statistics")
                stats = pred_results["stats"]
                st.markdown(
                    f"""
- **Estimated Min Backscatter:** `{stats['min_db']} dB`
- **Estimated Max Backscatter:** `{stats['max_db']} dB`
- **Estimated Mean Backscatter:** `{stats['mean_db']} dB`
- **Estimated Standard Deviation:** `{stats['std_db']} dB`
                    """
                )
            with col_st2:
                st.markdown("#### Geospatial Metadata")
                geo = pred_results["geospatial_metadata"]
                if geo.get("is_geospatial"):
                    st.info(f"**Geospatial CRS:** `{geo.get('crs')}`")
                    st.markdown(f"- **Transform:** `{geo.get('transform')}`")
                    st.markdown(f"- **Bounds:** `{geo.get('bounds')}`")
                else:
                    st.caption("Standard image input — no embedded geospatial coordinate system detected.")

            # Download Buttons
            st.markdown("---")
            st.markdown("#### Export Synthetic SAR Products")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "Download Visual SAR Image (PNG)",
                    data=pred_results["png_bytes"],
                    file_name="synthetic_sar_visual.png",
                    mime="image/png",
                    use_container_width=True
                )
            with col_d2:
                st.download_button(
                    "Download Calibrated SAR Backscatter (GeoTIFF)",
                    data=pred_results["geotiff_bytes"],
                    file_name="synthetic_sar_calibrated.tif",
                    mime="image/tiff",
                    use_container_width=True
                )
        except Exception as e:
            import traceback
            st.error(f"Translation execution error: {str(e)}")
            with st.expander("Full Traceback"):
                st.code(traceback.format_exc())
