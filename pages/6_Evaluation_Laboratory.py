"""
Page 6: Evaluation Laboratory
Comprehensive quantitative benchmarking (PSNR, SSIM, MAE, MSE), residual error maps, and cross-dataset testing.
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ui_theme import apply_theme, render_mission_banner
from src.training.checkpoints import list_checkpoints
from src.datasets.dataset_registry import DatasetRegistry
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.comparison import create_comparison_figure

st.set_page_config(page_title="Evaluation Laboratory | Optical-to-SAR", page_icon="🧪", layout="wide")
apply_theme()
render_mission_banner("Evaluation Laboratory & Residual Error Analysis")

checkpoints = list_checkpoints()

if not checkpoints:
    st.info("ℹ️ No model checkpoints found for evaluation. Please train a baseline in the **Model Training Laboratory**.")
else:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        ckpt_name = st.selectbox("Select Model Checkpoint:", [c["filename"] for c in checkpoints])
        selected_ckpt = next(c for c in checkpoints if c["filename"] == ckpt_name)
    with col_c2:
        eval_dataset = st.selectbox("Select Evaluation Dataset:", DatasetRegistry.list_datasets(), index=0)
        
    adapter = DatasetRegistry.get_adapter(eval_dataset)
    cached_pairs = adapter.get_cached_pairs()
    
    if not cached_pairs:
        st.warning(f"No cached pairs found for `{eval_dataset}`. Please retrieve pairs in the Dataset Manager first.")
    else:
        st.markdown(f"**Found {len(cached_pairs)} verified pairs for evaluation.**")
        
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        with col_ctrl1:
            eval_count = st.slider("Number of Test Pairs to Evaluate:", min_value=1, max_value=min(50, len(cached_pairs)), value=min(10, len(cached_pairs)))
            run_eval_btn = st.button("🧪 Run Full Batch Evaluation", type="primary", use_container_width=True)
            
        if run_eval_btn:
            with st.spinner("Evaluating model across held-out satellite pairs..."):
                evaluator = ModelEvaluator(checkpoint_path=Path(selected_ckpt["path"]))
                results = evaluator.evaluate_pairs(cached_pairs[:eval_count])
                st.session_state["eval_results"] = results
                
        if "eval_results" in st.session_state:
            res = st.session_state["eval_results"]
            agg = res["aggregate_metrics"]
            
            st.markdown("### 📈 Aggregate Quantitative Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PSNR (Peak Signal-to-Noise)", f"{agg['psnr_mean']} dB", f"± {agg['psnr_std']}")
            m2.metric("SSIM (Structural Similarity)", f"{agg['ssim_mean']}", f"± {agg['ssim_std']}")
            m3.metric("MAE (Mean Absolute Error)", f"{agg['mae_mean']}")
            m4.metric("MSE (Mean Squared Error)", f"{agg['mse_mean']}")
            
            st.caption(f"Evaluated **{res['sample_count']}** pairs in **{res['evaluation_time_sec']}s** using checkpoint `{res['checkpoint']}`.")
            
            st.markdown("---")
            st.markdown("### 🔬 Individual Test Case Inspection & Error Maps")
            
            samples = res.get("samples", [])
            if samples:
                sample_select = st.selectbox("Select Test Sample:", [s["sample_id"] for s in samples])
                sample = next(s for s in samples if s["sample_id"] == sample_select)
                
                # Interactive 4-panel comparison figure
                fig = create_comparison_figure(
                    optical_rgb=sample["optical_visual"],
                    reference_sar=sample["reference_sar_visual"],
                    synthetic_sar=sample["synthetic_sar_visual"],
                    metrics=sample["metrics"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown(
                        f"""
                        **Sample ID:** `{sample['sample_id']}`  
                        - **PSNR:** `{sample['metrics']['psnr']} dB`  
                        - **SSIM:** `{sample['metrics']['ssim']}`
                        """
                    )
                with col_d2:
                    st.markdown(
                        f"""
                        - **MAE:** `{sample['metrics']['mae']}`  
                        - **MSE:** `{sample['metrics']['mse']}`  
                        - **Interpretation:** The Error Map (Panel 4) highlights high-frequency discrepancy areas such as building edges or fine radar speckle.
                        """
                    )
