"""
Page 7: Experiment History
Persistent experiment tracking, comparative leaderboards, and report exports.
"""
import streamlit as st
import sys
import json
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ui_theme import apply_theme, render_mission_banner
from src.storage.experiment_storage import ExperimentStorage

st.set_page_config(page_title="Experiment History | Optical-to-SAR", page_icon="📜", layout="wide")
apply_theme()
render_mission_banner("Experiment Registry & Historical Runs")

storage = ExperimentStorage()
experiments = storage.list_experiments()

if not experiments:
    st.info("ℹ️ No training runs have been recorded yet. Launch an experiment from the **Model Training Laboratory**.")
else:
    st.markdown(f"**Found {len(experiments)} recorded experiment runs.**")
    
    # Leaderboard Table
    table_rows = []
    for exp in experiments:
        m = exp.get("final_metrics", {})
        table_rows.append({
            "Experiment ID": exp.get("experiment_id"),
            "Date": exp.get("timestamp"),
            "Dataset": exp.get("dataset"),
            "Model": exp.get("model_architecture"),
            "Epochs": exp.get("epochs_trained"),
            "PSNR (dB)": m.get("psnr", 0.0),
            "SSIM": m.get("ssim", 0.0),
            "Loss G": m.get("loss_g", 0.0),
            "Loss D": m.get("loss_d", 0.0),
            "Duration (s)": exp.get("total_duration_sec", 0.0)
        })
    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔍 Experiment Deep Dive")
    exp_ids = [e["experiment_id"] for e in experiments]
    selected_exp_id = st.selectbox("Select Experiment ID:", exp_ids)
    selected_exp = next(e for e in experiments if e["experiment_id"] == selected_exp_id)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.json(selected_exp)
    with col_d2:
        # History chart if epochs > 1
        hist = selected_exp.get("history", [])
        if hist:
            ep_df = pd.DataFrame(hist)
            st.markdown("#### Convergence History")
            st.line_chart(ep_df.set_index("epoch")[["loss_g", "loss_d"]])
            st.line_chart(ep_df.set_index("epoch")[["val_psnr"]])
            
    # Export and Cleanup
    st.markdown("---")
    col_exp1, col_exp2 = st.columns([1, 1])
    with col_exp1:
        st.download_button(
            "📥 Export All Experiments (JSON)",
            data=json.dumps(experiments, indent=2),
            file_name="cross_sensor_experiments.json",
            mime="application/json",
            use_container_width=True
        )
    with col_exp2:
        if st.button("🗑️ Delete Selected Experiment", type="secondary", use_container_width=True):
            success = storage.delete_experiment(selected_exp_id)
            if success:
                st.success(f"Deleted experiment {selected_exp_id}")
                st.rerun()
