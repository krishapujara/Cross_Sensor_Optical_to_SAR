"""
Page 4: Model Training Laboratory
Interactive PyTorch training controller with live epoch tracking, Plotly loss curves, and checkpoint saving.
"""
import streamlit as st
import sys
import time
from pathlib import Path
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ui_theme import apply_theme, render_mission_banner
from src.utils.device import get_device_info, get_torch_device
from src.datasets.dataset_registry import DatasetRegistry
from src.preprocessing.patch_extraction import split_dataset, create_dataloaders
from src.models.model_registry import build_pix2pix_models
from src.training.trainer import Pix2PixTrainer
from src.training.experiment_manager import ExperimentManager
from src.config.settings import CHECKPOINTS_DIR

st.set_page_config(page_title="Model Training | Optical-to-SAR", page_icon="⚡", layout="wide")
apply_theme()
render_mission_banner("Model Training Laboratory & Neural Architecture Control")

device_info = get_device_info()
torch_dev = get_torch_device()

# Hardware Warning
if not device_info["cuda_available"]:
    st.warning(
        f"⚠️ **Hardware Notice:** NVIDIA CUDA acceleration is not detected. "
        f"Training will execute on **CPU ({device_info['cpu_count']} cores)**. "
        f"For fast iterations and responsive demonstration, use small subset sizes (e.g. 5–20 pairs) and 2–5 epochs."
    )
else:
    st.success(f"🚀 **GPU Hardware Acceleration Active:** {device_info['device_name']} ({device_info['gpu_memory']['total_gb']} GB VRAM)")

col_cfg1, col_cfg2 = st.columns([1.2, 1.0])

with col_cfg1:
    st.markdown("### ⚙️ Training Hyperparameters")
    dataset_name = st.selectbox("Select Training Dataset:", DatasetRegistry.list_datasets(), index=0)
    adapter = DatasetRegistry.get_adapter(dataset_name)
    cached_pairs = adapter.get_cached_pairs()
    
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        model_arch = st.selectbox("Model Architecture:", ["Pix2Pix (Conditional U-Net + PatchGAN)", "CycleGAN (Unpaired ResNet - Optional)"], index=0)
        epochs = st.slider("Training Epochs:", min_value=1, max_value=50, value=3, step=1)
        batch_size = st.select_slider("Batch Size:", options=[1, 2, 4, 8, 16], value=2)
    with col_h2:
        learning_rate = st.number_input("Learning Rate (Adam):", min_value=1e-5, max_value=1e-2, value=0.0002, format="%.5f")
        lambda_l1 = st.slider("L1 Loss Weight (λ_L1):", min_value=10.0, max_value=200.0, value=100.0, step=10.0)
        checkpoint_freq = st.slider("Checkpoint Frequency (epochs):", min_value=1, max_value=10, value=1)
        
with col_cfg2:
    st.markdown("### 📊 Data Partitioning")
    st.markdown(f"**Available Cached Pairs:** `{len(cached_pairs)}`")
    
    train_ratio = st.slider("Train Split Ratio:", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    val_ratio = 1.0 - train_ratio
    st.caption(f"Split: {int(train_ratio*100)}% Train / {int(val_ratio*100)}% Validation (with held-out evaluation)")
    
    experiment_id = st.text_input("Experiment Run Name:", value=f"pix2pix_{dataset_name.lower()}_run")
    
    if len(cached_pairs) < 2:
        st.error("Insufficient cached pairs! Please retrieve at least 2 pairs in Dataset Manager before training.")
        can_train = False
    else:
        can_train = True

st.markdown("---")
# Action Buttons
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    start_train_btn = st.button("🚀 Start Training", type="primary", disabled=not can_train, use_container_width=True)

# Training Progress Container
progress_placeholder = st.empty()
chart_placeholder = st.empty()
metrics_table_placeholder = st.empty()

if start_train_btn and can_train:
    st.info("Initializing neural network architectures, data loaders, and optimizers...")
    
    # 1. Split data
    split_dict = split_dataset(cached_pairs, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=0.0, group_by_region=False)
    loaders = create_dataloaders(split_dict, batch_size=batch_size, img_size=256)
    
    train_loader = loaders.get("train")
    val_loader = loaders.get("val") or loaders.get("train") # Fallback if small
    
    # 2. Build Models
    gen, disc = build_pix2pix_models(in_channels=3, out_channels=1, device=torch_dev)
    
    # 3. Instantiate Trainer
    trainer = Pix2PixTrainer(
        generator=gen,
        discriminator=disc,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_id=experiment_id,
        lr=learning_rate,
        lambda_l1=lambda_l1,
        checkpoint_dir=CHECKPOINTS_DIR,
        device=torch_dev
    )
    
    # UI live tracking state
    epoch_bar = progress_placeholder.progress(0, text="Training started...")
    
    loss_g_history = []
    loss_d_history = []
    psnr_history = []
    epochs_axis = []
    
    def on_epoch_update(epoch_data):
        ep = epoch_data["epoch"]
        loss_g_history.append(epoch_data["loss_g"])
        loss_d_history.append(epoch_data["loss_d"])
        psnr_history.append(epoch_data["val_psnr"])
        epochs_axis.append(ep)
        
        epoch_bar.progress(ep / epochs, text=f"Epoch {ep}/{epochs} complete — Loss G: {epoch_data['loss_g']:.4f} | Val PSNR: {epoch_data['val_psnr']:.2f} dB")
        
        # Update live chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs_axis, y=loss_g_history, mode="lines+markers", name="Generator Loss (G)", line=dict(color="#38BDF8", width=2)))
        fig.add_trace(go.Scatter(x=epochs_axis, y=loss_d_history, mode="lines+markers", name="Discriminator Loss (D)", line=dict(color="#F87171", width=2)))
        fig.add_trace(go.Scatter(x=epochs_axis, y=psnr_history, mode="lines+markers", name="Val PSNR (dB)", yaxis="y2", line=dict(color="#34D399", width=2, dash="dot")))
        
        fig.update_layout(
            title="Real-Time Convergence & Validation PSNR",
            xaxis=dict(title="Epoch", dtick=1),
            yaxis=dict(title="Loss Value"),
            yaxis2=dict(title="PSNR (dB)", overlaying="y", side="right"),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            font=dict(color="#E2E8F0"),
            margin=dict(l=20, r=20, t=40, b=20),
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
    with st.spinner("Executing genuine PyTorch training and backpropagation..."):
        results = trainer.train(epochs=epochs, checkpoint_freq=checkpoint_freq, epoch_callback=on_epoch_update)
        
    epoch_bar.progress(1.0, text=f"Training finished in {results['total_duration_sec']} seconds!")
    
    # Save experiment
    exp_mgr = ExperimentManager()
    exp_mgr.record_training_run(results, dataset_name=dataset_name, split_info=split_dict)
    
    st.balloons()
    st.success(
        f"🎉 **Training Run Completed Successfully!** "
        f"Total Duration: {results['total_duration_sec']}s. "
        f"Best Checkpoint: `{Path(results['best_checkpoint']).name}`. "
        f"Saved to experiment history."
    )
