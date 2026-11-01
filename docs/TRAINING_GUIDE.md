# Training Guide & Optimization Protocols
**Project:** Cross-Sensor Image Translation (Optical-to-SAR)  
**Target:** ISRO / URSC Research Demonstration  

---

## 1. Hardware Recommendations & Environment Sizing

* **CPU Execution (Default Demo Mode):**
  - Use smaller subsets (e.g. 5–20 pairs) and 2–5 epochs.
  - Takes 1–3 seconds per epoch on modern multi-core x86_64 CPUs.
  - Ideal for rapid demonstration without heating or fan throttling.
* **GPU Execution (NVIDIA CUDA):**
  - Set batch size to $4$ or $8$.
  - Train across 20–50 epochs with 100–500 pairs.

---

## 2. Training Controls on Streamlit Dashboard

1. Navigate to Page 4: **Model Training Laboratory**.
2. Select **Dataset:** `SEN1-2` (ensure pairs are retrieved via Dataset Manager).
3. Set **Training Epochs:** 2–5 for quick demo, 10–20 for full convergence.
4. Set **Batch Size:** 2 (recommended for CPU).
5. Set **L1 Loss Weight ($\lambda_{L1}$):** Default $100.0$.
6. Click **Start Training**:
   - The interactive progress bar updates per epoch.
   - Live Plotly curves display Generator Loss, Discriminator Loss, and Validation PSNR in real time.
   - Checkpoints are saved automatically into `artifacts/checkpoints/`.
   - Results are automatically committed to the permanent experiment history.
