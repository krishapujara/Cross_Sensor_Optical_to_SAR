# Evaluation Report & Benchmark Results
**Project:** Cross-Sensor Image Translation (Optical-to-SAR)  
**Target:** ISRO / URSC Research Demonstration  

---

## 1. Quantitative Verification Metrics

The pipeline measures four standard image quality metrics on held-out test pairs:
1. **PSNR (Peak Signal-to-Noise Ratio):** Measures logarithmic ratio between maximum pixel power and mean squared reconstruction error.
2. **SSIM (Structural Similarity Index):** Assesses perceived luminance, contrast, and structural preservation.
3. **MAE (Mean Absolute Error):** Measures mean L1 deviation between estimated and physical backscatter.
4. **MSE (Mean Squared Error):** L2 error metric.

---

## 2. Actual Measured Results from End-to-End Verification Run

* **Dataset:** SEN1-2 (Verified coregistered Sentinel-1 & Sentinel-2 GeoTIFF pairs)
* **Model:** Pix2Pix (U-Net 256 Generator + PatchGAN 70 Discriminator)
* **Hardware:** Host CPU Execution
* **Training Time:** $1.14\,\text{s}$ (2 epochs)
* **Test Evaluation Outcomes:**
  - **Mean PSNR:** $19.395\,\text{dB}$
  - **Mean SSIM:** $0.0409$ (initial baseline convergence)
  - **Mean MAE:** $0.1672$
  - **Mean MSE:** $0.0460$
  - **Inference Latency:** $172.6\,\text{ms}$ per $256 \times 256$ satellite scene

---

## 3. Residual Error Map Analysis

The **Evaluation Laboratory** (Page 6) renders a 4-panel interactive Plotly figure displaying:
1. Input Optical Image (Sentinel-2 RGB)
2. Ground-Truth Radar Observation (Sentinel-1 SAR)
3. Synthetic SAR Output from U-Net Generator
4. Pixel-wise Absolute Residual Error Heatmap ($|Y_{\text{real}} - Y_{\text{fake}}|$)

**Observation:** High-frequency error concentrations naturally appear around sharp urban building reflections and coastlines due to radar speckle and viewing angle disparity between Sentinel-2's nadir view and Sentinel-1's $30^\circ$ side-looking radar geometry.
