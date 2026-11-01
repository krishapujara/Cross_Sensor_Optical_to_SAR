# Deployment Guide: Local Host & Streamlit Community Cloud
**Project:** Cross-Sensor Image Translation (Optical-to-SAR)  
**Target:** ISRO / URSC Research Demonstration  

---

## 1. Local Execution (Recommended for Full Training & Evaluation)

### 1.1 Requirements
* Python 3.10+ (Tested on Python 3.13.5)
* Dependencies installed from `requirements.txt`

### 1.2 Start Command
```powershell
cd C:\Users\DELL\.gemini\antigravity-ide\scratch\Cross_Sensor_Optical_to_SAR
streamlit run app.py
```

---

## 2. Streamlit Community Cloud Deployment

### 2.1 Cloud Resource Constraints
* **Memory Limits:** Cloud runners allocate approximately $1\,\text{GB}$ of RAM.
* **Storage Limits:** Ephemeral filesystem (data written to disk is not persistent across restarts).
* **Compute Limits:** Shared CPU without GPU acceleration.

### 2.2 Recommended Cloud Deployment Strategy
1. **Pre-train Locally:** Train the Pix2Pix model locally and save the best checkpoint (`.pt`).
2. **Commit Checkpoint:** Commit the lightweight checkpoint to `artifacts/checkpoints/`.
3. **Deploy Inference & Evaluation:** In the cloud, the application uses the pre-trained checkpoint for instantaneous single-image translation, test pair evaluation, and live demonstrations without triggering memory-intensive backpropagation.
