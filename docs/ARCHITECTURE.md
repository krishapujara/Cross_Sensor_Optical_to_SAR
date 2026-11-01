# Application Architecture & Design Specifications
**Project:** Cross-Sensor Image Translation (Optical-to-SAR)  
**Target:** ISRO / URSC Research Demonstration  

---

## 1. Architectural Philosophy

The application follows a **strict separation of concerns**:
* **UI Layer (`app.py`, `pages/`):** Pure presentation, Streamlit widgets, layout cards, reactive state binding, Plotly charts. Contains **zero** raw PyTorch training or HTTP downloading logic.
* **Service & Domain Layer (`src/`):** Reusable, framework-agnostic Python classes and functions covering datasets, preprocessing, neural networks, training, inference, and evaluation.
* **Storage Layer (`data/`, `artifacts/`):** Local cache, experiment records, and serialized model checkpoints.

```
Cross_Sensor_Optical_to_SAR/
├── app.py                         # Central Streamlit entry point & dashboard hub
├── pages/                         # 8 Operational Streamlit pages
│   ├── 1_Mission_Overview.py
│   ├── 2_Dataset_Manager.py
│   ├── 3_Data_Exploration.py
│   ├── 4_Model_Training.py
│   ├── 5_Optical_to_SAR_Translation.py
│   ├── 6_Evaluation_Laboratory.py
│   ├── 7_Experiment_History.py
│   └── 8_System_Health.py
├── src/
│   ├── config/settings.py         # Paths, constants, normalization limits
│   ├── datasets/                  # Abstract adapter interface & dataset connectors
│   ├── preprocessing/             # Multi-band loading, dB normalization, splits
│   ├── models/                    # PyTorch U-Net Generator, PatchGAN, CycleGAN
│   ├── training/                  # Pix2PixTrainer, GAN loss, checkpoint manager
│   ├── inference/                 # Single-image predictor & GeoTIFF export
│   ├── evaluation/                # PSNR, SSIM, MAE, MSE, difference heatmaps
│   ├── storage/                   # Local cache and JSON experiment tracker
│   └── utils/                     # Device diagnostics, logging, UI themes
├── artifacts/
│   └── checkpoints/               # Model weights (.pt) and metadata (.json)
├── data/
│   ├── cache/                     # Retrieved satellite GeoTIFF pairs
│   └── experiments/               # Experiment history records
├── tests/                         # 25 automated pytest tests
└── docs/                          # Complete technical documentation
```

---

## 2. Component Design Patterns

1. **Adapter Pattern (`src/datasets/`):** `BaseDatasetAdapter` enforces a uniform interface (`check_availability()`, `get_metadata()`, `retrieve_samples()`, `validate_sample_pair()`).
2. **Factory Registry Pattern (`src/datasets/dataset_registry.py`):** Enables dynamic runtime lookup of dataset adapters without hardcoding.
3. **Observer / Callback Pattern (`src/training/trainer.py`):** The training loop emits `epoch_callback` events enabling real-time Streamlit UI and Plotly graph updates without blocking the Python runtime.
4. **Resilient Network Decorator (`src/utils/error_handling.py`):** Implements exponential backoff retries for transient remote HTTP queries.
