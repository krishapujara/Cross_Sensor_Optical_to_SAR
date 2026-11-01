# API & Data Flow Specification
**Project:** Cross-Sensor Image Translation (Optical-to-SAR)  
**Target:** ISRO / URSC Research Demonstration  

---

## 1. System Data Flow Diagram

```
[ Remote Endpoint / Hugging Face Mirror ]
                     │
                     ▼ (HTTP Range Request / Stream)
          [ SEN12DatasetAdapter ]
                     │
                     ▼ (Validate: channels, NaN/Inf, size)
             [ Local Cache ]
                     │
                     ▼
          [ PairedSatelliteDataset ]
                     │ (Split: Train / Val / Test)
                     ▼
             [ DataLoaders ]
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
[ Pix2PixTrainer ]      [ ModelEvaluator ]
         │                       │
         ▼ (Save .pt)            ▼
[ Checkpoint Manager ]  [ Comparison Figure ]
         │                       │
         ▼                       ▼
[ OpticalToSARPredictor ] [ Streamlit UI View ]
         │
         ├──► Visual PNG Download
         └──► Calibrated GeoTIFF Download
```

---

## 2. Core Python Service Interfaces

### 2.1 `BaseDatasetAdapter`
* `check_availability() -> Dict[str, Any]`
* `get_metadata() -> Dict[str, Any]`
* `retrieve_samples(count: int, split: str) -> Dict[str, Any]`
* `validate_sample_pair(opt_path: Path, sar_path: Path) -> Dict[str, Any]`

### 2.2 `OpticalToSARPredictor`
* `predict(input_image: Union[Path, bytes], img_size: int = 256) -> Dict[str, Any]`
  Returns visual RGB array, synthetic SAR visual array, calibrated dB float array, timing metrics, and export byte streams.

### 2.3 `Pix2PixTrainer`
* `train(epochs: int, checkpoint_freq: int, epoch_callback: Callable) -> Dict[str, Any]`
