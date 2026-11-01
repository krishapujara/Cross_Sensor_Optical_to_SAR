"""
Configuration and settings for Cross-Sensor Optical-to-SAR Translation.
"""
from pathlib import Path
import os

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Ensure runtime directories exist
for path in [CACHE_DIR, EXPERIMENTS_DIR, CHECKPOINTS_DIR, PREDICTIONS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset URLs and Mirrors
HF_SEN12_BASE_URL = "https://huggingface.co/datasets/Sul06/sen12mscr-replica/resolve/main"
QXS_SAROPT_REPO = "https://github.com/yaoxu008/QXS-SAROPT"
SAR2OPT_REPO = "https://github.com/MarsZhaoYT/SAR2Opt-Heterogeneous-Dataset"

# Default Model & Preprocessing Parameters
DEFAULT_IMG_SIZE = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_BETA1 = 0.5
DEFAULT_BETA2 = 0.999
DEFAULT_LAMBDA_L1 = 100.0
DEFAULT_EPOCHS = 10

# Optical & SAR Normalization constants
# S1 SAR backscatter VV/VH in dB typical limits
SAR_MIN_DB = -35.0
SAR_MAX_DB = 0.0

# Supported image file extensions
SUPPORTED_EXTENSIONS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
MAX_UPLOAD_SIZE_MB = 50
