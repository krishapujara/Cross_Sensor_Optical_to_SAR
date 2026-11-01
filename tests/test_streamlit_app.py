"""
Smoke tests for Streamlit application pages and navigation integrity.
"""
import pytest
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_page_files_exist():
    """All 8 required application pages must exist in the pages/ folder."""
    pages_dir = Path(__file__).resolve().parent.parent / "pages"
    required_pages = [
        "1_Mission_Overview.py",
        "2_Dataset_Manager.py",
        "3_Data_Exploration.py",
        "4_Model_Training.py",
        "5_Optical_to_SAR_Translation.py",
        "6_Evaluation_Laboratory.py",
        "7_Experiment_History.py",
        "8_System_Health.py"
    ]
    for p in required_pages:
        assert (pages_dir / p).exists(), f"Missing page: {p}"

def test_main_app_exists():
    """Main app.py must exist."""
    app_file = Path(__file__).resolve().parent.parent / "app.py"
    assert app_file.exists()
