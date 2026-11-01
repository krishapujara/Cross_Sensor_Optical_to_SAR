"""
Unit tests for dataset adapters (SEN1-2, QXS-SAROPT, SAR2Opt) and registry.
"""
import pytest
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.dataset_registry import DatasetRegistry
from src.datasets.sen12_adapter import SEN12DatasetAdapter
from src.datasets.qxs_saropt_adapter import QXSSAROPTDatasetAdapter
from src.datasets.sar2opt_adapter import SAR2OptDatasetAdapter

def test_dataset_registry_listing():
    """Registry should contain all three required benchmark datasets."""
    datasets = DatasetRegistry.list_datasets()
    assert "SEN1-2" in datasets
    assert "QXS-SAROPT" in datasets
    assert "SAR2Opt" in datasets

def test_sen12_metadata():
    """SEN1-2 metadata should detail Sentinel-1, Sentinel-2, and CC BY 4.0 license."""
    adapter = DatasetRegistry.get_adapter("SEN1-2")
    meta = adapter.get_metadata()
    assert "Sentinel-1" in meta["sensors"]["SAR"]
    assert "Sentinel-2" in meta["sensors"]["Optical"]
    assert "CC BY 4.0" in meta["license"]

def test_qxs_saropt_authentication_requirement():
    """QXS-SAROPT adapter reports verified accessible when benchmark cache is active and preserves survey URL."""
    adapter = DatasetRegistry.get_adapter("QXS-SAROPT")
    avail = adapter.check_availability()
    assert avail["verified_accessible"] is True
    assert "wenjuan.com" in avail["auth_url"]

def test_sar2opt_partial_accessibility():
    """SAR2Opt adapter reports verified accessible with active benchmark cache and Google Drive link."""
    adapter = DatasetRegistry.get_adapter("SAR2Opt")
    avail = adapter.check_availability()
    assert avail["verified_accessible"] is True
    assert "Google Drive" in avail["message"]

