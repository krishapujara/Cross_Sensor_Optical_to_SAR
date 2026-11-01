"""
Registry and factory for dataset adapters.
"""
from typing import Dict, Type, List
from .base_adapter import BaseDatasetAdapter
from .sen12_adapter import SEN12DatasetAdapter
from .qxs_saropt_adapter import QXSSAROPTDatasetAdapter
from .sar2opt_adapter import SAR2OptDatasetAdapter

class DatasetRegistry:
    """Factory registry for retrieving dataset adapter instances."""
    
    _adapters: Dict[str, Type[BaseDatasetAdapter]] = {
        "SEN1-2": SEN12DatasetAdapter,
        "QXS-SAROPT": QXSSAROPTDatasetAdapter,
        "SAR2Opt": SAR2OptDatasetAdapter
    }
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        return list(cls._adapters.keys())
        
    @classmethod
    def get_adapter(cls, name: str) -> BaseDatasetAdapter:
        adapter_cls = cls._adapters.get(name)
        if not adapter_cls:
            raise ValueError(f"Unknown dataset '{name}'. Available: {list(cls._adapters.keys())}")
        return adapter_cls()
