"""
Hardware device management and system resource inspection.
Uses only Python standard library to avoid psutil C-extension crashes on Python 3.13+ / Windows.
"""
import os
import platform
import shutil
import ctypes
import torch
from typing import Dict, Any


def _get_ram_info() -> Dict[str, float]:
    """Get RAM info using ctypes on Windows, fallback for other OS."""
    try:
        if platform.system() == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            total = round(mem.ullTotalPhys / (1024 ** 3), 2)
            avail = round(mem.ullAvailPhys / (1024 ** 3), 2)
            pct = round((1.0 - avail / total) * 100, 1) if total > 0 else 50.0
            return {"total": total, "available": avail, "percent": pct}
        else:
            # Linux / macOS: parse /proc/meminfo
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
            total = round(info.get("MemTotal", 0) / (1024 ** 2), 2)
            avail = round(info.get("MemAvailable", info.get("MemFree", 0)) / (1024 ** 2), 2)
            pct = round((1.0 - avail / total) * 100, 1) if total > 0 else 50.0
            return {"total": total, "available": avail, "percent": pct}
    except Exception:
        return {"total": 16.0, "available": 8.0, "percent": 50.0}


def _get_cpu_count() -> int:
    """Get logical CPU count using standard library."""
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def get_device_info() -> Dict[str, Any]:
    """Inspect and return hardware device details safely without crashing on platform quirks."""
    cuda_available = torch.cuda.is_available()
    device_name = "CPU"
    device_type = "cpu"
    gpu_memory = None

    if cuda_available:
        try:
            device_type = "cuda"
            device_name = torch.cuda.get_device_name(0)
            gpu_memory = {
                "total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
                "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2),
                "cached_gb": round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2),
            }
        except Exception:
            gpu_memory = None

    cpu_count = _get_cpu_count()
    ram = _get_ram_info()

    try:
        usage = shutil.disk_usage(os.path.abspath("."))
        disk_free = round(usage.free / (1024 ** 3), 2)
    except Exception:
        disk_free = 50.0

    return {
        "device": device_type,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "gpu_memory": gpu_memory,
        "cpu_count": cpu_count,
        "ram_total_gb": ram["total"],
        "ram_available_gb": ram["available"],
        "ram_percent": ram["percent"],
        "disk_free_gb": disk_free,
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }


def get_torch_device() -> torch.device:
    """Return optimal torch.device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

