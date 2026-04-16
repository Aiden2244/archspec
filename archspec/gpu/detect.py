# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Detection of GPU microarchitectures"""

import collections
import functools
import os
import platform
from typing import Callable, Dict, List, Set

from archspec.gpu.gpu_microarch import GPUMicroarch

#: PCI class codes for GPU devices
#: https://admin.pci-ids.ucw.cz/read/PD/03
GPU_PCI_CLASSES = ("0x030000", "0x030200")

#: Mapping from PCI vendor IDs to vendor names
#: https://devicehunt.com/view/type/pci/vendor/10DE -- NVIDIA
#: https://devicehunt.com/view/type/pci/vendor/1002 -- AMD
#: https://devicehunt.com/view/type/pci/vendor/8086 -- INTEL
GPU_VENDORS = {
    "0x10de": "nvidia",
    "0x1002": "amd",
    "0x8086": "intel",
}

#: Path to the sysfs PCI devices directory
SYSFS_PCI_DEVICES = "/sys/bus/pci/devices"

#: Mapping from operating systems to chain of commands
#: to obtain a list of raw info on the current gpus
INFO_FACTORY: Dict[str, List[Callable]] = collections.defaultdict(list)


def detection(operating_system: str):
    """Decorator to mark functions that are meant to return raw information on detected GPUs.

    Args:
        operating_system: operating system where this function can be used.
    """

    def decorator(factory):
        INFO_FACTORY[operating_system].append(factory)
        return factory

    return decorator


def _read_sysfs_file(path: str) -> str:
    """Read and strip the contents of a sysfs file."""
    with open(path) as f:
        return f.read().strip()


@detection(operating_system="Linux")
def _detect_gpu_vendors_linux() -> Set[str]:
    """Detect which GPU vendors are present by scanning sysfs PCI devices on Linux.

    Iterates over ``/sys/bus/pci/devices/`` and filters for devices whose PCI class
    indicates a VGA controller (``0x0300``) or 3D controller (``0x0302``).

    Returns:
        A set of vendor names (e.g. ``{"nvidia", "intel"}``) present on the system.
    """
    vendors: Set[str] = set()

    if not os.path.isdir(SYSFS_PCI_DEVICES):
        return vendors

    for entry in os.listdir(SYSFS_PCI_DEVICES):
        device_dir = os.path.join(SYSFS_PCI_DEVICES, entry)
        if not os.path.isdir(device_dir):
            continue

        class_path = os.path.join(device_dir, "class")
        if not os.path.exists(class_path):
            continue

        class_code = _read_sysfs_file(class_path)
        if class_code not in GPU_PCI_CLASSES:
            continue

        vendor_id = _read_sysfs_file(os.path.join(device_dir, "vendor"))
        vendor_name = GPU_VENDORS.get(vendor_id)
        if vendor_name is not None:
            vendors.add(vendor_name)

    return vendors


def _nvidia_info() -> List[GPUMicroarch]:
    """Retrieve info for all NVIDIA GPUs using nvidia-smi."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=gpu_name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "NVIDIA GPU detected but nvidia-smi is not installed"
        )
    except subprocess.CalledProcessError:
        return []

    gpus: List[GPUMicroarch] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpus.append(
                GPUMicroarch(
                    brand_string=parts[0],
                    vendor="nvidia",
                    driver_version=parts[1],
                    vendor_pci_code="0x10de",
                )
            )
    return gpus


def _amd_info() -> List[GPUMicroarch]:
    """Retrieve info for all AMD GPUs.

    TODO: Use rocm-smi or similar to query device details.
    """
    return []


def _intel_info() -> List[GPUMicroarch]:
    """Retrieve info for all Intel GPUs.

    TODO: Use appropriate tooling to query device details.
    """
    return []


#: Mapping from vendor names to detail-fetching functions
_VENDOR_DETAIL_FN: Dict[str, Callable] = {
    "nvidia": _nvidia_info,
    "amd": _amd_info,
    "intel": _intel_info,
}


@functools.lru_cache(maxsize=None)
def host() -> List[GPUMicroarch]:
    """Detects the GPUs on the host system and returns information about them.

    Stage 1 determines which GPU vendors are present via OS-specific detection
    (sysfs on Linux). Stage 2 calls each present vendor's CLI tool to retrieve
    detailed information for all GPUs of that vendor.

    Returns:
        A list of GPUMicroarch objects, one per detected GPU.
    """
    vendors: Set[str] = set()
    for factory in INFO_FACTORY[platform.system()]:
        try:
            vendors = factory()
            break
        except Exception:
            continue

    results: List[GPUMicroarch] = []
    for vendor in vendors:
        detail_fn = _VENDOR_DETAIL_FN.get(vendor)
        if detail_fn is not None:
            results.extend(detail_fn())

    return results
