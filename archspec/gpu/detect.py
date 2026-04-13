# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Detection of GPU microarchitectures"""

import collections
import functools
import os
import platform
from typing import Any, Callable, Dict, List

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
def _detect_gpus_linux() -> List[Dict[str, str]]:
    """Detect GPUs by scanning sysfs PCI devices on Linux.

    Iterates over ``/sys/bus/pci/devices/`` and filters for devices whose PCI class
    indicates a VGA controller (``0x0300``) or 3D controller (``0x0302``). For each
    matching device, reads the vendor and device ID to identify the GPU.

    Returns:
        A list of dicts, each containing ``pci_address``, ``vendor``, and ``device_id``.
    """
    gpus: List[Dict[str, str]] = []

    if not os.path.isdir(SYSFS_PCI_DEVICES):
        return gpus

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
        if vendor_name is None:
            continue

        device_id = _read_sysfs_file(os.path.join(device_dir, "device"))

        gpus.append(
            {
                "pci_address": entry,
                "vendor": vendor_name,
                "device_id": device_id,
            }
        )

    return gpus


def _nvidia_info(pci_address: str, device_id: str) -> Dict[str, Any]:
    """Retrieve detailed GPU info for an NVIDIA device.

    TODO: Use nvidia-smi to query compute capability, driver version, etc.
    """
    return {"vendor": "nvidia", "pci_address": pci_address, "device_id": device_id}


def _amd_info(pci_address: str, device_id: str) -> Dict[str, Any]:
    """Retrieve detailed GPU info for an AMD device.

    TODO: Use rocm-smi or similar to query device details.
    """
    return {"vendor": "amd", "pci_address": pci_address, "device_id": device_id}


def _intel_info(pci_address: str, device_id: str) -> Dict[str, Any]:
    """Retrieve detailed GPU info for an Intel device.

    TODO: Use appropriate tooling to query device details.
    """
    return {"vendor": "intel", "pci_address": pci_address, "device_id": device_id}


#: Mapping from vendor names to detail-fetching functions
_VENDOR_DETAIL_FN: Dict[str, Callable] = {
    "nvidia": _nvidia_info,
    "amd": _amd_info,
    "intel": _intel_info,
}


@functools.lru_cache(maxsize=None)
def host() -> List[Dict[str, Any]]:
    """Detects the GPUs on the host system and returns information about them.

    Stage 1 identifies GPUs via OS-specific detection (sysfs on Linux).
    Stage 2 calls vendor-specific tooling for detailed information.

    Returns:
        A list of dicts, each describing a detected GPU.
    """
    for factory in INFO_FACTORY[platform.system()]:
        try:
            raw_gpus = factory()
        except Exception:
            continue

        results: List[Dict[str, Any]] = []
        for gpu in raw_gpus:
            detail_fn = _VENDOR_DETAIL_FN.get(gpu["vendor"])
            if detail_fn is not None:
                results.append(detail_fn(gpu["pci_address"], gpu["device_id"]))
            else:
                results.append(gpu)

        return results

    return []
