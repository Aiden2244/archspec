# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Detection of GPU microarchitectures"""

import collections
import functools
import os
import platform
import subprocess
import json
from typing import Callable, Dict, List, Set, Tuple

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

#: Path to the NVIDIA driver's procfs directory, populated when the
#: kernel module is loaded and bound to at least one GPU
PROC_NVIDIA_GPUS = "/proc/driver/nvidia/gpus"

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


def _scan_sysfs_pci_for_gpus() -> Set[str]:
    """Detect GPU vendors by scanning sysfs PCI devices.

    Iterates over ``/sys/bus/pci/devices/`` and filters for devices whose PCI class
    indicates a VGA controller (``0x0300``) or 3D controller (``0x0302``).

    Returns:
        A set of vendor names present on the system per sysfs.
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


def _detect_nvidia_fallback() -> bool:
    """Detect NVIDIA GPUs via the kernel driver's procfs interface.

    ``/proc/driver/nvidia/gpus/`` is populated by the NVIDIA kernel module
    once it has bound to at least one GPU. This signal is independent of
    PCI topology, so it works in virtualized environments (e.g. vGPU or
    passthrough) where the device may not appear under the expected PCI
    classes in sysfs.

    Returns:
        True if the NVIDIA driver has bound to at least one GPU.
    """
    # TODO remove
    print("DEBUG: nvidia fallback logic dispatched")
    try:
        return os.path.isdir(PROC_NVIDIA_GPUS) and bool(os.listdir(PROC_NVIDIA_GPUS))
    except OSError:
        return False


def _detect_amd_fallback() -> bool:
    """Fallback detection for AMD GPUs.

    TODO: probe the amdgpu/radeon kernel modules and ``/dev/dri`` render nodes.
    """
    return False


def _detect_intel_fallback() -> bool:
    """Fallback detection for Intel GPUs.

    TODO: probe the i915/xe kernel modules and ``/dev/dri`` render nodes.
    """
    return False


#: Mapping from vendor names to fallback detection functions, used when
#: sysfs PCI scanning does not surface a given vendor. Iteration order
#: (NVIDIA -> AMD -> Intel) follows discrete-GPU market share.
_VENDOR_FALLBACK_FN: Dict[str, Callable[[], bool]] = {
    "nvidia": _detect_nvidia_fallback,
    "amd": _detect_amd_fallback,
    "intel": _detect_intel_fallback,
}


@detection(operating_system="Linux")
def _detect_gpu_vendors_linux() -> Set[str]:
    """Detect which GPU vendors are present on Linux.

    First scans sysfs PCI devices. For any vendor not surfaced by the sysfs
    scan, invokes that vendor's fallback detection (e.g. probing kernel
    driver state) to catch environments where the device does not appear
    under the expected PCI classes.

    Returns:
        A set of vendor names (e.g. ``{"nvidia", "intel"}``) present on the system.
    """

    # TODO this is the sysfs scan logic, move to alternate location?
    vendors = _scan_sysfs_pci_for_gpus()

    for vendor, fallback_fn in _VENDOR_FALLBACK_FN.items():
        if vendor in vendors:
            continue
        if fallback_fn():
            vendors.add(vendor)

    return vendors

    # TODO: this bypasses sysfs logic. Determine program flow
    if False:
        return {"nvidia", "intel", "amd"}


def _parse_nvidia_pci_device_id(combined_id: str) -> Tuple[str, str]:
    """Parse a combined PCI device ID into (device, vendor) codes.

    Args:
        combined_id: 10-character hex string from nvidia-smi ``pci.device_id``
            (e.g. ``0x2C0210DE``).

    Returns:
        A tuple of ``(component_pci_code, vendor_pci_code)`` in lowercase
        (e.g. ``("0x2c02", "0x10de")``).

    Raises:
        ValueError: if *combined_id* is not a valid 10-character hex string
            with a ``0x`` prefix.
    """
    if len(combined_id) != 10 or combined_id[:2] != "0x":
        raise ValueError(
            f"invalid PCI device ID: expected 10-character '0x'-prefixed hex string, got {combined_id!r}"
        )

    hex_digits = combined_id[2:]
    return (f"0x{hex_digits[:4]}".lower(), f"0x{hex_digits[4:]}".lower())


def _nvidia_info() -> List[GPUMicroarch]:
    """Retrieve info for all NVIDIA GPUs using nvidia-smi."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=gpu_name,driver_version,pci.device_id",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("NVIDIA GPU detected but nvidia-smi is not installed")
    except subprocess.CalledProcessError:
        return []

    gpus: List[GPUMicroarch] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        # based on nvidia-smi query, parts = ["brand_string", "driver_version", "combined vendor and device pci code"]
        if len(parts) >= 3:
            pci_codes = _parse_nvidia_pci_device_id(parts[2])
            gpus.append(
                GPUMicroarch(
                    vendor="nvidia",
                    brand_string=parts[0],
                    driver_version=parts[1],
                    component_pci_code=pci_codes[0],
                    vendor_pci_code=pci_codes[1],
                )
            )
    return gpus


def _amd_info() -> List[GPUMicroarch]:
    """Retrieve info for all AMD GPUs using rocm-smi."""

    try:
        result = subprocess.run(
            [
                "rocm-smi",
                "--showproductname",
                "--showdriverversion",
                "--showid",
                "--json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("AMD GPU detected but rocm-smi is not installed")
    except subprocess.CalledProcessError:
        return []

    try:
        data = json.loads(result.stdout)
    except ValueError:
        return []

    # AMD's PCI vendor code is not reported by rocm-smi, so derive it from the
    # known vendor mapping the same way the ``vendor`` field is hardcoded.
    vendor_pci_code = next(
        code for code, name in GPU_VENDORS.items() if name == "amd"
    )

    # The driver version is reported once for the whole system rather than
    # per-card, under a top-level "system" entry.
    system_info = data.get("system", {})
    driver_version = system_info.get("Driver version", "")

    gpus: List[GPUMicroarch] = []
    for key, info in data.items():
        if not key.startswith("card"):
            continue

        # Key names vary across rocm-smi versions, so fall back across the
        # known aliases for the marketing name and the PCI device ID.
        brand_string = info.get("Card Series") or info.get("Market Name") or ""
        component_pci_code = info.get("Device ID") or info.get("GPU ID") or ""

        gpus.append(
            GPUMicroarch(
                vendor="amd",
                brand_string=brand_string,
                driver_version=driver_version,
                component_pci_code=component_pci_code.lower(),
                vendor_pci_code=vendor_pci_code,
            )
        )
    return gpus


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
