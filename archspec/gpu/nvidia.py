# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""NVIDIA GPU vendor implementation."""

import os
import subprocess
from typing import List

from archspec.gpu.gpu_microarch import GPUMicroarch
from archspec.gpu.vendor import Vendor


class Nvidia(Vendor):
    """NVIDIA GPU vendor.

    https://devicehunt.com/view/type/pci/vendor/10DE
    """

    name: str = "nvidia"
    pci_code: str = "0x10de"

    #: Path to the NVIDIA driver's procfs directory
    PROC_NVIDIA_GPUS: str = "/proc/driver/nvidia/gpus"

    def detect(self) -> bool:
        """Return True if NVIDIA hardware is present."""
        if super().detect():
            return True

        # if sysfs detection fails, fall back to nvidia proc detection
        try:
            return os.path.isdir(self.PROC_NVIDIA_GPUS) and bool(
                os.listdir(self.PROC_NVIDIA_GPUS)
            )
        except OSError:
            return False

    def info(self) -> List[GPUMicroarch]:
        """Retrieve info for all NVIDIA GPUs using nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=gpu_name,driver_version,pci.device_id",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError:
            raise RuntimeError("NVIDIA GPU detected but nvidia-smi is not installed")
        except subprocess.CalledProcessError:
            return []

        gpus: List[GPUMicroarch] = []
        for line in result.stdout.strip().splitlines():
            brand_string, driver_version, pci_device_id = (
                p.strip() for p in line.split(",")
            )
            hex_digits = pci_device_id[2:]
            gpus.append(
                GPUMicroarch(
                    vendor=self.name,
                    brand_string=brand_string,
                    driver_version=driver_version,
                    component_pci_code=f"0x{hex_digits[:4]}".lower(),
                    vendor_pci_code=f"0x{hex_digits[4:]}".lower(),
                )
            )
        return gpus
