# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""AMD GPU vendor implementation."""

from typing import List

from archspec.gpu.gpu_microarch import GPUMicroarch
from archspec.gpu.vendor import Vendor


class Amd(Vendor):
    """AMD GPU vendor.

    https://devicehunt.com/view/type/pci/vendor/1002
    """

    name: str = "amd"
    pci_code: str = "0x1002"

    def _detect_kernel_driver(self) -> bool:
        """Detect AMD GPUs via kernel driver state.

        TODO: probe appropriate modules for fallback logic.
        """
        return False

    def detect(self) -> bool:
        """Return True if AMD hardware is present.

        First runs the inherited sysfs PCI scan; if that fails to surface
        an AMD device, falls back to checking kernel driver state.
        """
        if super().detect():
            return True
        return self._detect_kernel_driver()

    def info(self) -> List[GPUMicroarch]:
        """Retrieve info for all AMD GPUs.

        TODO: shell out to ``rocm-smi`` or similar and parse its output into
        one :class:`GPUMicroarch` per detected device.
        """
        return []
