# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Intel GPU vendor implementation."""

from typing import List

from archspec.gpu.gpu_microarch import GPUMicroarch
from archspec.gpu.vendor import Vendor


class Intel(Vendor):
    """Intel GPU vendor.

    https://devicehunt.com/view/type/pci/vendor/8086
    """

    name: str = "intel"
    pci_code: str = "0x8086"

    def _detect_kernel_driver(self) -> bool:
        """Detect Intel GPUs via kernel driver state.

        TODO: probe appropriate modules for fallback logic.
        """
        return False

    def detect(self) -> bool:
        """Return True if Intel hardware is present.

        First runs the inherited sysfs PCI scan; if that fails to surface
        an Intel device, falls back to checking kernel driver state.
        """
        if super().detect():
            return True
        return self._detect_kernel_driver()

    def info(self) -> List[GPUMicroarch]:
        """Retrieve info for all Intel GPUs.

        TODO: shell out to an appropriate tool (e.g. ``intel_gpu_top``,
        ``xpu-smi``) and parse its output into one :class:`GPUMicroarch`
        per detected device.
        """
        return []
