# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


class GPUMicroarch:
    """Specific GPU Microarchitecture"""

    def __init__(
        self,
        name: str = "",
        brand_string: str = "",
        vendor: str = "",
        driver_version: str = "",
        vendor_pci_code: str = "",
        component_pci_code: str = "",
    ):
        """
        Args:
            name: micorarchitecture name (e.g. ``blackwell``)
            brand_string: marketing name of specific device (e.g. ``NVIDIA GeForce RTX 5080``)
            vendor: name of chip manufacturer (e.g. ``nvidia``)
            driver_version: version number of currently installed driver (e.g. ``595.58.03``)
            vendor_pci_code: 4-digit hex string used to identify vendor (e.g. ``0x8086`` for intel)
            component_pci_code: 4-digit hex string used to identify GPU (e.g. ``0x2c02``)
        """
        self.name = name
        self.brand_string = brand_string
        self.vendor = vendor
        self.driver_version = driver_version
        self.vendor_pci_code = vendor_pci_code
        self.component_pci_code = component_pci_code

    def __str__(self) -> str:
        # TODO eventually change to name
        return self.brand_string
