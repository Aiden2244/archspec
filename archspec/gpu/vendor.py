# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Abstract contract for a GPU vendor.

Encapsulates everything archspec needs to know about a GPU manufacturer to
detect and describe its hardware: its PCI vendor ID, how to detect that its
hardware is present on the current system, and how to fetch detailed per-device
information once presence is confirmed.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Tuple

from archspec.gpu.gpu_microarch import GPUMicroarch


class Vendor(ABC):
    """Abstract base class for a GPU vendor."""

    #: Vendor name, e.g. ``"nvidia"``. Subclasses must override.
    name: str = ""

    #: PCI vendor ID as a lowercase ``0x``-prefixed hex string,
    #: e.g. ``"0x10de"`` for NVIDIA. Subclasses must override.
    pci_code: str = ""

    #: PCI class codes that identify a device as a GPU
    #: (VGA controller and 3D controller respectively).
    #: https://admin.pci-ids.ucw.cz/read/PD/03
    GPU_PCI_CLASSES: Tuple[str, ...] = ("0x030000", "0x030200")

    #: Path to the sysfs PCI devices directory.
    SYSFS_PCI_DEVICES: str = "/sys/bus/pci/devices"

    @staticmethod
    def _read_sysfs_file(path: str) -> str:
        with open(path) as f:
            return f.read().strip()

    def _scan_sysfs(self) -> bool:
        """Return True if a PCI GPU device matching :attr:`pci_code` is present."""
        if not os.path.isdir(self.SYSFS_PCI_DEVICES):
            return False

        for entry in os.listdir(self.SYSFS_PCI_DEVICES):
            device_dir = os.path.join(self.SYSFS_PCI_DEVICES, entry)
            if not os.path.isdir(device_dir):
                continue

            class_path = os.path.join(device_dir, "class")
            if not os.path.exists(class_path):
                continue

            class_code = self._read_sysfs_file(class_path)
            if class_code not in self.GPU_PCI_CLASSES:
                continue

            vendor_id = self._read_sysfs_file(os.path.join(device_dir, "vendor"))
            if vendor_id == self.pci_code:
                return True

        return False

    def detect(self) -> bool:
        """Return True if this vendor's hardware is present on the system."""
        return self._scan_sysfs()

    @abstractmethod
    def info(self) -> List[GPUMicroarch]:
        """Return detailed info for each GPU of this vendor on the system."""
