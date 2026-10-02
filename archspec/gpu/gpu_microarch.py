# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Defines the GPUMicroarch class, describing a detected GPU microarchitecture."""

from typing import Dict

from .gpu import GPU


class GPUMicroarch:
    """Represents a GPU Microarchitecture"""

    def __init__(
        self,
        name: str = "",
        vendor: str = "",
    ):
        """
        Args:
            name: compatibility identifier name (e.g. ``9.0`` for NVIDIA, ``gfx90a`` for AMD)
            vendor: name of chip manufacturer (e.g. ``nvidia``)
        """
        self.name = name
        self.vendor = vendor

    def __eq__(self, other):
        if not isinstance(other, GPUMicroarch):
            return NotImplemented
        return vars(self) == vars(other)

    def __hash__(self) -> int:
        return hash((self.vendor, self.name))

    def __repr__(self) -> str:
        fields = ", ".join(f"{field}={value!r}" for field, value in vars(self).items() if value)
        return f"{self.__class__.__name__}({fields})"

    def __str__(self) -> str:
        if self.name and self.vendor:
            return f"{self.name} ({self.vendor})"
        return "unknown gpu"

    def to_dict(self) -> Dict[str, str]:
        """Returns a dictionary representation of this object."""
        return {
            "name": self.name,
            "vendor": self.vendor,
        }

    @classmethod
    def from_gpu(cls, gpu: GPU) -> "GPUMicroarch":
        """Construct a GPU microarchitecture from a GPU object."""
        if not gpu.vendor:
            raise ValueError(f"Cannot construct GPUMicroarch from {gpu!r}: missing vendor")

        name: str = ""
        if gpu.vendor == "nvidia":
            if not gpu.compute_capability:
                raise ValueError(
                    f"Cannot construct GPUMicroarch from {gpu!r}: missing compute_capability"
                )
            name = gpu.compute_capability.name

        elif gpu.vendor == "amd":
            if not gpu.gfx_target:
                raise ValueError(f"Cannot construct GPUMicroarch from {gpu!r}: missing gfx_target")
            name = gpu.gfx_target.name

        # If vendor field is populated but unsupported (e.g. Intel for the time being)
        else:
            raise ValueError(
                f"Cannot construct GPUMicroarch from {gpu!r}: invalid vendor {gpu.vendor!r}"
            )

        return cls(name=name, vendor=gpu.vendor)
