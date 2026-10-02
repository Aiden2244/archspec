# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import subprocess

import pytest

import archspec.gpu
import archspec.gpu.amd
import archspec.gpu.nvidia
from archspec.gpu.gpu import GPU, ComputeCapability, GFXTarget, VendorPciCode


def mock_smi(stdout, returncode=0):
    """Return a ``subprocess.run`` replacement that yields the given stdout."""

    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout)

    return _run


@pytest.mark.parametrize(
    "combined,expected",
    [
        ("0x2C0210DE", ("0x2c02", "0x10de")),
        ("0x233010DE", ("0x2330", "0x10de")),
    ],
)
def test_nvidia_pci_device_id_parsing(combined, expected):
    """Test that a combined PCI device ID splits into lowercase (device, vendor) codes."""
    assert archspec.gpu.nvidia._parse_pci_device_id(combined) == expected


@pytest.mark.parametrize("bad_id", ["0x2C0210", "2C0210DE00", "0xZZZZ10DE0", "0xZZZZ10DE"])
def test_nvidia_pci_device_id_invalid(bad_id):
    """Test that a malformed PCI device ID raises ValueError."""
    with pytest.raises(ValueError):
        archspec.gpu.nvidia._parse_pci_device_id(bad_id)


def test_nvidia_smi_info_parses_smi_output(monkeypatch):
    """Test that nvidia.smi_info parses nvidia-smi CSV output into GPU objects."""
    nvidia_smi_csv = (
        "NVIDIA GeForce RTX 5080, 595.58.03, 0x2C0210DE, 12.0\n"
        "NVIDIA H100 PCIe, 550.54.15, 0x233010DE, 9.0\n"
    )
    monkeypatch.setattr(archspec.gpu.nvidia.subprocess, "run", mock_smi(nvidia_smi_csv))

    gpus = archspec.gpu.nvidia.smi_info()

    assert len(gpus) == 2
    assert all(gpu.vendor == "nvidia" for gpu in gpus)
    assert all(gpu.vendor_pci_code == "0x10de" for gpu in gpus)

    assert gpus[0].name == "NVIDIA GeForce RTX 5080"
    assert gpus[0].driver_version == "595.58.03"
    assert gpus[0].component_pci_code == "0x2c02"
    assert gpus[0].compute_capability == ComputeCapability(12, 0)

    assert gpus[1].name == "NVIDIA H100 PCIe"
    assert gpus[1].driver_version == "550.54.15"
    assert gpus[1].component_pci_code == "0x2330"
    assert gpus[1].compute_capability == ComputeCapability(9, 0)


def test_rocm_smi_info_handles_malformed_json(monkeypatch):
    """Test that amd.smi_info returns no GPUs when rocm-smi emits unparseable output."""
    monkeypatch.setattr(archspec.gpu.amd.subprocess, "run", mock_smi("not json"))

    assert archspec.gpu.amd.smi_info() == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "name": "NVIDIA H100 PCIe",
            "vendor": "nvidia",
            "driver_version": "550.54.15",
            "vendor_pci_code": VendorPciCode.NVIDIA,
            "component_pci_code": "0x2330",
            "compute_capability": ComputeCapability(9, 0),
        },
        {
            "name": "AMD Instinct MI300A",
            "vendor": "amd",
            "driver_version": "6.16.13",
            "vendor_pci_code": VendorPciCode.AMD,
            "component_pci_code": "0x74a0",
            "gfx_target": GFXTarget(9, 4, 2),
        },
    ],
)
def test_round_trip_dict(kwargs):
    """A GPU survives a round trip through to_dict/from_dict."""
    gpu = GPU(**kwargs)
    assert GPU.from_dict(gpu.to_dict()) == gpu


def test_equality_and_hash():
    """Equal GPU objects compare equal, hash equal, and deduplicate in sets."""
    kwargs = {
        "name": "AMD Instinct MI300A",
        "vendor": "amd",
        "vendor_pci_code": VendorPciCode.AMD,
        "component_pci_code": "0x74a0",
        "gfx_target": GFXTarget(9, 4, 2),
    }
    first = GPU(**kwargs)
    second = GPU(**kwargs)
    other_driver = GPU(**kwargs, driver_version="6.16.13")

    assert first == second
    assert hash(first) == hash(second)
    assert first != other_driver
    assert first != "gfx942"
    assert len({first, second}) == 1
