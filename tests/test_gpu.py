# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import json
import subprocess

import archspec.gpu.detect


def test_detect_amd_instinct_accelerator(tmp_path, monkeypatch):
    """Test that an AMD Instinct accelerator (PCI class 0x120000) is detected."""
    device_dir = tmp_path / "0000:c1:00.0"
    device_dir.mkdir()
    (device_dir / "class").write_text("0x120000")
    (device_dir / "vendor").write_text("0x1002")
    (device_dir / "device").write_text("0x74a0")
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", str(tmp_path))

    vendors = archspec.gpu.detect._detect_gpu_vendors_linux()
    assert vendors == {"amd"}


def test_amd_info_parses_rocm_smi_json(monkeypatch):
    """Test that _amd_info parses rocm-smi --json output into GPUMicroarch objects."""
    rocm_smi_json = json.dumps(
        {
            "card0": {"Card Series": "AMD Radeon RX 6800 XT", "Device ID": "0x73BF"},
            "card1": {"Market Name": "AMD Instinct MI250X", "GPU ID": "0x740C"},
            "system": {"Driver version": "6.7.0"},
        }
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=rocm_smi_json)

    monkeypatch.setattr(archspec.gpu.detect.subprocess, "run", fake_run)

    gpus = archspec.gpu.detect._amd_info()

    assert len(gpus) == 2
    assert all(gpu.vendor == "amd" for gpu in gpus)
    assert all(gpu.vendor_pci_code == "0x1002" for gpu in gpus)
    assert all(gpu.driver_version == "6.7.0" for gpu in gpus)

    assert gpus[0].brand_string == "AMD Radeon RX 6800 XT"
    assert gpus[0].component_pci_code == "0x73bf"

    assert gpus[1].brand_string == "AMD Instinct MI250X"
    assert gpus[1].component_pci_code == "0x740c"


def test_amd_info_parses_real_mi300a_output(monkeypatch):
    """Test _amd_info against real rocm-smi output from a 4x MI300A machine."""
    rocm_smi_json = (
        '{"card0": {"Device Name": "AMD Instinct MI300A", "Device ID": "0x74a0", '
        '"Device Rev": "0x00", "Subsystem ID": "0x74a0", "GUID": "46363", '
        '"Card Series": "AMD Instinct MI300A", "Card Model": "0x74a0", '
        '"Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]", "Card SKU": "N/A", '
        '"Node ID": "4", "GFX Version": "gfx942"}, '
        '"card1": {"Device Name": "AMD Instinct MI300A", "Device ID": "0x74a0", '
        '"Device Rev": "0x00", "Subsystem ID": "0x74a0", "GUID": "46186", '
        '"Card Series": "AMD Instinct MI300A", "Card Model": "0x74a0", '
        '"Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]", "Card SKU": "N/A", '
        '"Node ID": "5", "GFX Version": "gfx942"}, '
        '"card2": {"Device Name": "AMD Instinct MI300A", "Device ID": "0x74a0", '
        '"Device Rev": "0x00", "Subsystem ID": "0x74a0", "GUID": "58426", '
        '"Card Series": "AMD Instinct MI300A", "Card Model": "0x74a0", '
        '"Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]", "Card SKU": "N/A", '
        '"Node ID": "6", "GFX Version": "gfx942"}, '
        '"card3": {"Device Name": "AMD Instinct MI300A", "Device ID": "0x74a0", '
        '"Device Rev": "0x00", "Subsystem ID": "0x74a0", "GUID": "5131", '
        '"Card Series": "AMD Instinct MI300A", "Card Model": "0x74a0", '
        '"Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]", "Card SKU": "N/A", '
        '"Node ID": "7", "GFX Version": "gfx942"}, '
        '"system": {"Driver version": "6.16.13"}}'
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=rocm_smi_json)

    monkeypatch.setattr(archspec.gpu.detect.subprocess, "run", fake_run)

    gpus = archspec.gpu.detect._amd_info()

    assert len(gpus) == 4
    for gpu in gpus:
        assert gpu.vendor == "amd"
        assert gpu.vendor_pci_code == "0x1002"
        assert gpu.driver_version == "6.16.13"
        assert gpu.brand_string == "AMD Instinct MI300A"
        assert gpu.component_pci_code == "0x74a0"
