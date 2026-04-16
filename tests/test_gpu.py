# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import io
from unittest import mock

import pytest

import archspec.cli
import archspec.gpu.detect


@pytest.fixture(autouse=True)
def clear_gpu_host_cache():
    """Clears the cache for ``archspec.gpu.detect.host`` before each test."""
    archspec.gpu.detect.host.cache_clear()


def _create_sysfs_device(tmp_path, pci_address, class_code, vendor_id, device_id):
    """Helper to create a fake sysfs PCI device directory."""
    device_dir = tmp_path / pci_address
    device_dir.mkdir()
    (device_dir / "class").write_text(class_code)
    (device_dir / "vendor").write_text(vendor_id)
    (device_dir / "device").write_text(device_id)


def test_detect_nvidia_vendor(tmp_path, monkeypatch):
    """Test that an NVIDIA VGA controller vendor is detected."""
    _create_sysfs_device(tmp_path, "0000:01:00.0", "0x030000", "0x10de", "0x2c02")
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", str(tmp_path))

    vendors = archspec.gpu.detect._detect_gpu_vendors_linux()
    assert vendors == {"nvidia"}


def test_detect_amd_vendor(tmp_path, monkeypatch):
    """Test that an AMD VGA controller vendor is detected."""
    _create_sysfs_device(tmp_path, "0000:72:00.0", "0x030000", "0x1002", "0x13c0")
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", str(tmp_path))

    vendors = archspec.gpu.detect._detect_gpu_vendors_linux()
    assert vendors == {"amd"}


def test_detect_intel_vendor(tmp_path, monkeypatch):
    """Test that an Intel VGA controller vendor is detected."""
    _create_sysfs_device(tmp_path, "0000:00:02.0", "0x030000", "0x8086", "0x4680")
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", str(tmp_path))

    vendors = archspec.gpu.detect._detect_gpu_vendors_linux()
    assert vendors == {"intel"}


def test_detect_multiple_vendors(tmp_path, monkeypatch):
    """Test detection of multiple GPU vendors."""
    _create_sysfs_device(tmp_path, "0000:01:00.0", "0x030000", "0x10de", "0x2c02")
    _create_sysfs_device(tmp_path, "0000:72:00.0", "0x030000", "0x1002", "0x13c0")
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", str(tmp_path))

    vendors = archspec.gpu.detect._detect_gpu_vendors_linux()
    assert vendors == {"nvidia", "amd"}


def test_detect_no_gpus(tmp_path, monkeypatch):
    """Test that non-GPU PCI devices are filtered out."""
    # USB controller (class 0x0c0300)
    _create_sysfs_device(tmp_path, "0000:00:14.0", "0x0c0300", "0x8086", "0xa0ed")
    # NVMe controller (class 0x010802)
    _create_sysfs_device(tmp_path, "0000:03:00.0", "0x010802", "0x144d", "0xa809")
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", str(tmp_path))

    vendors = archspec.gpu.detect._detect_gpu_vendors_linux()
    assert len(vendors) == 0


def test_detect_3d_controller(tmp_path, monkeypatch):
    """Test that PCI class 0x0302 (3D controller) is also detected."""
    _create_sysfs_device(tmp_path, "0000:01:00.0", "0x030200", "0x10de", "0x1db4")
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", str(tmp_path))

    vendors = archspec.gpu.detect._detect_gpu_vendors_linux()
    assert vendors == {"nvidia"}


def test_detect_unknown_vendor_skipped(tmp_path, monkeypatch):
    """Test that GPUs from unknown vendors are skipped."""
    _create_sysfs_device(tmp_path, "0000:01:00.0", "0x030000", "0x9999", "0x0001")
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", str(tmp_path))

    vendors = archspec.gpu.detect._detect_gpu_vendors_linux()
    assert len(vendors) == 0


def test_detect_missing_sysfs_dir(monkeypatch):
    """Test graceful handling when sysfs PCI directory does not exist."""
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", "/nonexistent/path")

    vendors = archspec.gpu.detect._detect_gpu_vendors_linux()
    assert len(vendors) == 0


def test_host_returns_detailed_info(tmp_path, monkeypatch):
    """Test that host() calls vendor detail functions for detected vendors."""
    _create_sysfs_device(tmp_path, "0000:01:00.0", "0x030000", "0x10de", "0x2c02")
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", str(tmp_path))
    monkeypatch.setattr(
        archspec.gpu.detect,
        "INFO_FACTORY",
        {"Linux": [archspec.gpu.detect._detect_gpu_vendors_linux]},
    )
    monkeypatch.setattr(archspec.gpu.detect.platform, "system", lambda: "Linux")

    fake_gpu = archspec.gpu.detect.GPUMicroarch(
        brand_string="NVIDIA GeForce RTX 5080",
        vendor="nvidia",
        driver_version="595.58.03",
        vendor_pci_code="0x10de",
    )
    monkeypatch.setattr(archspec.gpu.detect, "_nvidia_info", lambda: [fake_gpu])

    gpus = archspec.gpu.detect.host()
    assert len(gpus) == 1
    assert gpus[0].vendor == "nvidia"
    assert gpus[0].brand_string == "NVIDIA GeForce RTX 5080"
    assert gpus[0].driver_version == "595.58.03"


def test_cli_gpu_subcommand(tmp_path, monkeypatch):
    """Test that the ``archspec gpu`` CLI subcommand runs and prints detected GPUs."""
    _create_sysfs_device(tmp_path, "0000:01:00.0", "0x030000", "0x10de", "0x2c02")
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", str(tmp_path))
    monkeypatch.setattr(
        archspec.gpu.detect,
        "INFO_FACTORY",
        {"Linux": [archspec.gpu.detect._detect_gpu_vendors_linux]},
    )
    monkeypatch.setattr(archspec.gpu.detect.platform, "system", lambda: "Linux")

    fake_gpu = archspec.gpu.detect.GPUMicroarch(
        brand_string="NVIDIA GeForce RTX 5080",
        vendor="nvidia",
        driver_version="595.58.03",
        vendor_pci_code="0x10de",
    )
    monkeypatch.setattr(archspec.gpu.detect, "_nvidia_info", lambda: [fake_gpu])

    with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
        result = archspec.cli.main(["gpu"])
    assert result == 0
    assert "nvidia" in stdout.getvalue()


def test_cli_gpu_no_gpus(tmp_path, monkeypatch):
    """Test that the ``archspec gpu`` CLI subcommand handles no GPUs gracefully."""
    monkeypatch.setattr(archspec.gpu.detect, "SYSFS_PCI_DEVICES", str(tmp_path))
    monkeypatch.setattr(
        archspec.gpu.detect,
        "INFO_FACTORY",
        {"Linux": [archspec.gpu.detect._detect_gpu_vendors_linux]},
    )
    monkeypatch.setattr(archspec.gpu.detect.platform, "system", lambda: "Linux")

    with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
        result = archspec.cli.main(["gpu"])
    assert result == 0
    assert "No GPUs detected" in stdout.getvalue()
