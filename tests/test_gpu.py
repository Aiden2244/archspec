# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import json
import subprocess

import archspec.gpu.detect


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
