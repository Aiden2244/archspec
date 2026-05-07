# Copyright 2019-2026 Lawrence Livermore National Security, LLC and other
# Archspec Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""Detection of GPU microarchitectures"""

import collections
import functools
import platform
from typing import Callable, Dict, List

from archspec.gpu.amd import Amd
from archspec.gpu.gpu_microarch import GPUMicroarch
from archspec.gpu.intel import Intel
from archspec.gpu.nvidia import Nvidia
from archspec.gpu.vendor import Vendor

#: Mapping from operating systems to chain of factories that return
#: the list of GPU vendors detected on the current system. Iteration
#: order within an OS allows fallback factories.
INFO_FACTORY: Dict[str, List[Callable[[], List[Vendor]]]] = collections.defaultdict(list)


def detection(operating_system: str):
    """Decorator to mark functions that return the list of detected GPU vendors.

    Args:
        operating_system: operating system where this function can be used.
    """

    def decorator(factory):
        INFO_FACTORY[operating_system].append(factory)
        return factory

    return decorator


@detection(operating_system="Linux")
def _detect_vendors_linux() -> List[Vendor]:
    """Return one :class:`Vendor` instance per GPU vendor present on Linux.

    """
    candidates: List[Vendor] = [Nvidia(), Amd(), Intel()]
    return [v for v in candidates if v.detect()]


@functools.lru_cache(maxsize=None)
def host() -> List[GPUMicroarch]:
    """Detects the GPUs on the system and returns information about them.

    Returns:
        A list of GPUMicroarch objects, one per detected GPU.
    """
    detected: List[Vendor] = []
    for factory in INFO_FACTORY[platform.system()]:
        try:
            detected = factory()
            break
        except Exception:
            continue

    results: List[GPUMicroarch] = []
    for vendor in detected:
        results.extend(vendor.info())

    return results
