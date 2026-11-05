# SPDX-License-Identifier: Apache-2.0
"""p72 dense-FA ESIMD kernel loader (cap-stage form).

Loads the built extension ``fa_dense_ops.so`` from a path (default
``/root/fa_dense_ops.so`` — the W14 cap-stage pattern; the image-bake form
lands with the p90/p91 Dockerfile esimd stage). Loader law (W14): the
importlib module name MUST equal the .so basename — the pybind init is
``PyInit_fa_dense_ops``; an alias name raises "dynamic module does not
define module export function".

No import of this package happens unless the serve patch's env gate is on
(default off) — importing it is inert on a kernel-less image.
"""
from __future__ import annotations

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location

_FA_DENSE_OPS = None


def load_fa_dense_ops():
    """Return the loaded fa_dense_ops module (cached). Raises on failure —
    the caller (patches/dense_fa_esimd.py) converts that into a loud
    one-time warning + permanent stock fall-through (visibility law)."""
    global _FA_DENSE_OPS
    if _FA_DENSE_OPS is not None:
        return _FA_DENSE_OPS
    path = os.environ.get("ARC_DENSE_FA_ESIMD_SO", "/root/fa_dense_ops.so")
    # name == basename (loader law); sys.modules guard for idempotence.
    if "fa_dense_ops" in sys.modules:
        _FA_DENSE_OPS = sys.modules["fa_dense_ops"]
        return _FA_DENSE_OPS
    spec = spec_from_file_location("fa_dense_ops", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"fa_dense_ops: cannot load {path}")
    import torch  # noqa: F401 — libc10 must be in-process before dlopen
    # (rpath law, results/2026-10-31-p72-rungb-dense-fa-esimd/BUILD.md)
    mod = module_from_spec(spec)
    sys.modules["fa_dense_ops"] = mod
    spec.loader.exec_module(mod)
    _FA_DENSE_OPS = mod
    return _FA_DENSE_OPS
