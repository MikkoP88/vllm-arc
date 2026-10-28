"""arc p40c — honor LMCACHE_DISABLE_PINNED_MEMORY under the native SYCL ops
(lmcache-side glue; installed via sitecustomize exec_module hook on
`lmcache.v1.platform.xpu.device_ops`).

PROBLEM (r30 boot 5, 2026-10-28): LocalCPUBackend init →
memory_management._allocate_cpu_memory → device_ops.alloc_pinned_ptr
attempted sycl::malloc_host of the FULL 100 GiB CPU pool (LMCACHE_MAX_
LOCAL_CPU_SIZE=100) and died: "sycl::malloc_host failed for 107374182400
bytes (XPU pinned host)".

Root cause: the certified env (C serve_lmc_v133.sh, verbatim in our
r30_serve.sh) runs the CPU pools PAGEABLE — "SYCL USM pinned host alloc
segfaults above ~16-32GiB per process on this stack (probed 2026-10-08);
run the CPU pools pageable" — via LMCACHE_DISABLE_PINNED_MEMORY=1, which
lmcache's torch baseline (v1/platform/torch_ops.py, the e01 C delta)
honors (_alloc_page_aligned_pinned_view checks the env, allocates
pin_memory=False). But our 0.5.5-built lmcache.xpu_ops exports NATIVE
alloc_pinned_ptr/free_pinned_ptr, and XpuDeviceOps.ensure_native →
DeviceOps.bind_native INSTANCE-binds every public native symbol over the
class-level torch baseline — the native pinned alloc ignores the env.
C's 0.4.7-era binary xpu_ops.so did not export the pair, so C always
served these two ops from the env-honoring baseline.

FIX: with the env set, drop the two instance bindings after the native
bind so attribute lookup falls back to the class-level torch baseline
(pageable, env-honoring — C's effective posture). All other native ops
(kernels, transfers) stay native.

Disable knob: unset LMCACHE_DISABLE_PINNED_MEMORY (or =0) leaves the
native pinned binding in place.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "True", "on", "yes"}


def _install(mod) -> None:
    cls = getattr(mod, "XpuDeviceOps", None)
    if cls is None or getattr(cls, "_arc_p40c", False):
        return  # unexpected module shape or already patched — idempotent
    if os.environ.get("LMCACHE_DISABLE_PINNED_MEMORY", "0") not in _TRUTHY:
        return  # pinned allowed — native binding stands

    orig = cls.ensure_native

    def ensure_native(self):
        orig(self)
        for name in ("alloc_pinned_ptr", "free_pinned_ptr"):
            if name in self.__dict__:  # instance-bound by bind_native
                del self.__dict__[name]
        print(
            "[xpu_device_ops] native pinned alloc dropped "
            f"(DISABLE_PINNED_MEMORY) (pid={os.getpid()})",
            flush=True,
        )

    cls.ensure_native = ensure_native
    cls._arc_p40c = True
