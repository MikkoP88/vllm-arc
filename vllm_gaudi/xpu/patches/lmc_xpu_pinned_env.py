# SPDX-License-Identifier: Apache-2.0
"""arc p40c — honor LMCACHE_DISABLE_PINNED_MEMORY under the native SYCL ops
+ arc p40d — KV-transfer ops on the torch baseline (C's XPU posture)
(lmcache-side glue; installed via sitecustomize exec_module hook on
`lmcache.v1.platform.xpu.device_ops`).

p40c PROBLEM (r30 boot 5, 2026-10-28): LocalCPUBackend init →
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

p40c FIX: with the env set, drop the two instance bindings after the
native bind so attribute lookup falls back to the class-level torch
baseline (pageable, env-honoring — C's effective posture).

p40d PROBLEM (r31 DEBUG attribution, 2026-10-28): EVERY lagged attn
store died — "Lagged attn store failed … (cached chunk absent)" ×68 —
with the real traceback: lmcache_engine.store → gpu_connector
xpu_connectors.py:258 from_gpu → device_ops.multi_layer_kv_transfer →
RuntimeError: Unsupported non-MLA EngineKVFormat (raise is inside our
built native lmcache.xpu_ops; the string appears nowhere in the python
tree). The native KV-transfer generation supports MLA layouts only; our
model is GQA (kv_shape (17,2,896,2,256), use_mla=False). Net effect: the
CPU pool stayed COMPLETELY empty (no attn chunks, and the joint lookup
gates on the attn hit > 0, so the mamba pair never engaged either) —
every warm face logged "LMCache hit tokens: 0" and ALL warm speedup to
date was vllm APC alone (r31 probe: warm ≈ 0.70–0.75 ms × residual
tokens after the APC's last-1–2-mamba-pages miss: 8192→residual
1024→771 ms; 4096→residual 1408→976 ms).

p40d FIX (C's posture, verified by md5-identical python + symbol-sparse
native): C's image ships NO lmcache.xpu_ops at all (package root: only
lmcache_native/cuda_ops/lmcache_fs .so; strings(lmcache_native.so) has
ZERO multi_layer_kv_transfer symbols), so C's ensure_native takes the
ImportError path — "XpuDeviceOps stays on the torch baseline for all
ops" — and its KV transfers run the FULLY VECTORIZED torch fallback
(torch_ops.py:598 multi_layer_kv_transfer — the e01-delta file, already
in our wheel; only HND layouts are NotImplementedError there, GQA is
served). Mirror C exactly: after bind_native, drop the instance
bindings for the four KV-transfer ops the base class routes to
torch_ops (multi_layer_kv_transfer, multi_layer_kv_transfer_unilateral,
single_layer_kv_transfer, single_layer_kv_transfer_sgl) so the class-
level torch baseline serves them. Other native ops stay native.

Disable knobs: unset LMCACHE_DISABLE_PINNED_MEMORY (or =0) leaves the
native pinned binding in place; VLLM_ARC_LMC_TORCH_KV=0 leaves the
native KV-transfer bindings in place (A/B evidence knob — expect the
68× store failures and hit=0 to return).
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "True", "on", "yes"}
_PINNED_OPS = ("alloc_pinned_ptr", "free_pinned_ptr")
_KV_TRANSFER_OPS = (
    "multi_layer_kv_transfer",
    "multi_layer_kv_transfer_unilateral",
    "single_layer_kv_transfer",
    "single_layer_kv_transfer_sgl",
)


def _install(mod) -> None:
    cls = getattr(mod, "XpuDeviceOps", None)
    if cls is None or getattr(cls, "_arc_p40c", False):
        return  # unexpected module shape or already patched — idempotent

    drop_pinned = os.environ.get("LMCACHE_DISABLE_PINNED_MEMORY", "0") in _TRUTHY
    drop_kv = os.environ.get("VLLM_ARC_LMC_TORCH_KV", "1") != "0"
    if not drop_pinned and not drop_kv:
        return  # nothing to do — native bindings stand

    orig = cls.ensure_native

    def ensure_native(self):
        orig(self)
        if drop_pinned:
            for name in _PINNED_OPS:
                if name in self.__dict__:  # instance-bound by bind_native
                    del self.__dict__[name]
            print(
                "[xpu_device_ops] native pinned alloc dropped "
                f"(DISABLE_PINNED_MEMORY) (pid={os.getpid()})",
                flush=True,
            )
        if drop_kv:
            for name in _KV_TRANSFER_OPS:
                if name in self.__dict__:  # instance-bound by bind_native
                    del self.__dict__[name]
            print(
                "[xpu_device_ops] native kv-transfer dropped -> torch "
                f"baseline (p40d) (pid={os.getpid()})",
                flush=True,
            )

    cls.ensure_native = ensure_native
    cls._arc_p40c = True
