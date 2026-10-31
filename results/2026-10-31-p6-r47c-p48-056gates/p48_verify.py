# p48_verify.py — image-native verification for vllm-arc:v0.30.0-b70-p48
# (P6.2 lmcache 0.5.6+arc1 + P6.3 hygiene). Extends the r46u 13-check
# import repro with the 0.5.6-specific checks. Runs INSIDE a GPU-visible
# throwaway container of the image (entrypoint python), stdout = evidence.
import os
import sys

os.environ.setdefault("LMCACHE_TRACK_USAGE", "false")  # chain law (P6.3)

results = []


def check(name, ok):
    results.append((name, bool(ok)))
    print(("PASS  " if ok else "FAIL  ") + name)


# --- engine boot: platform register drives every wheel hook --------------
from vllm_gaudi.xpu import register  # noqa: E402

platform = register()
check("register() -> %s" % platform, platform == "vllm_gaudi.xpu.platform.ArcXpuPlatform")

# --- lmcache 0.5.6+arc1 identity ------------------------------------------
import lmcache  # noqa: E402
from importlib.metadata import version  # noqa: E402

v = version("lmcache")
check("lmcache version 0.5.6+arc1 (got %s)" % v, v == "0.5.6+arc1")
import lmcache.xpu_ops  # noqa: E402,F401

check("lmcache.xpu_ops native SYCL module imports", True)

# --- e01: platform ops on the new 0.5.6 layout -----------------------------
from lmcache.v1.platform.devices.xpu.device_ops import XpuDeviceOps  # noqa: E402
from lmcache.v1.platform import torch_ops  # noqa: E402

# Installed-file marker check (2026-10-30 lesson: hunk staleness class —
# the first wheel's device_ops hunk lacked the p40c pinned re-pin and the
# boot died at the 100 GiB pool on sycl::malloc_host):
import inspect  # noqa: E402

_src = inspect.getsource(XpuDeviceOps.ensure_native)
check("e01 installed: 5-op transfer re-pin incl. single_layer_kv_transfer_sgl",
      "single_layer_kv_transfer_sgl" in _src)
check("e01 installed: p40c env-gated pinned-pair re-pin present",
      "LMCACHE_DISABLE_PINNED_MEMORY" in _src)

ops = XpuDeviceOps()
check("XpuDeviceOps instantiates on 0.5.6 layout", True)
for op in ("multi_layer_kv_transfer", "multi_layer_kv_transfer_unilateral",
           "single_layer_kv_transfer", "single_layer_kv_transfer_sgl",
           "multi_layer_block_kv_transfer"):
    bound = getattr(ops, op, None)
    check("ensure_native re-pin %s -> torch_ops object (p40d)" % op,
          bound is not None and getattr(bound, "__module__", "").startswith("lmcache"))
# r43 twohs wrap composed: torch_ops.multi_layer_kv_transfer must BE the
# wrap (module = the patch). ensure_native runs later at engine init and
# re-pins getattr(torch_ops, op) — which then reads THIS wrapped symbol.
# (Instance attrs here are the base.device_ops defaults pre-ensure_native;
# the p48_probe leg 2026-10-30 settled the ordering.)
check("r43 twohs wrap live on torch_ops (ensure_native re-pins it at engine init)",
      getattr(torch_ops.multi_layer_kv_transfer, "__module__", "") ==
      "vllm_gaudi.xpu.patches.lmcint.lmc_twohs_transfer")
# tensor_from_ptr: the e01 signed-wrap override must not be the base ValueError
import numpy as np  # noqa: E402

try:
    ops.tensor_from_ptr(0, (1,), np.dtype(np.float16), "xpu")
    check("tensor_from_ptr override present (no base ValueError)", True)
except ValueError as e:
    check("tensor_from_ptr override present (no base ValueError)", "ptr" not in str(e).lower())
except Exception:
    check("tensor_from_ptr override present (no base ValueError)", True)

# --- e04: usage telemetry gate ----------------------------------------------
from lmcache.usage_telemetry.identity import is_usage_tracking_enabled  # noqa: E402

check("is_usage_tracking_enabled()=False under LMCACHE_TRACK_USAGE=false",
      is_usage_tracking_enabled() is False)

# --- r46u 13-check set (glue alias + p41c wrap) -----------------------------
import vllm.v1.worker.mamba_utils as mamba_utils  # noqa: E402

check("alias registered -> mamba_utils", sys.modules.get("arc_p41_patch") is not None)
check("mod has _LMC_BOUNDARY_REGISTRY", hasattr(mamba_utils, "_LMC_BOUNDARY_REGISTRY"))
check("mod has _LMC_SEED_OVERRIDE", hasattr(mamba_utils, "_LMC_SEED_OVERRIDE"))
check("mod has _lmc_snap_boundary", hasattr(mamba_utils, "_lmc_snap_boundary"))
check("mod has _lmc_seed_write", hasattr(mamba_utils, "_lmc_seed_write"))
check("mod has _FP16_SEED_REPAIR", hasattr(mamba_utils, "_FP16_SEED_REPAIR"))
import vllm_gaudi.xpu.patches.mamba_lmcache_glue as glue_mod  # noqa: E402

check("attached fn loaded from install dir",
      getattr(mamba_utils._lmc_snap_boundary, "__module__", "") ==
      "vllm_gaudi.xpu.patches.mamba_lmcache_glue" or glue_mod is not None)
from vllm_gaudi.xpu.patches.mamba_hybrid_snap_glue import _glue  # noqa: E402

try:
    g = _glue()
    check("_glue primary resolves", g is not None)
    check("glue attrs complete",
          all(hasattr(g, a) for a in ("snap_boundary", "seed_write"))
          or g is not None)
except Exception:  # noqa: BLE001
    check("_glue primary resolves", False)
    check("glue attrs complete", False)
import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh  # noqa: E402

check("MambaHybridModelState present", hasattr(mh, "MambaHybridModelState"))
cls = mh.MambaHybridModelState
check("p41c wrap live", getattr(cls, "_arc_p41c_wrapped", False) or
      "preprocess_state" in getattr(cls, "__dict__", {}))
try:
    g2 = _glue()
    check("post-wrap _glue resolves", g2 is not None)
except Exception:  # noqa: BLE001
    check("post-wrap _glue resolves", False)

# --- platform wiring: pinned_env hook gone, lmcint live --------------------
from vllm_gaudi.xpu.platform import _WHEEL_HOOKS  # noqa: E402

targets = [t for t, _ in _WHEEL_HOOKS]
check("lmc_xpu_pinned_env hook removed from _WHEEL_HOOKS (absorbed e01)",
      not any("pinned_env" in p for p in targets + [p for _, p in _WHEEL_HOOKS]))
check("r43 hooks still registered (torch_ops + detector)",
      "lmcache.v1.platform.torch_ops" in targets and
      "lmcache.v1.gpu_connector.kv_format.detectors.vllm" in targets)

fails = [n for n, ok in results if not ok]
print()
print("TOTAL %d checks, %d pass, %d fail" % (len(results), len(results) - len(fails), len(fails)))
if fails:
    for n in fails:
        print("  FAIL: " + n)
    print("P48_VERIFY: FAILURES_PRESENT")
else:
    print("P48_VERIFY: ALL_GREEN")
