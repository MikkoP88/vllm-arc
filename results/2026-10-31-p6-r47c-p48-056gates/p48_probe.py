# p48_probe.py — settle the composition order: does XpuDeviceOps.ensure_native
# re-pin the r43-WRAPPED torch_ops.multi_layer_kv_transfer or the original?
import sys

from vllm_gaudi.xpu import register

register()
import lmcache
import lmcache.xpu_ops  # noqa: F401
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.devices.xpu.device_ops import XpuDeviceOps

t = torch_ops.multi_layer_kv_transfer
ops = XpuDeviceOps()
o = ops.multi_layer_kv_transfer
print("torch_ops.multi_layer_kv_transfer: module=%s id=%s" % (getattr(t, "__module__", "?"), id(t)))
print("ops.multi_layer_kv_transfer:       module=%s id=%s" % (getattr(o, "__module__", "?"), id(o)))
print("IDENTICAL" if t is o else "DIFFERENT")
print("torch_ops pkg module name:", torch_ops.__name__)
print("twohs patch in sys.modules:", "vllm_gaudi.xpu.patches.lmcint.lmc_twohs_transfer" in sys.modules)
p = sys.modules.get("vllm_gaudi.xpu.patches.lmcint.lmc_twohs_transfer")
if p is not None:
    print("patch _ORIG id:", id(getattr(p, "_ORIG", None)))
    print("_ORIG is ops attr:", getattr(p, "_ORIG", None) is o)
    print("_ORIG is torch_ops attr:", getattr(p, "_ORIG", None) is t)
