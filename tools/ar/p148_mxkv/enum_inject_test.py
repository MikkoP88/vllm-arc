# SPDX-License-Identifier: Apache-2.0
"""One-shot rig probe: post-hoc MXKV member injection into the wheel's
AttentionBackendEnum (P6.47 leg B, registry-preimported case)."""

import vllm  # noqa: F401 — resolves the platform plugin chain like serve does
from vllm.v1.attention.backends.registry import AttentionBackendEnum

PATH = "vllm_gaudi.xpu.patches.mx_kv.engine.MxAttentionBackend"
member = object.__new__(AttentionBackendEnum)
member._name_ = "MXKV"
member._value_ = PATH
AttentionBackendEnum._member_map_["MXKV"] = member
AttentionBackendEnum._value2member_map_[PATH] = member
try:
    setattr(AttentionBackendEnum, "MXKV", member)  # noqa: B010 — probing setattr itself
    print("setattr: ok")
except Exception as e:  # noqa: BLE001
    print("setattr BLOCKED:", repr(e))
m = AttentionBackendEnum["MXKV"]
print("lookup:", m)
print("value:", m.value)
print("get_path:", m.get_path())
print("members:", "MXKV" in AttentionBackendEnum.__members__)
print("by-value:", AttentionBackendEnum(PATH) is member)
print("repr:", repr(m))
