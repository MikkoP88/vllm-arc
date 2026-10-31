# p65_verify.py — image-native p65 hunk checks for cert4 (append to the
# p48 battery). Verifies the SHIPPED lmcint adapter — the src-override
# target that is the live row-builder (r48j lesson) — carries the gated
# chunk-set store deferral, defaults OFF, and carries no probe markers.
# File-based by necessity: lmcint's adapter imports siblings via the
# VENDORED lmcache_integration fullnames (it is exec'd under that name in
# production by _SrcOverrideFinder), so a standalone import of the
# vllm_gaudi path re-triggers upstream's multi_process_adapter
# ImportError (the exact skew the shim exists to bypass — r47c DEFECT 2).
# The behavioral evidence lives in the r48k/r48l legs (defer rows + store
# histogram); here we pin identity + content + compilability.
import os
import pathlib

results = []


def check(name, ok):
    results.append((name, bool(ok)))
    print(("PASS  " if ok else "FAIL  ") + name)


ADP = pathlib.Path(
    "/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/"
    "lmcint/vllm_v1_adapter.py"
)

# --- identity: the shim must map the vendored fullname to THIS file -----
from vllm_gaudi.xpu.patches.lmcache_hma_connector import (  # noqa: E402
    _LMCINT,
    _OVERRIDE_FILES,
)

check("ADP file exists: %s" % ADP, ADP.is_file())
check(
    "src-override maps vendored adapter fullname -> lmcint file",
    _OVERRIDE_FILES.get(f"{_LMCINT}.vllm_v1_adapter")
    == "lmcint/vllm_v1_adapter.py",
)

src = ADP.read_text()

# --- the gated deferral hunk ----------------------------------------------
i = src.find("def from_request_tracker")
body = src[i : i + 6000]
check("defer_chunk_store present in row builder", "defer_chunk_store" in body)
check("length gate LMC_DEFER_MIN_TOKENS present", "LMC_DEFER_MIN_TOKENS" in body)
check(
    "deferral default OFF (env default '0')",
    'os.environ.get("LMC_DEFER_CHUNK_STORE", "0")' in body,
)
check(
    "can_save gated on defer",
    "not skip_save and not defer_chunk_store" in body,
)
check(
    "num_saved_tokens advance gated on defer",
    "if not skip_save and not defer_chunk_store:" in body,
)

# --- worker-side can_save guards (3 consumer loops) ----------------------
check(
    "worker loops guard can_save (3 sites)",
    src.count("not save_spec.can_save") == 3,
)

# --- probe hygiene: nothing probe-only ships ------------------------------
check("no probe instrumentation (ARC_P65_) in shipped file",
      "ARC_P65_" not in src)

# --- the file compiles as a module ----------------------------------------
try:
    compile(src, str(ADP), "exec")
    check("lmcint adapter compiles", True)
except SyntaxError as e:  # noqa: BLE001
    check("lmcint adapter compiles (%s)" % e, False)

print(
    "P65_CHECKS: %d/%d PASS" % (sum(1 for _, ok in results if ok), len(results))
)
raise SystemExit(0 if all(ok for _, ok in results) else 1)
