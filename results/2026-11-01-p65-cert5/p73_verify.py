# p73_verify.py — image-native P6.13 slice-hunk checks for cert5 (append
# to the p48 battery). Same file-based law as cert4's p65_verify.py: the
# shipped lmcint adapter is the src-override target that is the live
# row-builder/flusher (r48j lesson) — a standalone import of the
# vllm_gaudi path re-triggers upstream's multi_process_adapter
# ImportError (the exact skew the shim exists to bypass — r47c DEFECT 2),
# so identity + content + compilability are pinned here; the behavioral
# evidence (store histogram 63232-class -> 0, 38 = 2x19 windows) lives in
# the gate's slice face (results/2026-11-01-p65-rung3-slice/).
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

# --- regression guard: the P6.5 deferral hunk is still intact -----------
i = src.find("def from_request_tracker")
body = src[i : i + 6000]
check(
    "P6.5 defer hunk intact (gate + default OFF)",
    "defer_chunk_store" in body
    and "LMC_DEFER_MIN_TOKENS" in body
    and 'os.environ.get("LMC_DEFER_CHUNK_STORE", "0")' in body,
)
check(
    "worker loops guard can_save (3 sites)",
    src.count("not save_spec.can_save") == 3,
)

# --- the P6.13 slice hunk ------------------------------------------------
check(
    "module-level _lmc_slice_store_window present",
    "def _lmc_slice_store_window" in src,
)
j = src.find("def _lmc_flush_pending")
fbody = src[j : j + 9000]
check(
    "slice default OFF (env default '0')",
    'os.environ.get("LMC_STORE_SLICE_TOKENS", "0")' in fbody,
)
check("flush calls the slice helper", "_lmc_slice_store_window(" in fbody)
check(
    "slice store keeps the chunk-row call shape (mask+slot+offset)",
    "mask=p_mask" in fbody
    and "slot_mapping=p_slot" in fbody
    and "offset=p_off" in fbody,
)
check(
    "remainder re-queues into _lmc_pending_stores (cursor rides tuple)",
    "_lmc_pending_stores.append" in fbody and "_cursor" in fbody,
)

# --- probe hygiene: nothing probe-only ships ----------------------------
check(
    "no probe instrumentation (ARC_P73_) in shipped file",
    "ARC_P73_" not in src,
)

# --- the file compiles as a module ---------------------------------------
try:
    compile(src, str(ADP), "exec")
    check("lmcint adapter compiles", True)
except SyntaxError as e:  # noqa: BLE001
    check("lmcint adapter compiles (%s)" % e, False)

print(
    "P73_CHECKS: %d/%d PASS" % (sum(1 for _, ok in results if ok), len(results))
)
raise SystemExit(0 if all(ok for _, ok in results) else 1)
