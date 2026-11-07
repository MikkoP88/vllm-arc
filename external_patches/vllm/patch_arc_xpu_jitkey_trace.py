#!/usr/bin/env python3
"""ARC P6.28 kit patch #4: per-call trace of the resolved triton cache key
(DIAGNOSTIC overlay — pairs with patch_arc_xpu_attn_trace.py; never part of
a certified image, only of the certNNdbg diagnostic lineage).

Why (r108u, banked results/2026-11-06-p628-gemma4/phaseC/
cert15dbg_boot/): the cert15 warm gate issued, at 04:43:16, the engine-
exact (98, 98) sliding head-256 prefill call — and the engine's own first
request-time call at 05:09:15, whose patch-#3 trace line is byte-identical
on every logged dimension (fnid, shapes, strides, dtypes, constexprs,
launch config, grid), cache-MISSED inside JITFunction.run and recompiled
for ~288 s until the 300 s RPC watchdog killed the step. A call that
differs from a cached call in no traced dimension differs in an UNTRACED
key input: the binder specialization attrs (pointer data_ptr alignment,
integer divisibility-by-16 / ==1) or the options object. Guessing those
has already cost two windows (r108t alignment-in-warm theory, r108u
wrapper-dimension audit), so this patch moves the telemetry to the
authority itself: one line per JITFunction.run call, printed at the
kernel_cache lookup, carrying the computed key, the hit/miss bit, every
specialization (type, attr) pair, the per-argument pointer alignment
(data_ptr % 16), and warps/stages. Warm key-set and request key in one
serve log — the next window reads the differing dimension off the diff.

Env-gated: ARC_XPU_TRACE_JITKEY (unset/""/"0" = off). The cert15dbg2
diagnostic image bakes it ON; it defaults OFF everywhere else.

Usage:
    python patch_arc_xpu_jitkey_trace.py [path/to/triton/runtime/jit.py] [--audit]

Without a path, the wheel location is found via importlib. --audit only
reports. Idempotent (mark count checked); a .arcxpubak backup is left
next to the original.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import py_compile
import shutil
import sys

MARK = "[ARC-P628-XPU-JITKEY]"

# Anchor 1 — module level: immediately before the JITFunction class
# statement (unique line in the stock triton wheel).
ANCHOR_MODULE = "class JITFunction(JITCallable, KernelInterface[T]):\n"

# Anchor 2 — inside JITFunction.run: the cache lookup the whole question
# hangs on. The two-line block is unique (kernel_cache.get appears once).
ANCHOR_RUN = (
    "        key = compute_cache_key(kernel_key_cache, specialization, options)\n"
    "        kernel = kernel_cache.get(key, None)\n"
)

HELPER_BLOCK = '''from os import environ as _arc_environ

# [ARC-P628-XPU-JITKEY] read once at import: the per-call gate below is a
# plain global bool so the off-state cost is one load + branch.
_ARC_JITKEY_ON = _arc_environ.get("ARC_XPU_TRACE_JITKEY", "") not in ("", "0")


def _arc_xpu_jitkey_trace(fn_obj, key, kernel, specialization, options,
                          bound_args):
    """''' + MARK + ''' one line per JITFunction.run call at the cache
    lookup: computed key, hit/miss, per-parameter specialization
    (type:attr — attr is where pointer alignment and integer
    divisibility live), per-argument data_ptr() % 16, warps/stages.
    Diagnostic for the P6.28 r108u residual miss (see kit README)."""
    import os
    import time

    try:
        spec = []
        for p, sp in zip(fn_obj.params, specialization):
            try:
                typ, attr = sp[0], sp[1]
            except Exception:
                typ, attr = sp, ""
            spec.append("%s:%s:%s" % (p.name, typ, attr))
        ptr = [str(int(a.data_ptr()) % 16) if hasattr(a, "data_ptr") else "-"
               for a in bound_args.values()]
        # pid because print() bypasses the logger: both TP ranks share the
        # serve stdout and logger lines carry the (Worker_TPr pid=...) tag
        # these lines would otherwise lack.
        print("ARCJIT_KEY t=%s pid=%d fn=%s key=%s hit=%s warps=%s "
              "stages=%s spec=[%s] ptr%%16=[%s]" % (
                  time.strftime("%H:%M:%S"), os.getpid(), fn_obj._fn_name,
                  key, kernel is not None, getattr(options, "num_warps", "?"),
                  getattr(options, "num_stages", "?"), " ".join(spec),
                  " ".join(ptr)), flush=True)
    except Exception:
        pass


'''

CALL_BLOCK = (
    "        # " + MARK + """ diagnostic key/hit trace at the cache lookup
        # (see _arc_xpu_jitkey_trace above; env ARC_XPU_TRACE_JITKEY).
        if _ARC_JITKEY_ON:
            _arc_xpu_jitkey_trace(self, key, kernel, specialization,
                                  options, bound_args)
"""
)


def find_jit_py() -> pathlib.Path:
    spec = importlib.util.find_spec("triton.runtime.jit")
    if spec is None or not spec.origin:
        raise SystemExit("FATAL: triton.runtime.jit not importable")
    return pathlib.Path(spec.origin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="triton/runtime/jit.py (default: locate wheel)")
    ap.add_argument("--audit", action="store_true", help="report only")
    args = ap.parse_args()

    path = pathlib.Path(args.path) if args.path else find_jit_py()
    src = path.read_text(encoding="utf-8")
    marks = src.count(MARK)

    if marks:
        if marks == 3:
            print(f"AUDIT: already patched ({marks} marks): {path}")
            return 0
        print(f"FATAL: unexpected mark count {marks} (expected 0 or 3): {path}")
        return 1
    if src.count(ANCHOR_MODULE) != 1:
        print(f"FATAL: module anchor not unique: {path}")
        return 1
    if src.count(ANCHOR_RUN) != 1:
        print(f"FATAL: run anchor not unique: {path}")
        return 1

    print(f"AUDIT: pristine (0 marks), anchors unique: {path}")
    if args.audit:
        return 0

    out = src.replace(ANCHOR_MODULE, HELPER_BLOCK + ANCHOR_MODULE, 1)
    out = out.replace(ANCHOR_RUN, ANCHOR_RUN + CALL_BLOCK, 1)

    if out.count(MARK) != 3:
        print(f"FATAL: post-patch mark count {out.count(MARK)} != 3")
        return 1
    if out.count(ANCHOR_MODULE) != 1 or out.count(ANCHOR_RUN) != 1:
        print("FATAL: anchors disturbed by patch")
        return 1

    backup = path.with_suffix(".py.arcxpubak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(out, encoding="utf-8", newline="")
    py_compile.compile(str(path), doraise=True)
    print(f"PATCHED: 2 hunks, 3 marks, py_compile OK, backup {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
