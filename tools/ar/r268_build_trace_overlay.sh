#!/bin/bash
# r268_build_trace_overlay.sh -- run LOCALLY (git-bash) to build the
# r268 TRACED mx_kv overlay dir: a byte-copy of the tree's
# vllm_gaudi/xpu/patches/mx_kv package plus the decode-side trace
# lines the r268 trace leg needs (the store-side smap trace already
# ships in engine.py -- ARC_MXKV_TRACE=1 gates it).  The overlay
# mounts read-only over the image's installed package via the
# serve_arc_b70.sh ARC_MXKV_OVERLAY knob -- NO image rebuild, NO
# wheel reinstall; the python surface is the fork's own.
#
# WHY a decode-side trace: e2 (2026-11-28 04:15Z) eliminated the
# XPU-graph hypothesis -- eager MX reproduces c3 byte-identically, so
# the defect is in the EAGER decode-step store/walk.  The plain decode
# path reads attn_metadata.block_table + seq_lens while the store
# writes slot_mapping-derived slots, and prefill's first chunk NEVER
# reads the cache (raw flash, engine.py ~869) -- a wrong-table or
# wrong-pool-space read is invisible through prefill and lands
# exactly at token 2.  The r267 canary covers store/kernel math
# offline; IF the canary PASSES, this overlay + one c3-posture leg +
# a 2-prompt x 8-token workload + ARC_MXKV_TRACE=1 gives the direct
# G3 readout: per decode step, the store's smap slot ids vs the
# walk's bt row ids / seq_lens / cache shape.  Small-B only (B<=4) --
# the .tolist() D2H syncs are debug-leg-only (the item-d2h law).
#
# Output: tools/ar/p268_trace/overlay/ (the full package copy, traced)
# + a md5 manifest vs the tree (ship via scp -r, then
# ARC_MXKV_OVERLAY=/root/build/r268_overlay on the leg).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/../../vllm_gaudi/xpu/patches/mx_kv
OUT=$HERE/p268_trace/overlay

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$SRC"/*.py "$OUT/"
rm -rf "$OUT/__pycache__"

python - <<'PYEOF'
from pathlib import Path

out = Path("tools/ar/p268_trace/overlay")
eng = out / "engine.py"
t = eng.read_text()

# 1) decode-side trace: the walk's bt row ids + seq_lens + pool shape,
#    small-B only (debug leg -- the D2H sync law)
old = """        _trace(
            "dec", f"B={query.shape[0]}", f"msl={attn_metadata.max_seq_len}",
            f"id={id(self)}",
        )"""
new = """        _trace(
            "dec", f"B={query.shape[0]}", f"msl={attn_metadata.max_seq_len}",
            f"id={id(self)}",
        )
        if _TRACE and query.shape[0] <= 4:
            # r268 trace leg: the walk addressing vs the store's slots.
            # EAGER ONLY (the e2 law: ARC_GRAPH=0 reproduces the c3
            # corruption; under capture this python runs once over the
            # dummy batch -- replays never re-run it, so live G3 values
            # would never print).
            _bt = attn_metadata.block_table
            _sl = attn_metadata.seq_lens
            _trace(
                "decx",
                f"B={query.shape[0]}",
                f"bt0={_bt[0, :8].tolist()}",
                f"bt1={_bt[1, :8].tolist() if _bt.shape[0] > 1 else None}",
                f"bts={_bt.stride()}",
                f"btc={int(_bt.is_contiguous())}",
                f"sl={_sl[:4].tolist()}",
                f"msl={attn_metadata.max_seq_len}",
                f"cache={tuple(kv_cache.shape)}",
                f"stride={kv_cache.stride()}",
                f"id={id(self)}",
            )"""
assert t.count(old) == 1, "decode trace anchor not found exactly once"
t = t.replace(old, new)

# 2) store-source trace: the REAL k/v layouts at every store call
#    (the G2 readout: prefill [N,Hk,D] contiguous vs decode-step
#    single rows with the model's actual views/strides; the wrapper
#    normalizes stride(2)!=1 only -- everything else reaches the
#    kernel as-is)
old2 = """            _trace(
                "store", f"cache={tuple(kv_cache.shape)}",
                f"stride={kv_cache.stride()}", f"slots={slot_mapping.shape[0]}",
                f"smap={_smap}",
                f"smax={int(slot_mapping.max()) if slot_mapping.numel() else -1}",
                f"bs_from_shape1={block_size}", f"id={id(self)}",
            )"""
new2 = """            _trace(
                "store", f"cache={tuple(kv_cache.shape)}",
                f"stride={kv_cache.stride()}", f"slots={slot_mapping.shape[0]}",
                f"smap={_smap}",
                f"smax={int(slot_mapping.max()) if slot_mapping.numel() else -1}",
                f"bs_from_shape1={block_size}",
                f"k={tuple(key.shape)}:{key.stride()}",
                f"v={tuple(value.shape)}:{value.stride()}",
                f"kd={key.dtype}",
                f"id={id(self)}",
            )"""
assert t.count(old2) == 1, "store trace anchor not found exactly once"
t = t.replace(old2, new2)

eng.write_text(t, newline="\n")
print("overlay engine.py traced (decx + store sources)")
PYEOF

python -m py_compile "$OUT"/*.py && echo PYCOMPILE_OK
echo "== overlay vs tree md5 (engine.py MUST differ; all others MUST match):"
( cd "$SRC" && md5sum *.py ) > /tmp/r268_tree_sums
( cd "$OUT" && md5sum *.py ) > /tmp/r268_overlay_sums
diff /tmp/r268_tree_sums /tmp/r268_overlay_sums || true
echo "R268_OVERLAY_BUILT $OUT"
