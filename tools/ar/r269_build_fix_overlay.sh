#!/bin/bash
# r269_build_fix_overlay.sh -- run LOCALLY (git-bash) to build the r269
# FIX overlay dir: a byte-copy of the tree's vllm_gaudi/xpu/patches/mx_kv
# package plus the D-B class fix at the WRAPPER level (all three decode
# entry points -- mx_decode, mx_decode_rows, mx_decode_rows_dot -- take
# block_size: int | None = None and resolve None to the cache's OWN page
# cache.shape[1] at body start).
#
# WHY the wrapper, not the engine call site: the engine has FIVE
# mx_decode-family call sites that all omit block_size (engine.py 354
# A/B-dump ref, 823 the decode dispatch, 995 split-row continuation,
# 1239 small-continuation synth_bt, 1546 the verify else-branch -- and
# 1528/1537 ride the rows/rows_dot wrappers with the same default).
# The wrapper default 64 becomes the kernel BLOCK_SIZE
# (triton_mx_decode.py: bt[b, offs_n // BLOCK_SIZE] + offs_n %
# BLOCK_SIZE pitch, kernel lines 387/393 + siblings), so the gemma
# p128 global pool is walked HALF-page at seq_len>64 -- wrong rows
# from blocks that hold tokens 128+ (never-written torch.empty bytes
# -> inf/NaN scales -> the r266 c3/c4 400-nan + 'our' attractor) and
# off-table block ids past 2x table width. qwen's p64 pools equal the
# default, so every qwen lane stayed green (m8g/m4g/mkg stay valid).
# One wrapper fix closes every current and future call site;
# explicit-page callers (the p242 harness, the canary's pagefix arm)
# are byte-identical in behavior.
#
# This overlay is the r269 fix-gate VEHICLE (serve_arc_b70.sh's
# ARC_MXKV_OVERLAY knob mounts it ro over the image's installed
# package): boot the c3 posture with the fix and A/B against the
# banked c3 RED (full p266 matrix probe).  The tree itself stays
# unfixed until the leg evidence lands (banking directive); the
# r268 TRACE overlay (shipped, engine.py 0ce0954c) is built from
# unfixed tree bytes and stays byte-pure.
#
# Output: tools/ar/p269_fix/overlay/ (untracked, deterministic regen
# from tracked inputs; .git/info/exclude) + md5 manifest vs the tree.
# Ship via: scp -r tools/ar/p269_fix/overlay root@RIG:/root/build/r269_fix
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/../../vllm_gaudi/xpu/patches/mx_kv
OUT=$HERE/p269_fix/overlay

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$SRC"/*.py "$OUT/"

python - <<'PYEOF'
from pathlib import Path

out = Path("tools/ar/p269_fix/overlay")
td = out / "triton_mx_decode.py"
t = td.read_text()

SIG_OLD = "    block_size: int = 64,"
assert t.count(SIG_OLD) == 3, f"signature anchor x{t.count(SIG_OLD)}, expected 3"
SIG_NEW = "    block_size: int | None = None,  # r269: None = the cache's own page"
t = t.replace(SIG_OLD, SIG_NEW)

RESOLVE = """    if block_size is None:
        # r269 D-B fix: the cache's OWN page. The old default 64 walked
        # page-128 pools HALF-page at seq_len>64 (block bt[j//64], pos
        # j%64 -- wrong rows / off-table walks on the gemma global pool;
        # qwen's p64 pools were immune). Explicit-page callers unchanged.
        block_size = cache.shape[1]
"""


def patch_fn(text: str, header: str, anchor: str) -> str:
    """Insert the None-resolution before the FIRST body anchor of the
    function identified by its def header (segments keep the three
    same-named wrappers independent)."""
    i = text.index(header)
    j = text.index(anchor, i)
    return text[:j] + RESOLVE + text[j:]


# mx_decode: body starts at the decoder pick
t = patch_fn(t, "def mx_decode(", "    dec = decoder or _DECODER")
# mx_decode_rows: body resolves format from the cache slot width
t = patch_fn(t, "def mx_decode_rows(", "    hk = cache.shape[2]")
# mx_decode_rows_dot: body starts at the shape unpack
t = patch_fn(t, "def mx_decode_rows_dot(", "    r, hq, d = q.shape")

td.write_text(t, newline="\n")
print("overlay triton_mx_decode.py D-B-fixed (3 wrappers, None->cache page)")
PYEOF

python -m py_compile "$OUT"/*.py && echo PYCOMPILE_OK
rm -rf "$OUT/__pycache__"  # never ship stale pyc next to shadowed sources

echo "== overlay vs tree md5 (triton_mx_decode.py MUST differ; all others MUST match):"
( cd "$SRC" && md5sum *.py ) > /tmp/r269_tree_sums
( cd "$OUT" && md5sum *.py ) > /tmp/r269_overlay_sums
diff /tmp/r269_tree_sums /tmp/r269_overlay_sums || true
echo "R269_FIX_OVERLAY_BUILT $OUT"
