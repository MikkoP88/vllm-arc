#!/usr/bin/env python3
"""r231d_apply.py -- P6.47 r231d wheel patch (seqused-tightened splits).

Lever (r231c lever 1, the largest measured remaining delta on the MX
native call): the host split count prices from max_seqlen_k CAPACITY
(the r212f no-sync law -- seqused_k is device-resident, reading it on
the host would sync) so every shallow-ctx decode runs ~128 splits:
r231c v2 measured +92% over-provisioning at 2K walked ctx (auto
428-448 us vs ns32 202-244) evaporating by 8K and essential at 32K.

Patch: clamp the split count ON DEVICE from the windowed k-block
count both kernels already compute -- >=4 k-tiles per split:
    seq_num_kv_splits = min(num_kv_splits, max(1, windowed_k_blocks/4))
at the legacy path only (splits_per_seq == nullptr; a host plan stays
trusted). MIRRORED in mx_paged_decode_kernel.hpp AND mx_reduce_split:
the reduce must derive the IDENTICAL count or it would iterate past
the last written Oaccum/exp_sums slot (the stale-buffer class). The
buffers keep their capacity layout (offsets/shape still size from the
host num_kv_splits) -- exactly the env-pin semantics, derived per-seq.

Measured anchors (quota 4): 2K (128 blocks) -> S=32 = the measured
optimum (-52%); 8K/32K -> S=128 = capacity, byte-identical execution
(record-neutral by construction); every split owns exactly 4 k-tiles
at mid ctx; <512 walked tokens the untouched kMinBlocksForSplit guard
keeps single-split (one knob per leg). Gates: cross-wheel bitwise
build25-ns32 == build26-auto @2K (clamp == pin, identical partition),
build25-auto == build26-auto @32K (record identity), 12 r229
fp64-band cells, the r231a-identical ladder.

Run ON THE RIG against the live tree /root/build/vxk-arc-src (builder
/src law). Refuses unless both pristine md5s match; leaves .r231dprebak
siblings (tree convention).
"""

import hashlib
import sys

BASE = "/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/"

KERNEL = BASE + "mx_paged_decode_kernel.hpp"
KERNEL_MD5 = "80f88dd89574c8c0e9225472ea5194a0"

REDUCE = BASE + "mx_reduce_split.hpp"
REDUCE_MD5 = "017cbc805304fe7e9bedd0f915bbf471"

# --- edit 1: the FMHA kernel's legacy split derivation ----------------
# (anchored on the `if (idx_kv_split` follower -- the worklist branch
# carries the same ternary but is followed by is_single_split and
# keeps stock semantics)
K1_OLD = """        seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                ? p.splits_per_seq[idx_b]
                                : num_kv_splits;

        if (idx_kv_split >= seq_num_kv_splits) {
"""
K1_NEW = """        // r231d SEQUSED-TIGHTENED SPLITS (r231c pricing): the host
        // count prices from max_seqlen_k capacity (the r212f no-sync
        // law) and over-provisions at shallow walked ctx (+92% @2K);
        // clamp on the device-side windowed block count -- >=4
        // k-tiles per split. 8K+ ctx is unchanged (the clamp hits
        // capacity) and MxReduceSplitK derives the identical count
        // (mirrored edit) or it would read unwritten Oaccum slots.
        seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                ? p.splits_per_seq[idx_b]
                                : cute::min(num_kv_splits,
                                            cute::max(1,
                                                      windowed_k_blocks /
                                                          4));

        if (idx_kv_split >= seq_num_kv_splits) {
"""

# --- edit 2: the reduce kernel's derivation (the mirror) --------------
R1_OLD = """      // Per-sequence adaptive split count
      int seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                  ? p.splits_per_seq[idx_b]
                                  : num_kv_splits;
"""
R1_NEW = """      // Per-sequence adaptive split count
      // r231d MIRROR (seqused-tightened splits): must derive the SAME
      // clamped count the FMHA kernel derived for this seq -- the
      // legacy path clamps on the windowed block count (>=4 k-tiles
      // per split) so shallow-ctx decode stops over-provisioning
      // splits it then has to memset/reduce; a count mismatch here
      // would iterate past the last written Oaccum slot.
      int seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                  ? p.splits_per_seq[idx_b]
                                  : cute::min(num_kv_splits,
                                              cute::max(1,
                                                        windowed_k_blocks /
                                                            4));
"""


def patch(path, pristine_md5, edits):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        print(f"R231D_ABORT: cannot read {path}: {e}")
        return None
    md5 = hashlib.md5(data).hexdigest()
    if "r231d" in data.decode("utf-8", "replace"):
        print(f"R231D_ABORT: {path} already carries an r231d marker")
        return None
    if md5 != pristine_md5:
        print(f"R231D_ABORT: {path} md5 {md5} != {pristine_md5}")
        return None
    text = data.decode("utf-8")
    for name, old, new in edits:
        n = text.count(old)
        if n != 1:
            print(f"R231D_ABORT: {path}: anchor {name!r} matched {n} times (want 1)")
            return None
        text = text.replace(old, new, 1)
        print(f"R231D_APPLIED {path.rsplit('/', 1)[-1]}: {name}")
    with open(path + ".r231dprebak", "wb") as f:
        f.write(data)
    with open(path, "w", newline="\n") as f:
        f.write(text)
    out_md5 = hashlib.md5(text.encode()).hexdigest()
    print(f"R231D_PATCHED {path.rsplit('/', 1)[-1]} md5={out_md5}")
    return out_md5


def main() -> int:
    ok = patch(KERNEL, KERNEL_MD5, (("legacy split clamp", K1_OLD, K1_NEW),))
    if ok is None:
        return 1
    ok = patch(REDUCE, REDUCE_MD5, (("reduce mirror clamp", R1_OLD, R1_NEW),))
    if ok is None:
        return 1
    print("R231D_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
