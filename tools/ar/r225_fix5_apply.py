#!/usr/bin/env python3
"""r225 fix 5 apply -- sentinel LSE at one KV split (S==1).

Root cause (r225l on the build24 wheel): the epilogue writes the split
LSE only on `is_single_split || num_kv_splits > 1`. At FORCED
num_splits_kv == 1 both arms are false (is_single_split requires S > 1),
so exp_sums/max_logits stay unwritten at::empty bytes while the O path
IS written (normalized, into the fp32 temp). r225 delta (4) removed the
S==1 `temp_out = out` alias and made the reduce the only copy to `out`
-- so the reduce now runs at S==1, reads the unwritten LSE slot, and a
garbage/negative exp_sum skips the lone partial: 0/0 -> the whole
output NaN (r225l: nsplit1 mism 3072/3072 on every payload; auto/2/8
clean -- the auto heuristic never picks S==1 here).

Fix: extend the sentinel arm's predicate to
`is_single_split || num_kv_splits <= 1` -- exactly the predicate the
same epilogue uses two paragraphs below to decide whether O is
normalized. Whenever O is normalized the LSE is the pass-through
sentinel (exp_sums = 1, max_logits = 0): the reduce rescales by
exp2(0-0) = 1 and divides by 1, i.e. bitwise the old single-split
numerics (fp32 normalize + one bf16 round).

One edit, one file: csrc/xpu/attn/xe_2/collective/chunk_prefill_epilogue.hpp.
Writes a .r225fix4prebak beside it. Run from /src root (or the tree's
csrc parent).
"""

import os
import sys

REL = "csrc/xpu/attn/xe_2/collective/chunk_prefill_epilogue.hpp"

OLD = """    // Always store exp sum and max logits for current KV split.
    // assume seq_len_qo == 1
    if (row_valid) {
      if (is_single_split) {
        // Sentinel values: make ReduceSplitK a pass-through copy.
        exp_sums(q_row, idx_kv_split) = ElementA(1);
        max_logits(q_row, idx_kv_split) = ElementA(0);
      } else if (num_kv_splits > 1) {
"""

NEW = """    // Always store exp sum and max logits for current KV split.
    // assume seq_len_qo == 1
    if (row_valid) {
      // r225 delta (5): the sentinel arm mirrors the O-normalize
      // predicate below (`is_single_split || num_kv_splits <= 1`).
      // delta (4) removed the S==1 temp alias and made the reduce the
      // only copy to out, so S==1 must also park the pass-through
      // sentinel -- the old `else if (num_kv_splits > 1)` left the
      // slot unwritten and the reduce read at::empty bytes (0/0 ->
      // NaN at every forced-nsplit1 cell, r225l).
      if (is_single_split || num_kv_splits <= 1) {
        // Sentinel values: make ReduceSplitK a pass-through copy.
        exp_sums(q_row, idx_kv_split) = ElementA(1);
        max_logits(q_row, idx_kv_split) = ElementA(0);
      } else if (num_kv_splits > 1) {
"""


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    path = os.path.join(root, REL)
    with open(path, encoding="utf-8", newline="") as f:
        src = f.read()
    if src.count(OLD) != 1:
        print(f"ANCHOR_FAIL: count={src.count(OLD)}")
        return 1
    with open(path + ".r225fix4prebak", "w", encoding="utf-8",
              newline="") as f:
        f.write(src)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(src.replace(OLD, NEW, 1))
    print(f"EDIT_OK {REL} (1 edit)")
    print("R225_FIX5_APPLY_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
