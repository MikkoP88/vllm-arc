#!/usr/bin/env python3
"""r225 fix 3 apply -- fp32 split partials through the decode/reduce O path.

Root cause (r225l + gate2): the multi-split partials buffer (temp_out)
was allocated bf16 (binding mirrored stock flash_api), quantizing each
split's unnormalized fp32 accumulator to 2^-9 relative BEFORE the
combine divides by the softmax sum. In a flat deep-layer decode softmax
(sum ~ 5e3) that shrinks |O| ~5000x under the same absolute partial
error -- hundreds-thousands of output ulp, 15-26% of elements off-grid.
A peaked softmax (sum ~ 1) masked it sub-ulp, which is why every
synthetic gate was clean while real payloads carried the residual, and
why nsplit=1 (no partial store; in-register fp32 normalize) was exact.

Delta (4), three fork-owned files:
  mx_paged_decode.hpp   ElementO role split (final bf16 vs fp32
                        partials), TensorO built on ElementOAccum,
                        reduce_arg out cast pinned to bf16, combine
                        always launched.
  mx_reduce_split.hpp   O write typed ElementOFinal (bf16); Oaccum
                        follows the decode kernel's fp32 ElementO.
  mx_paged_decode_xe2.cpp  temp_out always a dedicated fp32 buffer.

Run from /src root (or vxk-arc-src/csrc); writes .r225fix2prebak
backups beside each file.
"""

import os
import sys

BASE = "csrc/xpu/attn/xe_2/mx"

EDITS = {
    f"{BASE}/mx_paged_decode.hpp": [
        (
            """  using ElementQ = bfloat16_t;
  using ElementBytes = uint8_t;
  using ElementO = bfloat16_t;  // the out dtype; accumulators stay float
""",
            """  using ElementQ = bfloat16_t;
  using ElementBytes = uint8_t;
  // r225 delta (4): the O element roles split. FINAL out stays bf16,
  // but the decode kernel's O write -- the KV-split partials, or the
  // normalized value at one split -- lands in the temp buffer, which
  // the binding now allocates fp32: bf16 partials quantized every
  // split to 2^-9 relative BEFORE the combine divides by the softmax
  // sum, which a FLAT decode softmax (deep-layer Ks codes 119-122;
  // r225l) amplifies to hundreds-thousands of output ulp (sum ~ 5e3
  // shrinks |O| ~5000x under the same absolute partial error). A
  // PEAKED softmax (sum ~ 1) masked it sub-ulp -- why every synthetic
  // gate was clean while real deep-layer payloads carried 15-26%
  // off-grid elements. The reduce kernel is now the ONE bf16
  // quantization point.
  using ElementO = bfloat16_t;  // the FINAL out dtype (reduce write)
  using ElementOAccum = float;  // decode O write / split partials
""",
        ),
        (
            """    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));""",
            """    using TensorO =
        decltype(make_dummy_tensor(ElementOAccum{}, StrideO{}));""",
        ),
        (
            """         reinterpret_cast<ElementO*>(args.out),
         stride_O,""",
            """         reinterpret_cast<bfloat16_t*>(args.out),
         stride_O,""",
        ),
        (
            """    run(queue, params, reduce_params, args.num_kv_splits > 1);""",
            """    // r225 delta (4): the combine ALWAYS runs -- it is the single
    // bf16 quantization point now (fp32 partials in, final cast out;
    // one-split sequences ride the epilogue's sentinel pass-through).
    run(queue, params, reduce_params, /*need_reduce=*/true);""",
        ),
    ],
    f"{BASE}/mx_reduce_split.hpp": [
        (
            """  using ElementO = typename FMHAKernel_::ElementO;
""",
            """  using ElementO = typename FMHAKernel_::ElementO;
  // r225 delta (4): the O WRITE is the final bf16 out (the decode
  // kernel's ElementO became the fp32 partial type -- see
  // mx_paged_decode.hpp delta (4)); Oaccum follows it as fp32.
  using ElementOFinal = bfloat16_t;
""",
        ),
        (
            """    // outputs:
    ElementO* O;
    StrideO dO;""",
            """    // outputs:
    ElementOFinal* O;
    StrideO dO;""",
        ),
        (
            """        O(seq_idx, idx, head_q, l_coord) = static_cast<ElementO>(acc);""",
            """        O(seq_idx, idx, head_q, l_coord) =
            static_cast<ElementOFinal>(acc);""",
        ),
    ],
    f"{BASE}/mx_paged_decode_xe2.cpp": [
        (
            """  if (num_kv_splits == 1) {
    temp_out = out;
  } else if (!temp_out.defined()) {
    temp_out = at::empty(
        {total_seqlen_q, num_heads_q * num_kv_splits, head_size},
        query.options().device(query.device()));
  }""",
            """  // r225 delta (4): partials ALWAYS fp32 in a dedicated temp buffer.
  // The old alias (num_kv_splits == 1 -> temp_out = out) and the bf16
  // partial allocation quantized each split's unnormalized
  // accumulator to 2^-9 relative BEFORE the combine divided by the
  // softmax sum -- hundreds-thousands of output ulp in the flat
  // deep-layer decode regime (see mx_paged_decode.hpp delta (4)).
  // Single-split runs keep the fp32 buffer too; the reduce kernel is
  // the one bf16 quantization point (sentinel pass-through).
  if (!temp_out.defined()) {
    temp_out = at::empty(
        {total_seqlen_q, num_heads_q * num_kv_splits, head_size},
        query.options().dtype(at::kFloat).device(query.device()));
  }""",
        ),
    ],
}


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for rel, pairs in EDITS.items():
        path = os.path.join(root, rel)
        with open(path, encoding="utf-8", newline="") as f:
            src = f.read()
        for old, new in pairs:
            if src.count(old) != 1:
                print(f"ANCHOR_FAIL {rel}: count={src.count(old)} for "
                      f"{old.splitlines()[0][:60]}")
                return 1
            src = src.replace(old, new, 1)
        with open(path, encoding="utf-8", newline="") as f:
            orig = f.read()
        with open(path + ".r225fix2prebak", "w", encoding="utf-8",
                  newline="") as f:
            f.write(orig)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(src)
        print(f"EDIT_OK {rel} ({len(pairs)} edits)")
    print("R225_FIX3_APPLY_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
