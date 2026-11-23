#!/usr/bin/env python3
"""r231b_apply.py -- P6.47 r231b wheel patch (Q pre-stage to SLM).

Lever (r231a verdict, ranked lever 1 = walk inner-loop ALU reduction,
format-common component): the MX decode mainloop re-derives the
K-INVARIANT Q A-operand at every (k-tile, d-tile) -- copy_q (a gmem
tile load) plus reorder(tQrQ, tSrQ) (the full cross-WI layout-change
machinery) each time: 4 per 16-row k-tile, i.e. per call at 32K ctx on
the served _8/_256/_16 policy: 2048 k-tiles x 4 = 8192 redundant
copies+reorders of a 512-element fragment. The r212w probe priced a
same-class reorder at 595 us of a 32K walk (the K-lane reorder it
deleted); the Q reorders ride the same machinery at half the fragment
size.

Patch: stage each d-tile's reordered fragment to SLM ONCE per body()
call (SG0 writes float rows at wi16, one barrier), reload per use --
plain SLM reads + a float->bf16 convert. The staged operand is
BITWISE-IDENTICAL: the bf16->float widening at stage is exact and the
float->bf16 ctor of the same value returns identical bits (round of an
exactly-representable value is itself); the MMA consumes the same bits
in the same order. Gates: r231b offline leg (bitwise build25-vs-
build27 oracle + r229 calibrated band cells + the r231 split ladder).

Run ON THE RIG against the live tree /root/build/vxk-arc-src (builder
/src law). Refuses unless the pristine mainloop md5 matches; leaves a
.r231bprebak sibling (tree convention).
"""

import hashlib
import sys

TARGET = (
    "/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/"
    "mx_paged_decode_mainloop.hpp"
)
PRISTINE_MD5 = "a02286184bdb07a088e0321a22bb0111"

# --- edit 1: SharedStorage gains the q_stage image -------------------
A1_OLD = """    alignas(16) uint8_t v_stage[get<1>(TileShapeQK{}) * kVStagePitch];
  };
"""
A1_NEW = """    alignas(16) uint8_t v_stage[get<1>(TileShapeQK{}) * kVStagePitch];
    // r231b Q PRE-STAGE: the K-invariant reordered A fragments (Q is
    // decode-static), staged ONCE per body() call as floats and
    // reloaded per (k-tile, d-tile) use. The bf16->float widening is
    // exact and the reload's float->bf16 ctor of the same value returns
    // identical bits -- the operand is bitwise-identical to the stock
    // per-use reorder() result. Sized for the widest compiled D (256 =
    // 4 d-tiles); D=128 lanes use half. Rows are SG-relative (the A
    // operand is replicated across k-band subgroups; SG0 stages,
    // everyone reads after the one barrier at the stage site).
    static constexpr int kQSRowFloats =
        get<0>(TileShapeQK{}) * kDTileElems / intel::sg_size;
    static constexpr int kQSMaxDTiles = 4;
    alignas(16) float
        q_stage[kQSMaxDTiles * kQSRowFloats * intel::sg_size];
  };
"""

# --- edit 2: the pre-stage block (after the Q prefetch) ---------------
A2_OLD = """    /* Q prefetch */
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }
"""
A2_NEW = """    /* Q prefetch */
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    /* r231b Q PRE-STAGE: Q is K-invariant (decode) yet the stock body
    re-derived the A operand at every (k-tile, d-tile) -- a gmem tile
    copy plus a full reorder() layout change each time (4 per 16-row
    k-tile; the r212w probe priced a same-class reorder at 595 us of a
    32K walk). Stage each d-tile's reordered fragment to SLM once
    (SG0 writes, wi16 float rows), one barrier, reload per use in the
    K loop -- the in-loop cost drops to plain SLM reads. The staged
    operand is bitwise-identical (see SharedStorage). */
    float* qst = reinterpret_cast<float*>(&smem.q_stage);
    for (int D = 0; D < size<3>(tQgQ); D++) {
      copy(copy_q, tQgQ(_, _, _, D), tQrQ);
      reorder(tQrQ, tSrQ);
      if (thr_id < intel::sg_size) {
        float* qd =
            qst + D * int(tSrQ.size()) * intel::sg_size +
            wi16 * int(tSrQ.size());
        CUTE_UNROLL
        for (int i = 0; i < int(tSrQ.size()); ++i) {
          qd[i] = float(tSrQ(i));
        }
      }
    }
    barrier();
"""

# --- edit 3: the in-loop copy+reorder retires -------------------------
A3_OLD = """      for (int D = 0; D < size<4>(tKDgKD); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        reorder(tQrQ, tSrQ);
"""
A3_NEW = """      for (int D = 0; D < size<4>(tKDgKD); D++) {
        // r231b: reload the staged K-invariant A fragment (see the Q
        // PRE-STAGE block above) -- the per-(k-tile, d-tile)
        // copy+reorder retires; the operand bits are unchanged.
        {
          const float* qd =
              qst + D * int(tSrQ.size()) * intel::sg_size +
              wi16 * int(tSrQ.size());
          CUTE_UNROLL
          for (int i = 0; i < int(tSrQ.size()); ++i) {
            tSrQ(i) = typename decltype(tSrQ)::value_type(qd[i]);
          }
        }
"""

EDITS = (
    ("in-loop reload", A3_OLD, A3_NEW),
    ("pre-stage block", A2_OLD, A2_NEW),
    ("shared-storage q_stage", A1_OLD, A1_NEW),
)


def main() -> int:
    try:
        with open(TARGET, "rb") as f:
            data = f.read()
    except OSError as e:
        print(f"R231B_ABORT: cannot read {TARGET}: {e}")
        return 1
    md5 = hashlib.md5(data).hexdigest()
    if "r231b" in data.decode("utf-8", "replace"):
        print("R231B_ABORT: tree already carries an r231b marker")
        return 1
    if md5 != PRISTINE_MD5:
        print(f"R231B_ABORT: pristine md5 mismatch {md5} != {PRISTINE_MD5}")
        return 1

    text = data.decode("utf-8")
    for name, old, new in EDITS:
        n = text.count(old)
        if n != 1:
            print(f"R231B_ABORT: anchor {name!r} matched {n} times (want 1)")
            return 1
        text = text.replace(old, new, 1)
        print(f"R231B_APPLIED {name}")

    with open(TARGET + ".r231bprebak", "wb") as f:
        f.write(data)
    with open(TARGET, "w", newline="\n") as f:
        f.write(text)
    print(f"R231B_DONE md5={hashlib.md5(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
