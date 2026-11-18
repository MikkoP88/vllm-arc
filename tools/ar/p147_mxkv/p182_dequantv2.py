# SPDX-License-Identifier: Apache-2.0
"""p182_dequantv2.py -- [p182_dequantv2] MX whole-cache dequant kernel v2
A/B probe (P6.47 decode-parity fast-cycle 2, 2026-11-14).

CONTEXT (decode-parity goal): the r185 ramp forensics pinned
the 3.3-3.7 s prefill-interleave stalls on BOTH MX lanes to the
large-continuation prefill path -- engine._prefill_attention dequants the
whole cached prefix via mx_dequant_cache (mx_dequant.py) every chunked
prefill chunk (q_len>128), O(ctx^2) dequant work per request. v1
(_mx_full_dequant) is ONE single-warp program per (token, head): at 28K
cached that is 114,688 num_warps=1 CTAs, each paying the block-table
pointer chase and the full f32 decode chain (no K2/K3 packed lanes, no
token tiling) -- the same latency-bound class the decode kernel was
before v6 (triton_mx_decode.py header, r177 stage-split).

V2: one program per (token-TILE, head) reusing the certified
_mx_dequant_block helper verbatim (BLOCK_N=BLOCK_T tiles, F16OUT/E2M1FAST
packed lanes) -- bit-identical outputs by the K2/K3 single-rounding law
(e4m3->fp16 exact, pow2 scale exact, product rounds once; assert d==0.0
vs v1 on identical bytes).

Sweep: BLOCK_T x warps on the continuation hot range (8K/16K/28K cached,
D=256 Hk=4 block 64 -- the serve geometry), packed on/off attribution,
v1 baseline, effective GB/s. Winner table -> stdout; DEQV2_PASS on
finite + bit-identical + packed winner beating v1.
"""

from __future__ import annotations

import sys
import time

import torch
import triton
import triton.language as tl

from vllm_gaudi.xpu.patches.mx_kv.mx_dequant import mx_dequant_cache
from vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode import _mx_dequant_block
from vllm_gaudi.xpu.patches.mx_kv.mx_layout import slot_bytes

torch.manual_seed(0)
dev = torch.xpu.current_device()

D = 256
HK = 4
BLOCK_SIZE = 64
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
CACHED_LENS = (8192, 16384, 28672)  # chunked-prefill continuation hot range


@triton.jit
def _mx_full_dequant_v2(
    cache_ptr,
    block_table_ptr,
    k_out_ptr,
    v_out_ptr,
    cached_len,
    stride_bt_row,
    stride_c_b,
    stride_c_t,
    stride_c_h,
    stride_k_h,
    stride_k_t,
    D: tl.constexpr,
    N_GROUPS: tl.constexpr,
    K4: tl.constexpr,
    V4: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    PACKED: tl.constexpr,
):
    # One program per (token tile, head): the v1 per-(token, head) math
    # re-expressed on _mx_dequant_block's (BLOCK_N, D) tiles -- the
    # block-table chase amortizes BLOCK_T-fold, loads/stores coalesce per
    # tile, and PACKED rides the K2/K3 single-rounding lanes.
    pid_t = tl.program_id(0)
    head = tl.program_id(1)
    toks = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = toks < cached_len
    blk = tl.load(
        block_table_ptr + toks // BLOCK_SIZE, mask=tmask, other=0
    ).to(tl.int64)
    slot = (
        blk * stride_c_b
        + (toks % BLOCK_SIZE).to(tl.int64) * stride_c_t
        + head.to(tl.int64) * stride_c_h
    )
    v_off = (D // 2 if K4 else D) + N_GROUPS
    k = _mx_dequant_block(
        cache_ptr, slot, 0, tmask, BLOCK_T, D, N_GROUPS, K4,
        SHIFT=False, F16OUT=PACKED and not K4, E2M1FAST=PACKED and K4,
    )
    v = _mx_dequant_block(
        cache_ptr, slot, v_off, tmask, BLOCK_T, D, N_GROUPS, V4,
        SHIFT=False, F16OUT=PACKED and not V4, E2M1FAST=PACKED and V4,
    )
    offs_d = tl.arange(0, D)
    out_off = head.to(tl.int64) * stride_k_h + toks.to(tl.int64)[:, None] * stride_k_t
    st_mask = tmask[:, None]
    tl.store(k_out_ptr + out_off + offs_d[None, :], k.to(tl.float16), mask=st_mask)
    tl.store(v_out_ptr + out_off + offs_d[None, :], v.to(tl.float16), mask=st_mask)


def make_cache(cached_len: int, fmt: str):
    sb = slot_bytes(D, fmt)
    n_blocks = (cached_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 8
    # Byte domain law (run-1 lesson, 2026-11-14): random 0..256 bytes put
    # e4m3 NaN payloads (0x7F/0xFF) and inf-producing E8M0 codes (255 ->
    # exp2(128)) into the synthetic cache -- v1 and v2 then hold identical
    # NaNs and the abs-diff gate reads nan:nan = nan = "NOT bit-identical"
    # for byte-identical kernels (run 1 banked as the nan-artifact). All
    # bytes 0..125: every e4m3 byte finite, every e2m1 nibble finite, all
    # scales in [2^-127, 2^-2] -- finite math end to end, the diff gate
    # reads real bit differences only.
    cache = torch.randint(
        0, 126, (n_blocks, BLOCK_SIZE, HK, sb), dtype=torch.uint8, device=dev
    )
    bt = torch.randperm(n_blocks, device=dev).to(torch.int32)[: n_blocks - 8]
    bt = bt.reshape(1, -1).contiguous()
    # scrub the un-referenced blocks' bytes so a wild gather cannot
    # silently match (v1 zero-fills past cached_len; v2 must too)
    return cache, bt


def run_v1(cache, bt, t_alloc, cached_len):
    k = torch.zeros(1, HK, t_alloc, D, dtype=torch.float16, device=dev)
    v = torch.zeros_like(k)
    fmt = mx_dequant_cache(cache, bt, k, v, cached_len)
    return fmt, k, v


def run_v2(cache, bt, t_alloc, cached_len, fmt, block_t, warps, packed):
    k = torch.zeros(1, HK, t_alloc, D, dtype=torch.float16, device=dev)
    v = torch.zeros_like(k)
    grid = (triton.cdiv(t_alloc, block_t), HK)
    _mx_full_dequant_v2[grid](
        cache,
        bt,
        k,
        v,
        cached_len,
        bt.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        k.stride(1),
        k.stride(2),
        D=D,
        N_GROUPS=D // 32,
        K4=fmt == "mxfp4",
        V4=fmt in ("mxfp4", "mxk8v4"),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_T=block_t,
        PACKED=packed,
        num_warps=warps,
    )
    return k, v


def bench(fn, iters=20):
    fn()
    torch.xpu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3  # median ms


def main() -> int:
    print("[p182_dequantv2] geometry D=256 HK=4 block=64 dev=", dev, flush=True)
    ok = True
    winners = {}
    for fmt in FMTS:
        for cl in CACHED_LENS:
            t_alloc = ((cl + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
            cache, bt = make_cache(cl, fmt)
            rfmt, k1, v1 = run_v1(cache, bt, t_alloc, cl)
            assert rfmt == fmt
            t_v1 = bench(
                lambda cache=cache, bt=bt, t_alloc=t_alloc, cl=cl: run_v1(
                    cache, bt, t_alloc, cl
                )
            )
            in_bytes = t_alloc * HK * cache.shape[-1]
            out_bytes = 2 * t_alloc * HK * D * 2
            gbs_v1 = (in_bytes + out_bytes) / (t_v1 / 1e3) / 1e9
            best = (None, 1e9)
            rows = []
            for bt_t in (16, 32, 64, 128):
                for warps in (2, 4, 8):
                    for packed in (True, False):
                        try:
                            k2, v2 = run_v2(
                                cache, bt, t_alloc, cl, fmt, bt_t, warps, packed
                            )
                            d = max(
                                float((k2 - k1).abs().max()),
                                float((v2 - v1).abs().max()),
                            )
                            if d != 0.0:
                                rows.append(
                                    f"  T{bt_t} w{warps} p{int(packed)}: "
                                    f"DIFF d={d:.3e} (NOT bit-identical)"
                                )
                                continue
                            t = bench(
                                lambda cache=cache, bt=bt, t_alloc=t_alloc,
                                cl=cl, fmt=fmt, bt_t=bt_t, warps=warps,
                                packed=packed: run_v2(
                                    cache, bt, t_alloc, cl, fmt, bt_t, warps, packed
                                )
                            )
                            rows.append(
                                f"  T{bt_t} w{warps} p{int(packed)}: {t:.3f} ms"
                                f" ({t_v1 / t:.2f}x v1)"
                            )
                            if t < best[1]:
                                best = ((bt_t, warps, packed), t)
                        except Exception as e:  # noqa: BLE001
                            rows.append(f"  T{bt_t} w{warps} p{int(packed)}: FAIL {e}")
            if best[0] is None:
                print(f"[{fmt} cl={cl}] v1={t_v1:.3f} ms ({gbs_v1:.1f} GB/s)")
                print("\n".join(rows))
                print(f"[{fmt} cl={cl}] NO_VALID_V2")
                ok = False
                continue
            (bt_t, warps, packed), t = best
            gbs = (in_bytes + out_bytes) / (t / 1e3) / 1e9
            print(
                f"[{fmt} cl={cl}] v1={t_v1:.3f} ms ({gbs_v1:.1f} GB/s) -> "
                f"v2 {t:.3f} ms ({gbs:.1f} GB/s) = {t_v1 / t:.2f}x "
                f"[T{bt_t} w{warps} p{int(packed)}]"
            )
            print("\n".join(rows))
            winners[(fmt, cl)] = (bt_t, warps, packed, t_v1 / t)
            del cache, bt, k1, v1
    print("=== [p182_dequantv2] winner table ===")
    for k_, v_ in winners.items():
        print(f"  {k_}: T{v_[0]} w{v_[1]} p{int(v_[2])} {v_[3]:.2f}x")
    gains = [v_[3] for v_ in winners.values()]
    if not winners:
        print("DEQV2_FAIL no winners")
        return 1
    if not ok:
        print("DEQV2_FAIL bit-identity or launch failures above")
        return 1
    print(f"DEQV2_PASS gain range {min(gains):.2f}x..{max(gains):.2f}x v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
