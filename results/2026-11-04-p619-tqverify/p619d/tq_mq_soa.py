# SPDX-License-Identifier: Apache-2.0
"""tq_mq_soa.py — multi-query (spec-verify) TurboQuant decode attention for
the Opt#3 SoA cache layout (P6.19, vllm_gaudi/xpu/patches/).

PURPOSE: single-pass verify attention. The split-row dispatch in
tq_spec_fix (q_len separate B=1 decode calls, leg t5-proven clean under
XPU graphs) rescans the compressed context q_len times per layer per
verify step — measured cost on 2x Arc Pro B70 / qwen3.8-27b-fp8 / MTP
k3: k8v4 solo 54.5 tps / agg8 240.9 tps vs the fp8 posture's 84.5 /
299.7 (leg t6p). This kernel scores ALL verify rows against each shared
KV tile in one pass (flash-decoding style), restoring single-scan
bandwidth — the design of llm-scaler v19's
``triton_turboquant_mq_decode_attention`` (the donor's production verify
path, tuned on this exact hardware), adapted to this wheel's Opt#3 SoA
layout: the donor kernel targets the pre-Opt#3 slot-interleaved layout
(per-token inline scale/zero), so addressing and metadata loads below
follow the stock ``_tq_decode_stage1`` byte-for-byte (lines 168-300 of
``triton_turboquant_decode``), while the multi-row structure — kernel-
side causal ramp off a device-read q0, uniform splits, per-row online
softmax with -inf guards, blind stage-2 combine — follows the donor.

GRAPH SAFETY (the reason this is a kernel, per the P6.19 root cause):
every replay-varying quantity is read device-side from live buffers —
q0 (row-0 causal limit) from ``Q0_lens_ptr``, per-row limits as
``q0 + row``, split ranges as ``cdiv(q0 + Q_LEN - 1, splits)`` — so one
captured launch is correct for ANY context length at replay. Grid is
fixed (1, Hq, splits); stage 1 always writes its (possibly empty)
partial so stage 2 can combine blindly; no host-side read of a device
value anywhere on the replay path. q-rotation GEMM and scratch are the
stock decode wrapper's own per-call allocations — the exact allocation
pattern the t5 leg already proved replay-clean (every split-row decode
call runs it inside the captured verify graph).

Semantics == the split dispatch: row r attends [0 .. seq_len-q_len+r]
inclusive over the same SoA bytes the stock decode kernel reads.

Value path supports VQB in {3, 4} (every admitted turboquant preset);
key path supports MSE (any bit width) and fp8-e4nv (XPU always, per the
stock device probe).

Provenance: kernel structure llm-scaler v19 ``_tq_mq_decode_stage1`` /
``_tq_mq_fwd_stage2`` (READ-ONLY donor, line C; legacy/tq-drafter-mq-
verify-v19 + spec-kv-dtype-vram-v21); SoA addressing/dequant chain from
the stock 0.30 wheel's ``triton_turboquant_decode``; integration by
vllm-arc P6.19 (legs t4f/t5/t6p; MQ validation + t8 A/B,
results/2026-11-04-p619-tqverify/).
"""

import math
import os

import torch
import triton
import triton.language as tl

try:  # stock parity: XPU is never cuda_alike -> 0 (e4nv) everywhere here
    from vllm.platforms import current_platform as _current_platform
except Exception:  # noqa: BLE001 — standalone/injector import order
    _current_platform = None

_FP8_E4B15: dict[int, int] = {}

_MQ_SPLITS = max(1, int(os.environ.get("VLLM_ARC_TQ_MQ_SPLITS", "32") or 32))


def _use_fp8_e4b15(device: int = 0) -> int:
    """Stock ``triton_turboquant_decode._use_fp8_e4b15`` verbatim: e4b15
    only on pre-Hopper CUDA; XPU (this platform) always uses e4nv."""
    if device not in _FP8_E4B15:
        if _current_platform is not None and _current_platform.is_cuda_alike():
            cap = torch.cuda.get_device_capability(device)
            _FP8_E4B15[device] = 1 if cap < (8, 9) else 0
        else:
            _FP8_E4B15[device] = 0
    return _FP8_E4B15[device]


@triton.jit
def _tq_mq_soa_stage1(
    Q_rot_ptr,  # [Q_LEN, Hq, D] fp32 (MSE) or query dtype (fp8) — padded rows masked
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    KV_cache_u16_ptr,  # uint16 view of same storage — SoA metadata loads
    Block_table_ptr,  # [1, max_num_blocks] int32
    Q0_lens_ptr,  # [1] int32 — row-0 causal limit (cached_len + 1), LIVE
    Centroids_ptr,  # [n_centroids] float32
    Mid_o_ptr,  # [Hq, NUM_KV_SPLITS, Q_BLOCK, D+1] float32
    # Strides
    stride_qb,
    stride_qh,
    stride_cache_block: tl.int64,
    stride_mid_h,
    stride_mid_s,
    # Constexpr dims
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    Q_LEN: tl.constexpr,  # actual verify rows (1 < Q_LEN <= Q_BLOCK)
    Q_BLOCK: tl.constexpr,  # padded power-of-2 row tile
    # TQ layout constants (stock _tq_decode_stage1 parity)
    MSE_BITS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,  # ceil(D * VQB / 8)
    KEY_DATA_BYTES: tl.constexpr,  # MSE_BYTES or HEAD_DIM (fp8)
    META_REGION_OFFSET: tl.constexpr,
    NUM_SOA_FIELDS: tl.constexpr,  # 3 (MSE) / 2 (fp8)
    SOA_K_NORM: tl.constexpr,  # 0; unused for fp8
    SOA_V_SCALE: tl.constexpr,  # 1 (MSE) / 0 (fp8)
    SOA_V_ZERO: tl.constexpr,  # 2 (MSE) / 1 (fp8)
    # Score constants
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    KEY_FP8: tl.constexpr,
    FP8_E4B15: tl.constexpr = 0,  # XPU: always 0 (e4nv)
):
    hid = tl.program_id(1)  # q_head
    sid = tl.program_id(2)  # kv_split

    kv_head = hid // KV_GROUP_SIZE

    q_offs = tl.arange(0, Q_BLOCK)
    q_mask = q_offs < Q_LEN

    # Row-0 causal limit from the LIVE buffer; per-row limits ramp from
    # it: row r attends positions < q0 + r (== split-mode limit
    # seq_lens[i] - (q_len - 1 - r), inclusive bound).
    q0 = tl.load(Q0_lens_ptr)
    row_limit = q0 + q_offs
    seq_len_max = q0 + (Q_LEN - 1)  # union of all rows' extents

    split_len = tl.cdiv(seq_len_max, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len_max)

    # NOTE: no early return — every program writes its (possibly
    # all-empty) partial so stage 2 can combine blindly (empty partial:
    # lse = -inf, acc = 0 -> zero weight).

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    q_base = q_offs[:, None] * stride_qb + hid * stride_qh
    q_rot = tl.load(
        Q_rot_ptr + q_base + d_offs[None, :],
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    m_prev = tl.full([Q_BLOCK], -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros([Q_BLOCK], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCK, BLOCK_D], dtype=tl.float32)

    # TILED LOOP: BLOCK_KV tokens per iteration; all query rows share the
    # same KV tile loads (one pass over the compressed context). Bounds
    # are device scalars — the loop is replay-correct by construction.
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end
        # Valid lane = within the split (kv_mask — must match the LOAD
        # masks: a lane masked out of the K/V loads reads zeros and would
        # otherwise carry a PHANTOM finite score q.0 == 0 into the softmax)
        # AND within the row's causal limit (the multi-row generalization
        # of the stock kv_mask): row r sees positions < row_limit[r].
        tile_mask = (kv_offs[None, :] < row_limit[:, None]) & kv_mask[None, :]  # [Q_BLOCK, BLOCK_KV]

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        # Opt#3 SoA addressing — stock _tq_decode_stage1 lines 168-188.
        slot_within_block = page_off.to(tl.int64)
        block_base = block_nums * stride_cache_block
        DATA_BYTES_PER_SLOT: tl.constexpr = KEY_DATA_BYTES + VAL_DATA_BYTES
        data_bases = (
            block_base
            + slot_within_block * (NUM_KV_HEADS * DATA_BYTES_PER_SLOT)
            + tl.cast(kv_head, tl.int64) * DATA_BYTES_PER_SLOT
        )
        head_meta_u16_base = (block_base + META_REGION_OFFSET) // 2 + tl.cast(
            kv_head, tl.int64
        ) * (NUM_SOA_FIELDS * BLOCK_SIZE)
        knorm_u16_addrs = (
            head_meta_u16_base + SOA_K_NORM * BLOCK_SIZE + slot_within_block
        )
        vscale_u16_addrs = (
            head_meta_u16_base + SOA_V_SCALE * BLOCK_SIZE + slot_within_block
        )
        vzero_u16_addrs = (
            head_meta_u16_base + SOA_V_ZERO * BLOCK_SIZE + slot_within_block
        )

        # ============================================================
        # SCORES: [Q_BLOCK, BLOCK_KV]
        # ============================================================
        if KEY_FP8:
            k_addrs = data_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scores = (
                tl.sum(
                    tl.where(
                        d_mask[None, None, :],
                        q_rot[:, None, :] * k_float[None, :, :],
                        0.0,
                    ),
                    axis=2,
                )
                * ATTN_SCALE
            )
            scores = tl.where(tile_mask, scores, -float("inf"))
        else:
            # MSE unpack + norms (stock lines 213-251, row-generalized)
            mse_addrs0 = data_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            # Centroid gather + per-row dot product
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            # Norm correction is pre-folded into the stored per-token
            # scalar by the MSE store kernel (stock Opt#1) — no per-tile
            # correction here.
            term1 = tl.sum(
                tl.where(
                    d_mask[None, None, :],
                    q_rot[:, None, :] * c_vals[None, :, :],
                    0.0,
                ),
                axis=2,
            )

            # Opt#3: K-norms live in the per-block SoA region (single u16
            # load per token, contiguous within a block).
            norm_u16 = tl.load(
                KV_cache_u16_ptr + knorm_u16_addrs, mask=kv_mask, other=0
            )
            vec_norms = norm_u16.to(tl.float16, bitcast=True).to(tl.float32)

            scores = vec_norms[None, :] * term1 * ATTN_SCALE
            scores = tl.where(tile_mask, scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (per-row; a row with no valid token in
        # this tile keeps m = -inf -> guarded exponent, zero weight)
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, axis=1), m_prev)  # [Q_BLOCK]
        n_safe = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
        re_scale = tl.exp(m_prev - n_safe)
        p = tl.exp(scores - n_safe[:, None])

        # ============================================================
        # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D] (shared by rows;
        # stock lines 269-300 verbatim)
        # ============================================================
        val_bases = data_bases + KEY_DATA_BYTES

        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16v = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16v >> val_bit_shift[None, :]) & 0x7).to(tl.float32)
        else:  # VQB == 4
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

        scale_u16 = tl.load(KV_cache_u16_ptr + vscale_u16_addrs, mask=kv_mask, other=0)
        zero_u16 = tl.load(KV_cache_u16_ptr + vzero_u16_addrs, mask=kv_mask, other=0)
        v_scales = scale_u16.to(tl.float16, bitcast=True).to(tl.float32)
        v_zeros = zero_u16.to(tl.float16, bitcast=True).to(tl.float32)
        values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ============================================================
        # WEIGHTED VALUE ACCUMULATION (per-row)
        # ============================================================
        acc = acc * re_scale[:, None] + tl.sum(
            p[:, :, None] * values[None, :, :], axis=1
        )
        l_prev = l_prev * re_scale + tl.sum(p, axis=1)
        m_prev = n_e_max

    # Store partial (stock lines 309-314, row-generalized): padded rows
    # write benign finite values; stage 2 stores only rows < Q_LEN.
    out_base = hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(
        Mid_o_ptr + out_base + q_offs[:, None] * (HEAD_DIM + 1) + d_offs[None, :],
        acc / safe_l[:, None],
        mask=d_mask[None, :],
    )
    lse = m_prev + tl.log(safe_l)  # -inf + log(1) = -inf for empty rows
    tl.store(Mid_o_ptr + out_base + q_offs * (HEAD_DIM + 1) + HEAD_DIM, lse)


@triton.jit
def _tq_mq_soa_stage2(
    Mid_o_ptr,  # [Hq, NUM_KV_SPLITS, Q_BLOCK, D+1] float32
    Out_ptr,  # [Q_BLOCK, Hq, D] query dtype
    stride_mid_h,
    stride_mid_s,
    stride_obs,
    stride_oh,
    Q_LEN: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    hid = tl.program_id(1)

    q_offs = tl.arange(0, Q_BLOCK)
    q_mask = q_offs < Q_LEN
    d_offs = tl.arange(0, BLOCK_DV)
    d_mask = d_offs < Lv

    row_stride = Lv + 1
    base = hid * stride_mid_h

    m_run = tl.full([Q_BLOCK], -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros([Q_BLOCK], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCK, BLOCK_DV], dtype=tl.float32)

    # Blind combine (donor _tq_mq_fwd_stage2): empty partials carry
    # lse = -inf -> zero weight, no special casing.
    for sid in range(NUM_KV_SPLITS):
        sbase = base + sid * stride_mid_s
        tlogic = tl.load(Mid_o_ptr + sbase + q_offs * row_stride + Lv)
        tv = tl.load(
            Mid_o_ptr + sbase + q_offs[:, None] * row_stride + d_offs[None, :],
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        n_e_max = tl.maximum(m_run, tlogic)
        n_safe = tl.where(n_e_max == -float("inf"), 0.0, n_e_max)
        old_scale = tl.exp(m_run - n_safe)
        exp_logic = tl.where(tlogic == -float("inf"), 0.0, tl.exp(tlogic - n_safe))
        acc = acc * old_scale[:, None] + exp_logic[:, None] * tv
        e_sum = e_sum * old_scale + exp_logic
        m_run = n_e_max

    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    result = acc / safe_sum[:, None]
    tl.store(
        Out_ptr + q_offs[:, None] * stride_obs + hid * stride_oh + d_offs[None, :],
        result,
        mask=q_mask[:, None] & d_mask[None, :],
    )


def triton_turboquant_mq_decode_attention_soa(
    query: torch.Tensor,  # [Q_LEN, Hq, D] — one request's verify rows
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [1, max_num_blocks] int32
    q0_seq_lens: torch.Tensor,  # [1] int32 — row-0 limit (cached_len + 1)
    Pi: torch.Tensor,  # [D, D] float32
    centroids: torch.Tensor,  # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,  # store-side padding; unused by decode addressing
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,  # pre-folded in stored norms (Opt#1)
    PiT: torch.Tensor | None = None,  # [D, D] pre-computed Pi.T contiguous
    max_num_kv_splits: int = 0,
) -> torch.Tensor:
    """Multi-query TQ decode attention over the Opt#3 SoA layout.

    One pass over the compressed KV per (head, split): all Q_LEN verify
    rows are scored against each shared KV tile. ``q0_seq_lens`` is a
    LIVE device tensor; every replay-varying quantity is derived from it
    inside the kernels. Returns [Q_LEN, Hq, D] in query's dtype.
    """
    Q_LEN, Hq, D = query.shape
    assert 1 < Q_LEN <= 8, "MQ verify kernel is sized for 1 < q_len <= 8"
    assert value_quant_bits in (3, 4), (
        "MQ V dequant covers the admitted presets (VQB 3/4); fp8-V has no "
        "admitted preset"
    )
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    # Opt#3 SoA layout constants — stock decode wrapper lines 519-527
    # verbatim (packed key bytes; KPS is store-side padding only).
    key_data_bytes = D if key_fp8 else math.ceil(D * mse_bits / 8)
    val_data_bytes = math.ceil(D * value_quant_bits / 8)
    meta_region_offset = block_size * Hk * (key_data_bytes + val_data_bytes)
    num_soa_fields = 2 if key_fp8 else 3
    soa_k_norm = 0
    soa_v_scale = 0 if key_fp8 else 1
    soa_v_zero = 1 if key_fp8 else 2
    kv_cache_u16 = kv_cache.view(torch.uint16)

    # q_rot = q @ Pi.T — stock wrapper lines 529-538 verbatim (per-call
    # allocations; replay-proven by the t5 split legs that run this same
    # wrapper inside the captured verify graph).
    if key_fp8:
        q_rot = query.contiguous()
    else:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        q_rot = (q_float @ PiT).contiguous()

    Q_BLOCK = triton.next_power_of_2(max(Q_LEN, 2))
    NUM_KV_SPLITS = max_num_kv_splits if max_num_kv_splits > 0 else _MQ_SPLITS

    mid_o = torch.empty(
        Hq, NUM_KV_SPLITS, Q_BLOCK, D + 1, dtype=torch.float32, device=device
    )
    output = torch.empty(Q_BLOCK, Hq, D, dtype=query.dtype, device=device)

    grid = (1, Hq, NUM_KV_SPLITS)
    _tq_mq_soa_stage1[grid](
        q_rot,
        kv_cache,
        kv_cache_u16,
        block_table,
        q0_seq_lens,
        centroids,
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        Q_LEN=Q_LEN,
        Q_BLOCK=Q_BLOCK,
        MSE_BITS=mse_bits,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        KEY_DATA_BYTES=key_data_bytes,
        META_REGION_OFFSET=meta_region_offset,
        NUM_SOA_FIELDS=num_soa_fields,
        SOA_K_NORM=soa_k_norm,
        SOA_V_SCALE=soa_v_scale,
        SOA_V_ZERO=soa_v_zero,
        ATTN_SCALE=scale,
        BLOCK_D=triton.next_power_of_2(D),
        BLOCK_KV=4,  # stock decode wrapper constant
        KEY_FP8=1 if key_fp8 else 0,
        FP8_E4B15=_use_fp8_e4b15(device.index or 0),
        num_warps=1,  # donor v21 B70-tuned (BLOCK_KV=4, 1 warp, 1 stage)
        num_stages=1,
    )

    grid2 = (1, Hq)
    _tq_mq_soa_stage2[grid2](
        mid_o,
        output,
        mid_o.stride(0),
        mid_o.stride(1),
        output.stride(0),
        output.stride(1),
        Q_LEN=Q_LEN,
        Q_BLOCK=Q_BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=triton.next_power_of_2(D),
        Lv=D,
        num_warps=4,
        num_stages=2,
    )

    return output[:Q_LEN]
