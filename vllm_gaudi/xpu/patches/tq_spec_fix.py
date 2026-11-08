# SPDX-License-Identifier: Apache-2.0
"""tq_spec_fix.py — spec-verify XPU-graph correctness for TurboQuant KV
dtypes (P6.19, 2026-11-04).

Without this patch, any ``turboquant_*`` kv-cache dtype corrupts under
MTP speculative decoding whenever XPU graphs are on (reported
2026-11-03, P6.18 campaign; reopened as P6.19): deterministic repetition
attractors / "blank message" generations, dtype-wide, while the same
posture with spec off or graphs off serves clean.

ROOT CAUSE (rig-proven, campaign P6.19 — #TQ-blind-verify):
  MTP verify steps arrive at ``TurboQuantAttentionImpl`` as
  ``is_prefill=True`` q_len=k+1 extend batches, and those shapes are
  XPU-graph captured (solo-verify host syncs crash inside command-graph
  replay — the t2i3 canary fingerprint). The stock wheel's
  ``build_for_cudagraph_capture`` leaves capture dummies at
  ``seq_len == q_len``, so ``_prefill_attention``'s raw-KV flash fast
  path (``max_query_len == max_seq_len``) is what gets baked into the
  verify graphs: every replay attends ONLY the in-batch draft tokens and
  never reads the cached conversation -> garbage acceptance from the
  first token. Donor llm-scaler v19b hit the identical defect
  ("near-zero acceptance + blank-message hallucinations").

SECOND DEFECT (found while fixing the first, leg t4f):
  Capture forcing alone is not enough. The stock continuation branch
  dispatches the verify rows as ONE multi-row synthetic-decode call
  (stride-0 expanded block table + synthesized per-row seq_lens tensor +
  B=q_len kernel grid). That dispatch is numerically identical in eager
  mode (t3e clean) but computes wrong under graph REPLAY ONLY — residual
  deterministic repetition attractors (t4f: stable per-class hashes,
  token-cap ramming). Isolating the construction (leg t5): q_len
  SEPARATE single-row decode calls — the byte-for-byte W1-proven-clean
  call shape (B=1, normal-stride 1-row block table, per-row causal limit
  as a recorded device op off the live seq_lens buffer) — serves CLEAN
  under graphs: zero garble on all four probe classes, t0-stable
  essay/code/simple, flip probe byte-identical to eager.

FIX (both defects, this file):
  1. ``build_for_cudagraph_capture``: multi-token (spec-verify) captures
     force the continuation branch — ``max_seq_len=q_len+1``,
     ``seq_lens_cpu=q_len+1``, ``seq_lens=q_len`` (donor v19b idiom) —
     so the raw-KV flash fast path is never baked into verify graphs;
     capture-time kernel work stays trivial.
  2. ``_prefill_attention`` continuation, 1 < q_len <= 8: per-row
     split dispatch — for each verify row r a single-row decode call
     with limit ``seq_lens[i]-(q_len-1-r)`` (== stock synth values
     ``cached+1 .. cached+q``), all values recorded ops off the LIVE
     seq_lens buffer (replay-refreshed).
  3. Optional single-pass mode (``VLLM_ARC_TQ_VERIFY_MODE=mq``): the
     multi-query kernel ``patches/tq_mq_aos.py`` scores all verify rows
     against each shared KV tile in one pass (donor llm-scaler v19 MQ
     structure on this wheel's slot-interleaved layout — the engine's
     default ``ops.triton_turboquant_{store,decode}`` pair; the first
     draft targeted the FlyDSL-gated SoA tree and read the live cache
     as zeros — legs t8/t8e/t8f, MQVAL3c) — the split dispatch rescans
     the compressed context q_len times per layer
     (measured k8v4 cost, leg t6p: solo 54.5 tps / agg8 240.9 tps vs
     the fp8 posture's 84.5 / 299.7); MQ restores single-scan
     bandwidth. Offline gate (P619D MQVAL5/6): matches the split
     dispatch on all four presets, all shapes, AND reproduces the t8f
     engine dump (results/2026-11-04-p619-tqverify/). q0 comes from the
     same recorded device op as the split limits; every replay-varying
     quantity is read device-side (fixed grid, kernel-side split
     ranges), so one captured launch serves any context length.

Knobs (A/B, evidenced by legs t4f/t5/t6p + MQVAL5/6/t8*):
  ``VLLM_ARC_TQ_SPEC_FIX``       1 (default) active, 0 passthrough.
  ``VLLM_ARC_TQ_VERIFY_MODE``    split (default, t5-proven under
                                 graphs) | mq (single-pass kernel,
                                 MQVAL5/6-proven numerics on the AoS
                                 layout; serve-leg evidence t8g/t8h)
                                 | synth (pre-t5 multi-row dispatch,
                                 kept for differential diagnosis only).

Eager mode computes identical limit values either way — non-graph
serving is unaffected.

Provenance: capture idiom llm-scaler v19b; per-row limit construction
mirrors the donor's synthetic-decode fallback (v21 lines 1394-1404);
MQ kernel structure llm-scaler v19 with this wheel's AoS
``ops.triton_turboquant_decode`` addressing (the engine's actual
store/decode pair); integration + split dispatch by vllm-arc P6.19
(telemetry t1i/t2i3, wedges W1/t3e, fix legs t4f-t8f, MQVAL/MQVAL2/
MQVAL3c/MQVAL5/6; results/2026-11-04-p619-tqverify/).
"""

import logging
import os

import torch
import torch.nn.functional as F

try:  # vllm-configured channel (visible in serve logs); fallback standalone
    from vllm.logger import init_logger as _init_logger
    logger = _init_logger(__name__)
except Exception:  # noqa: BLE001 — injector legs outside vllm import order
    logger = logging.getLogger(__name__)

_TQ_SPEC_FIX = os.environ.get("VLLM_ARC_TQ_SPEC_FIX", "1") != "0"
_VERIFY_MODE = os.environ.get("VLLM_ARC_TQ_VERIFY_MODE", "split")
_SPLIT_MAX_Q = 8  # verify q_len <= k+1; donor _TQ_MQ_MAX_Q parity

# Single-pass MQ kernel (patches/tq_mq_aos.py). Absent module (e.g. an
# injector leg that mounts only this file) falls back to split loudly —
# never break serving over a perf mode.
try:
    from vllm_gaudi.xpu.patches.tq_mq_aos import (
        triton_turboquant_mq_decode_attention as _mq_decode,
    )
    _HAS_MQ = True
except Exception as _e:  # noqa: BLE001 — visibility law
    _mq_decode = None
    _HAS_MQ = False
    logger.warning("P619 tq_mq_aos unavailable (%r); mode=mq falls back to split", _e)

_VERIFY_MODE = (
    _VERIFY_MODE if (_VERIFY_MODE != "mq" or _HAS_MQ) else "split"
)

# Instrument state: shape-class -> logged (bounded; distinct classes few).
_SEEN_SHAPES: dict[tuple, bool] = {}
_LOUD_CLASSES = 12


def _log_shape(where: str, md) -> None:
    """One-time per-shape-class instrument line (visibility law)."""
    try:
        qsl_cpu = getattr(md, "query_start_loc_cpu", None)
        diffs = tuple(qsl_cpu.tolist()[1:]) if (
            qsl_cpu is not None and qsl_cpu.numel() > 1) else ()
        key = (where, bool(getattr(md, "is_prefill", False)),
               int(getattr(md, "num_actual_tokens", -1)), diffs)
        if _SEEN_SHAPES.get(key):
            return
        _SEEN_SHAPES[key] = True
        if len(_SEEN_SHAPES) > _LOUD_CLASSES:
            return
        sl_cpu = getattr(md, "seq_lens_cpu", None)
        seqs = sl_cpu.tolist()[:8] if (sl_cpu is not None and sl_cpu.numel()) else None
        logger.warning(
            "P619I %s: is_prefill=%s n_tok=%s qlens=%s seqs=%s ndec=%s "
            "mql=%s msl=%s",
            where,
            getattr(md, "is_prefill", None),
            getattr(md, "num_actual_tokens", None),
            diffs[:8], seqs,
            getattr(md, "num_decodes", None),
            getattr(md, "max_query_len", None),
            getattr(md, "max_seq_len", None),
        )
    except Exception:  # noqa: BLE001 — instrument must never break serving
        pass


def _install(mod) -> None:
    Builder = getattr(mod, "TurboQuantMetadataBuilder", None)
    Impl = getattr(mod, "TurboQuantAttentionImpl", None)
    if Builder is None or Impl is None:
        return
    if getattr(Impl.forward, "_arc_p619_tq_spec_fix", False):
        return

    _HAS_FLASH = bool(getattr(mod, "_HAS_FLASH_ATTN", False))
    _THRESHOLD = int(getattr(mod, "_CONTINUATION_DECODE_THRESHOLD", 128))
    _aosa_decode = getattr(mod, "triton_turboquant_decode_attention", None)
    orig_build_capture = Builder.build_for_cudagraph_capture
    orig_fwd = Impl.forward

    # ------------------------------------------------------------------ #
    # 1. capture-time branch forcing (donor v19b idiom)                   #
    # ------------------------------------------------------------------ #
    def _build_for_cudagraph_capture(self, common_attn_metadata):
        md = orig_build_capture(self, common_attn_metadata)
        # Multi-token (spec-verify) captures must NOT bake the raw-KV flash
        # fast path: capture dummies sit at seq_len == q_len, so
        # max_query_len == max_seq_len would record flash over ONLY the
        # in-batch verify tokens and replays would never read the KV cache
        # (#TQ-blind-verify). Force the continuation branch (its kernels
        # read the dynamic seq_lens/block_table buffers refreshed at
        # replay); keep capture-time kernel work trivial.
        if _TQ_SPEC_FIX and md.max_query_len > 1:
            md.max_seq_len = md.max_query_len + 1
            if md.seq_lens_cpu is not None:
                md.seq_lens_cpu.fill_(md.max_query_len + 1)
            md.seq_lens.fill_(md.max_query_len)
        return md

    # ------------------------------------------------------------------ #
    # 2. continuation branch: split-row dispatch (the fix) / v3 synth     #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query,   # (N, Hq, D)
        key,     # (N, Hk, D)
        value,   # (N, Hk, D)
        kv_cache,
        attn_metadata,
        Pi,
        centroids,
        PiT=None,
        layer=None,
    ):
        N, Hq, D = query.shape

        # Fast path: flash_attn for first-chunk prefills (all K/V in batch).
        if _HAS_FLASH and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        # Continuation or no flash_attn: per-request attention.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
        _max_seq = attn_metadata.max_seq_len
        _ac = getattr(self, "_arange_cache", None)
        if _ac is None or _ac.shape[0] <= _max_seq:
            _ac = torch.arange(
                0, _max_seq + 1, device=query.device,
                dtype=attn_metadata.seq_lens.dtype,
            )
            self._arange_cache = _ac
        _arange_cache = _ac

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if _HAS_FLASH:
                    self._cu_2[1:2] = q_len
                    cu = self._cu_2
                    out = self._flash_attn_varlen(
                        q=q_seq, k=k_seq, v=v_seq,
                        cu_seqlens_q=cu, cu_seqlens_k=cu,
                        max_seqlen_q=q_len, max_seqlen_k=q_len,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t, k_t, v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: tokens already stored to the TQ cache
                # by do_kv_cache_update. Use the decode kernel directly.
                cached_len = seq_len - q_len
                if q_len <= _THRESHOLD:
                    if _VERIFY_MODE == "mq" and 1 < q_len <= _SPLIT_MAX_Q:
                        # Single-pass MQ dispatch: all q_len verify rows in
                        # one two-stage kernel pass (one context scan).
                        # Row-0 limit q0 = seq-(q-1) as a recorded device op
                        # off the LIVE seq_lens buffer; the kernel derives
                        # per-row causal limits and split ranges from it
                        # device-side (graph-replay-safe by construction).
                        # Offline MQVAL5/6: matches the split rows and
                        # reproduces the t8f engine dump (AoS layout).
                        out = _mq_decode(
                            query=q_seq,
                            kv_cache=kv_cache,
                            block_table=attn_metadata.block_table[i : i + 1],
                            q0_seq_lens=(
                                attn_metadata.seq_lens[i : i + 1]
                                - (q_len - 1)
                            ),
                            Pi=Pi,
                            centroids=centroids,
                            scale=self.scale,
                            mse_bits=self.tq_config.key_mse_bits,
                            key_packed_size=self.tq_config.key_packed_size,
                            value_quant_bits=(
                                self.tq_config.effective_value_quant_bits
                            ),
                            key_fp8=self.tq_config.key_fp8,
                            norm_correction=self.tq_config.norm_correction,
                            PiT=PiT,
                        )
                        output[q_start:q_end] = out.to(query.dtype)
                    elif _VERIFY_MODE == "split" and 1 < q_len <= _SPLIT_MAX_Q:
                        # Split-row dispatch: q_len SEPARATE single-row
                        # decode calls — the W1-proven-correct call shape
                        # (B=1, normal-stride 1-row block table). Per-row
                        # causal limit as a recorded 0-d device op off the
                        # LIVE seq_lens buffer (replay-refreshed): row r
                        # attends [0 .. seq-q+r] inclusive == stock synth
                        # values [cached+1 .. cached+q]. The t4f leg proved
                        # the multi-row synthetic-decode dispatch (stride-0
                        # expand + synth tensor + B=q_len grid) is replay-
                        # broken; t5 proved this split form clean.
                        bt_row = attn_metadata.block_table[i : i + 1]
                        for r in range(q_len):
                            sl_r = (
                                attn_metadata.seq_lens[i : i + 1]
                                - (q_len - 1 - r)
                            )
                            out_r = _aosa_decode(
                                query=q_seq[r : r + 1],
                                kv_cache=kv_cache,
                                block_table=bt_row,
                                seq_lens=sl_r,
                                Pi=Pi,
                                centroids=centroids,
                                scale=self.scale,
                                mse_bits=self.tq_config.key_mse_bits,
                                key_packed_size=self.tq_config.key_packed_size,
                                value_quant_bits=(
                                    self.tq_config.effective_value_quant_bits
                                ),
                                key_fp8=self.tq_config.key_fp8,
                                norm_correction=self.tq_config.norm_correction,
                                PiT=PiT,
                            )
                            output[q_start + r : q_start + r + 1] = (
                                out_r.to(query.dtype)
                            )
                    else:
                        # Pre-t5 multi-row synthetic decode (A/B only; leg
                        # t4f proved it replay-broken — kept for diagnosis).
                        offs = _arange_cache[1 : q_len + 1]
                        synth_seq_lens = (
                            attn_metadata.seq_lens[i] - q_len
                        ) + offs
                        synth_bt = attn_metadata.block_table[i : i + 1].expand(
                            q_len, -1)
                        extent_hint = (
                            int(synth_bt.shape[1]) * int(kv_cache.shape[1])
                        )
                        if self._soa_store:
                            out = self._dispatch_decode_soa(
                                query=q_seq,
                                kv_cache=kv_cache,
                                block_table=synth_bt,
                                seq_lens=synth_seq_lens,
                                Pi=Pi,
                                centroids=centroids,
                                scale=self.scale,
                                mse_bits=self.tq_config.key_mse_bits,
                                key_packed_size=self.tq_config.key_packed_size,
                                value_quant_bits=(
                                    self.tq_config.effective_value_quant_bits
                                ),
                                value_packed_size=(
                                    self.tq_config.value_packed_size
                                ),
                                max_seq_len=extent_hint,
                                key_fp8=self.tq_config.key_fp8,
                                norm_correction=self.tq_config.norm_correction,
                                PiT=PiT,
                                sinks=self.sinks,
                                sliding_window=self.sliding_window,
                            )
                        else:
                            out = _aosa_decode(
                                query=q_seq,
                                kv_cache=kv_cache,
                                block_table=synth_bt,
                                seq_lens=synth_seq_lens,
                                Pi=Pi,
                                centroids=centroids,
                                scale=self.scale,
                                mse_bits=self.tq_config.key_mse_bits,
                                key_packed_size=self.tq_config.key_packed_size,
                                value_quant_bits=(
                                    self.tq_config.effective_value_quant_bits
                                ),
                                key_fp8=self.tq_config.key_fp8,
                                norm_correction=self.tq_config.norm_correction,
                                PiT=PiT,
                            )
                        output[q_start:q_end] = out.to(query.dtype)
                else:
                    # Large continuation: dequant cached K/V + flash_attn
                    # (never spec-verify-shaped; eager chunked-prefill only
                    # — host ints are correct there).
                    out = self._continuation_prefill(
                        layer,
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                    )
                    output[q_start:q_end] = out.to(query.dtype)

        return output

    def _forward(self, layer, query, key, value, kv_cache, attn_metadata,
                 output=None, output_scale=None, output_block_scale=None):
        try:
            if attn_metadata is not None:
                _log_shape("forward", attn_metadata)
        except Exception:  # noqa: BLE001
            pass
        return orig_fwd(self, layer, query, key, value, kv_cache,
                        attn_metadata, output=output,
                        output_scale=output_scale,
                        output_block_scale=output_block_scale)

    _forward._arc_p619_tq_spec_fix = True
    Builder.build_for_cudagraph_capture = _build_for_cudagraph_capture
    Impl._prefill_attention = _prefill_attention
    Impl.forward = _forward
    logger.warning(
        "P619 tq_spec_fix v6 installed (active=%s mode=%s mq_kernel=%s): "
        "v19b verify-capture forcing + %s continuation dispatch",
        _TQ_SPEC_FIX, _VERIFY_MODE, _HAS_MQ,
        {"mq": "single-pass MQ", "split": "split-row"}.get(
            _VERIFY_MODE, "v3-synth"),
    )
