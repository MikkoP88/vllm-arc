# SPDX-License-Identifier: Apache-2.0
"""tq_spec_fix.py v4 — P6.19 spec-verify XPU-graph correctness fix for the
TurboQuant backend (injector leg; repo port target vllm_gaudi/xpu/patches/).

ROOT CAUSE (rig-proven 2026-11-04, campaign P6.19, #TQ-blind-verify):
  XPU-graph capture of the MTP spec-verify forward bakes the raw-KV flash
  fast path into the verify graphs (capture dummies sit at seq_len ==
  q_len). Replays attend ONLY the in-batch draft tokens — the target never
  reads the cached conversation -> garbage acceptance ("blank message"
  class). Donor llm-scaler v19b hit the identical defect.

FIX HISTORY (this campaign):
  v3: capture forcing (donor idiom: max_seq_len=q+1, seq_lens_cpu=q+1,
      seq_lens=q) + device-dynamic synth limits ((seq_lens[i]-q_len) +
      arange[1..q_len]) + block-table-width extent hint.
      RESULT (leg t4f): PARTIAL — verify now reads the cache (real content,
      no blank-message class) but outputs remain WRONG AT REPLAY ONLY:
      deterministic REP-LOOP attractors on code/multi/simple, local count
      errors on the flip probe. Eager (t3e) runs the identical continuation
      code CLEAN, and instrument logs confirm the forcing applied at
      capture (mql=4 msl=5 seqs=[5]). So the residual defect is specific
      to the MULTI-ROW SYNTHETIC-DECODE dispatch under graph replay
      (stride-0 expanded block table + synthesized per-row seq_lens
      tensor + multi-row grid), not to the captured branch choice.

  v4 (this file): differential experiment + ship-path candidate.
      VLLM_ARC_TQ_VERIFY_MODE:
        "split" (default) — continuation for 1 < q_len <= 8 issues
            q_len SEPARATE single-row decode calls (B=1, normal-stride
            1-row block table, per-row limit as a recorded 0-d device
            op off the live seq_lens buffer). This is byte-for-byte the
            W1-proven-correct decode call shape (spec-off + graphs on =
            clean). If this is clean under graphs, the multi-row synth
            construction is proven to be the replay defect and the donor
            MQ kernel (kernel-side causal ramp) is the performant port.
        "synth" — v3 behavior (multi-row synth dispatch) for A/B.
      Capture forcing retained unchanged (donor v19b idiom).

Perf note: split mode runs q_lenx kernel launches per verify per layer —
a diagnostic posture; the donor MQ kernel (triton_turboquant_mq_decode_
attention, tuned on 2x Arc Pro B70 / qwen3.8-27b-fp8) is the single-pass
production form of the same idea and is the intended ship path.

Knob: VLLM_ARC_TQ_SPEC_FIX (1 default; 0 = passthrough).

Provenance: llm-scaler v19b/v19/v21 (READ-ONLY donor, line C); telemetry
legs t1i/t2i3, wedge legs W1/t3e, fix legs t4f (v3 partial); this v4 leg.
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
_MQ_MAX_Q = 8  # donor _TQ_MQ_MAX_Q; verify q_len <= k+1

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
    # 1. capture-time branch forcing (donor v19b idiom, unchanged)        #
    # ------------------------------------------------------------------ #
    def _build_for_cudagraph_capture(self, common_attn_metadata):
        md = orig_build_capture(self, common_attn_metadata)
        # Multi-token (spec-verify) captures must NOT bake the raw-KV flash
        # fast path (#TQ-blind-verify). Force the continuation branch; keep
        # capture-time kernel work trivial.
        if _TQ_SPEC_FIX and md.max_query_len > 1:
            md.max_seq_len = md.max_query_len + 1
            if md.seq_lens_cpu is not None:
                md.seq_lens_cpu.fill_(md.max_query_len + 1)
            md.seq_lens.fill_(md.max_query_len)
        return md

    # ------------------------------------------------------------------ #
    # 2. continuation branch: per-row split dispatch (v4) or v3 synth     #
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
                    if _VERIFY_MODE == "split" and 1 < q_len <= _MQ_MAX_Q:
                        # P6.19 v4: q_len SEPARATE single-row decode calls —
                        # the W1-proven-correct call shape (B=1, normal-
                        # stride 1-row block table). Per-row causal limit as
                        # a recorded 0-d device op off the LIVE seq_lens
                        # buffer: row r attends [0 .. seq-q+r] inclusive ==
                        # stock synth values [cached+1 .. cached+q]. No
                        # synthesized multi-row tensors, no stride-0 expand,
                        # no multi-row grid — isolates each replay suspect.
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
                        # v3 synth dispatch (A/B; multi-row synthetic decode)
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
        "P619 tq_spec_fix v4 installed (active=%s mode=%s): v19b verify-"
        "capture forcing + %s continuation dispatch",
        _TQ_SPEC_FIX, _VERIFY_MODE,
        "split-row" if _VERIFY_MODE == "split" else "v3-synth",
    )
