# SPDX-License-Identifier: Apache-2.0
"""tq_telemetry.py — P6.19c end-to-end telemetry for the TQ×MTP corruption
(diagnostic-only injector patch; no behavior change).

Owner directive 2026-11-04: comprehensive telemetry capture points
end-to-end to locate the root issue. All lines tagged P619T.

Capture points:

  FWD    — every distinct batch shape entering
           TurboQuantAttentionImpl.forward: is_prefill, n_tok, n_req,
           qsl/seq_lens heads, num_decodes, max_query_len/max_seq_len,
           + LAYER IDENTITY (target vs drafter layers are distinguishable
           by layer_name). Proven shapes (leg t1i): verify arrives as
           is_prefill=True q_len=4 extend batches.

  STORE  — every _store_kv call: n_tok + per-layer SLOT-REWRITE count
           (slots this layer already wrote). Under MTP, rejected-draft
           slots are re-stored — rewrite count tracks rejection activity.
           GPU uint8 flags per (layer, slot): no per-call host sync.

  CANARY — ground-truth divergence at the READ side: for the first two
           distinct layers, a bf16 SHADOW KV copy of the first 8192
           CONTEXT POSITIONS is kept from the raw key/value forward args
           (position-keyed: after a rejection the next step's raw args are
           the new ground truth for those positions). At every
           verify-shaped step (is_prefill=True, 2<=max_query_len<=8)
           reference causal attention (fp32 SDPA over shadow K/V) is
           computed for request 0 and compared to the backend's live
           output: per-row rel-err + norm-ratio. The ABSOLUTE floor is the
           dtype's quantization noise (k8v4: 4-bit values -> O(1e-2..1e-1)
           legitimate); the signal is the DYNAMICS vs the spec-off control
           leg (t2c) run with identical telemetry:
           - rel spikes at rewrite bursts -> store-rewrite (stale bytes)
           - rel grows smoothly with ctx  -> dequant/accumulation defect
           - rel flat-at-floor while output corrupts -> these layers are
             clean; hunt moves to drafter pool / GDN / other layers

Bounded cost: shadow 2 layers x 8192 pos x K+V bf16 (~50 MB), canary on
request 0 only, first 80 verify steps per layer, SDPA fp32 (24x4x8192).
"""

import os
import torch

try:  # vllm-configured channel (visible in serve logs)
    from vllm.logger import init_logger as _init_logger
    logger = _init_logger(__name__)
except Exception:  # noqa: BLE001
    import logging
    logger = logging.getLogger(__name__)

_SHADOW_TOKENS = int(os.environ.get("VLLM_ARC_TQ_TELEM_SHADOW", "8192"))
_SHADOW_LAYERS = 4          # first N distinct layers get shadow+canary
                           # (4 = 2 target + drafter layers if they use TQ)
_CANARY_MAX = 300           # verify steps canaried per layer (warmup eats some)
_FWD_LOUD = 20              # distinct FWD shape classes logged loudly
_SLOTS_CAP = 1_500_000      # slot-flag tensor bound (pool ~400k tokens)

_seen_shapes: dict = {}
_layer_ord: dict[int, int] = {}
_mtp_shadow: list = []       # first mtp.* layer also gets shadow+canary
_slot_flags: dict[int, torch.Tensor] = {}   # layer-key -> uint8 flags
_rewrite_count: dict[int, int] = {}         # layer-key -> events seen
_rewrite_logged: dict[int, int] = {}        # layer-key -> lines emitted
_canary_count: dict[int, int] = {}          # layer-key -> canary lines


def _md_fields(md):
    qsl_cpu = getattr(md, "query_start_loc_cpu", None)
    sl_cpu = getattr(md, "seq_lens_cpu", None)
    # tuples (hashable — round-1 lesson: list-in-dict-key raised TypeError
    # on every forward and killed the canary)
    qsl = tuple(qsl_cpu.tolist()[:9]) if (qsl_cpu is not None and qsl_cpu.numel()) else None
    seqs = tuple(sl_cpu.tolist()[:9]) if (sl_cpu is not None and sl_cpu.numel()) else None
    n_req = (len(qsl_cpu) - 1) if qsl_cpu is not None and qsl_cpu.numel() else None
    return (
        getattr(md, "is_prefill", None),
        getattr(md, "num_actual_tokens", None),
        n_req, qsl, seqs,
        getattr(md, "num_decodes", None),
        getattr(md, "num_decode_tokens", None),
        getattr(md, "max_query_len", None),
        getattr(md, "max_seq_len", None),
    )


class _Shadow:
    """bf16 ground-truth KV by CONTEXT POSITION for one layer."""

    __slots__ = ("k", "v", "filled", "hk", "d")

    def __init__(self):
        self.k = None
        self.v = None
        self.filled = 0
        self.hk = None
        self.d = None

    def ensure(self, hk: int, d: int, dev):
        if self.k is None:
            self.k = torch.zeros(_SHADOW_TOKENS, hk, d,
                                 dtype=torch.bfloat16, device=dev)
            self.v = torch.zeros(_SHADOW_TOKENS, hk, d,
                                 dtype=torch.bfloat16, device=dev)
            self.hk = hk
            self.d = d

    def put(self, pos0: int, k, v):
        n = k.shape[0]
        if pos0 >= _SHADOW_TOKENS or pos0 < 0:
            return
        n = min(n, _SHADOW_TOKENS - pos0)
        self.k[pos0 : pos0 + n] = k[:n].to(torch.bfloat16)
        self.v[pos0 : pos0 + n] = v[:n].to(torch.bfloat16)
        self.filled = max(self.filled, pos0 + n)


def _sdpa_reference(q_rows, k_all, v_all, scale: float, num_kv_groups: int):
    """Causal reference attention. q (r, Hq, D); k/v (T, Hk, D) -> (r, Hq, D)."""
    r, T = q_rows.shape[0], k_all.shape[0]
    qh = q_rows.transpose(0, 1).float()   # (Hq, r, D)
    kh = k_all.transpose(0, 1).float()    # (Hk, T, D)
    vh = v_all.transpose(0, 1).float()
    if num_kv_groups > 1:
        kh = kh.repeat_interleave(num_kv_groups, dim=0)
        vh = vh.repeat_interleave(num_kv_groups, dim=0)
    att = torch.matmul(qh, kh.transpose(1, 2)) * scale   # (Hq, r, T)
    # round-2 lesson: att is rank-3 — causal mask needs (1,1,T) vs (1,r,1)
    pos = torch.arange(T, device=q_rows.device).view(1, 1, -1)
    lim = torch.arange(r, device=q_rows.device).view(1, -1, 1) + (T - r)
    att = att.masked_fill(pos > lim, float("-inf"))
    out = torch.softmax(att, dim=-1) @ vh                # (Hq, r, D)
    return out.transpose(0, 1)                           # (r, Hq, D)


def _install(mod) -> None:
    Impl = getattr(mod, "TurboQuantAttentionImpl", None)
    if Impl is None:
        return
    if getattr(Impl.forward, "_arc_p619_telemetry", False):
        return

    shadows: dict[int, _Shadow] = {}
    orig_fwd = Impl.forward
    orig_store = Impl._store_kv

    def _lkey(layer):
        k = id(layer)
        if k not in _layer_ord:
            _layer_ord[k] = len(_layer_ord)
        return k

    def _lname(layer, lk):
        return getattr(layer, "layer_name", None) or f"ord{_layer_ord[lk]}"

    def _flags(lk, dev):
        f = _slot_flags.get(lk)
        if f is None or f.device != dev:
            f = torch.zeros(_SLOTS_CAP, dtype=torch.uint8, device=dev)
            _slot_flags[lk] = f
        return f

    def _store_kv(self, key, value, kv_cache, slot_mapping, layer):
        try:
            lk = _lkey(layer)
            nm = _lname(layer, lk)
            n = slot_mapping.shape[0]
            if str(nm) not in _seen_shapes:
                _seen_shapes[str(nm)] = True
                logger.warning("P619T STORE first: layer=%s n_tok=%d", nm, n)
            dev = slot_mapping.device
            f = _flags(lk, dev)
            valid = slot_mapping >= 0
            idx = slot_mapping.clamp_min(0)
            prev = f[idx]                                 # uint8 was-written
            rw_mask = (prev > 0) & valid                  # -1 pads don't count
            n_rw = int(rw_mask.sum().item())
            f[idx[valid]] = 1
            if n_rw:
                _rewrite_count[lk] = _rewrite_count.get(lk, 0) + n_rw
                emitted = _rewrite_logged.get(lk, 0)
                # first 3 rewrite events per layer + every 400th after
                if emitted < 3 or _rewrite_count[lk] // 400 > (emitted - 3):
                    _rewrite_logged[lk] = emitted + 1
                    logger.warning(
                        "P619T STORE rewrite: layer=%s n_tok=%d n_rewrite=%d "
                        "total_rw=%d slots=%s",
                        nm, n, n_rw, _rewrite_count[lk],
                        slot_mapping[rw_mask][:8].tolist(),
                    )
        except Exception:  # noqa: BLE001 — telemetry must never break serving
            logger.exception("P619T STORE telemetry error")
        return orig_store(self, key, value, kv_cache, slot_mapping, layer)

    def _capture(sh, k_full, v_full, qsl, seqs):
        """Stash raw K/V spans of every request whose chunk starts inside the
        shadow window; positions derived from seq_lens - q_len (telemetry
        shape data: seq_lens INCLUDES the current chunk)."""
        for i in range(len(qsl) - 1):
            q0, q1 = qsl[i], qsl[i + 1]
            r = q1 - q0
            if r <= 0:
                continue
            pos0 = seqs[i] - r
            if 0 <= pos0 < _SHADOW_TOKENS:
                sh.put(pos0, k_full[q0:q1], v_full[q0:q1])

    def _forward(self, layer, query, key, value, kv_cache, attn_metadata,
                 output=None, output_scale=None, output_block_scale=None):
        canary_ctx = None
        try:
            lk = _lkey(layer)
            nm = _lname(layer, lk)
            md = attn_metadata
            if md is not None:
                f = _md_fields(md)
                key_s = ("fwd",) + f
                if key_s not in _seen_shapes:
                    _seen_shapes[key_s] = True
                    if len(_seen_shapes) <= _FWD_LOUD:
                        logger.warning(
                            "P619T FWD layer=%s is_prefill=%s n_tok=%s n_req=%s "
                            "qsl=%s seqs=%s ndec=%s ndectok=%s mql=%s msl=%s",
                            nm, *f)
            # shadow+canary: first 4 distinct TARGET layers (warmup order
            # guarantees target prefills precede any mtp forward) + the
            # first mtp.* drafter layer
            nm_s = str(nm)
            if nm_s.startswith("mtp") and not _mtp_shadow:
                _mtp_shadow.append(lk)
            instrumented = (
                (_layer_ord[lk] < _SHADOW_LAYERS or lk in _mtp_shadow)
                and md is not None
            )
            if instrumented:
                qsl_cpu = getattr(md, "query_start_loc_cpu", None)
                sl_cpu = getattr(md, "seq_lens_cpu", None)
                if qsl_cpu is not None and qsl_cpu.numel() and \
                        sl_cpu is not None and sl_cpu.numel():
                    qsl = qsl_cpu.tolist()
                    seqs = sl_cpu.tolist()
                    sh = shadows.get(lk)
                    if sh is None:
                        sh = shadows[lk] = _Shadow()
                    k_full = key.view(key.shape[0], -1, self.head_size)
                    v_full = value.view(value.shape[0], -1, self.head_size)
                    sh.ensure(k_full.shape[1], self.head_size, k_full.device)
                    _capture(sh, k_full, v_full, qsl, seqs)
                    mql = int(getattr(md, "max_query_len", 0) or 0)
                    if getattr(md, "is_prefill", False) and 2 <= mql <= 8:
                        canary_ctx = (lk, sh, qsl, seqs, nm)
        except Exception:  # noqa: BLE001
            logger.exception("P619T FWD pre telemetry error")

        result = orig_fwd(self, layer, query, key, value, kv_cache,
                          attn_metadata, output=output,
                          output_scale=output_scale,
                          output_block_scale=output_block_scale)

        if canary_ctx is not None and result is not None:
            try:
                lk, sh, qsl, seqs, nm = canary_ctx
                if _canary_count.get(lk, 0) < _CANARY_MAX:
                    _canary_count[lk] = _canary_count.get(lk, 0) + 1
                    i = 0
                    q0, q1 = qsl[i], qsl[i + 1]
                    r = q1 - q0
                    ctx = seqs[i]
                    pos0 = ctx - r
                    if 0 < ctx <= sh.filled and ctx <= _SHADOW_TOKENS:
                        q_rows = query[q0:q1].view(r, -1, self.head_size)
                        k_t = key[q0:q1].view(r, -1, self.head_size)
                        v_t = value[q0:q1].view(r, -1, self.head_size)
                        kk = torch.cat([sh.k[:pos0], k_t.to(torch.bfloat16)], 0)
                        vv = torch.cat([sh.v[:pos0], v_t.to(torch.bfloat16)], 0)
                        ref = _sdpa_reference(q_rows, kk, vv,
                                              self.scale, self.num_kv_groups)
                        live = result[q0:q1].view(r, -1, self.head_size)
                        refn = ref.float().norm(dim=-1)
                        denom = refn.clamp_min(1e-6)
                        rel = ((live.float() - ref.float())
                               .norm(dim=-1) / denom)          # (r, Hq)
                        ratio = (live.float().norm(dim=-1) / denom)   # (r, Hq)
                        logger.warning(
                            "P619T CANARY layer=%s step=%d ctx=%d rows=%d "
                            "rel_max=%.3e rel_mean=%.3e ratio_min=%.3f "
                            "ratio_max=%.3f rw_total=%d",
                            nm, _canary_count[lk], ctx, r,
                            float(rel.max()), float(rel.mean()),
                            float(ratio.min()), float(ratio.max()),
                            _rewrite_count.get(lk, 0),
                        )
            except Exception:  # noqa: BLE001
                logger.exception("P619T CANARY telemetry error")
        return result

    _forward._arc_p619_telemetry = True
    Impl.forward = _forward
    Impl._store_kv = _store_kv
    logger.warning("P619T telemetry installed (shadow=%d pos x %d layers, "
                   "canary<=%d steps/layer)", _SHADOW_TOKENS, _SHADOW_LAYERS,
                   _CANARY_MAX)
