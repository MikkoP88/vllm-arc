# SPDX-License-Identifier: Apache-2.0
"""GDN attention-metadata fast path (p10) — decode/spec-decode builds without
CPU-mask indexing of device tensors.

Evidence (results/2026-10-28-p2-*: r19-r21b step decomposition, r22a A/B +
stepprobe + evprobe): the steady c8 graphed+MTP-k4 step runs ~80 ms wall of
which ~50 ms is CPU-serialized inside
``GDNAttentionMetadataBuilder.build``. The blocking sites are the two
CPU-bool-mask advanced indexes on XPU tensors —

    block_table_tensor[spec_sequence_masks_cpu, : num_spec + 1]
    num_accepted_tokens[spec_sequence_masks_cpu]

— each stages the mask host->device with an implicit *synchronous* copy. On
an in-order XPU stream a blocking H2D waits for everything enqueued before
it, i.e. the previous step's graph replay + eager tail, so the worker thread
stalls inside build() for the remaining pipeline length (trace signature:
~63 ms gaps "before IndexKernelFunctor"; one slow build of the three per
step, p90 69 ms). The evprobe (r22a) exonerated Event.synchronize itself
(completed-event sync 0.003 ms); the wait is honest in-order pipeline
progress, and the build's blocking ops are what couple the CPU serial
section to it.

This patch keeps the stock ``build`` for every non-steady shape (prefill /
mixed / non-spec / zero-spec / above capture size) and replaces only the
steady decode-spec construction with an all-async equivalent. Classification
stays on CPU tensors (no GPU syncs); row selection becomes plain slices —
spec rows are front-contiguous (the invariant core itself relies on when it
slices ``query_start_loc[: num_spec_decodes + 1]``; guarded at runtime, falls
back if ever violated) — so there is no mask indexing, no H2D, and no
``nonzero()`` (the p10 rig probe showed ``nonzero``'s count readback blocks
behind the stream exactly like the mask index it would replace). The token
index is a persistent arange slice; static-buffer tail fills use CPU-known
scalars (a 0-dim GPU ``fill_`` value would ``.item()`` and sync). Outputs
are bit-identical to core's; the full-cudagraph static-buffer staging is
core's own. Budget law (plan §4 p10): build_attn_metadata <= 5 ms/step @c8 —
measured by the in-patch sampler and logged every 256 fast builds.

Gate: ``VLLM_ARC_GDN_FASTPATH_DISABLE=1`` disables the install (A/B control).
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, mamba_get_block_table_tensor

logger = init_logger("vllm.platforms.arc.p10")

_installed: bool = False


class _BuildStats:
    """Per-process build-wall sampler (p10 budget-law evidence)."""

    def __init__(self) -> None:
        self.fast_walls: deque[float] = deque(maxlen=512)
        self.n_fast = 0
        self.n_fallback = 0
        self.fallback_reasons: dict[str, int] = {}
        self._last_log = 0.0

    def record_fast(self, wall_ms: float) -> None:
        self.fast_walls.append(wall_ms)
        self.n_fast += 1
        if self.n_fast % 256 == 0:
            walls = sorted(self.fast_walls)
            n = len(walls)
            logger.info(
                "[arc-p10] gdn fast build: n=%d p50=%.2fms p90=%.2fms max=%.2fms "
                "fallbacks=%d %s",
                self.n_fast,
                statistics.median(walls),
                walls[int(n * 0.9)],
                walls[-1],
                self.n_fallback,
                self.fallback_reasons or "",
            )

    def record_fallback(self, reason: str) -> None:
        self.n_fallback += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1


def _fast_build(  # noqa: PLR0911, PLR0912, PLR0915
    self: GDNAttentionMetadataBuilder,
    common_prefix_len: int,
    common_attn_metadata,
    num_accepted_tokens=None,
    num_decode_draft_tokens_cpu=None,
    fast_build: bool = False,
):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    stats: _BuildStats = self._arc_p10_stats  # type: ignore[attr-defined]
    t0 = time.perf_counter()

    m = common_attn_metadata
    query_start_loc = m.query_start_loc
    query_start_loc_cpu = m.query_start_loc_cpu

    if (
        not self.use_spec_decode
        or not self.use_full_cuda_graph
        or num_decode_draft_tokens_cpu is None
        or num_accepted_tokens is None
    ):
        stats.record_fallback("shape")
        return self._arc_p10_orig_build(
            common_prefix_len,
            m,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )

    # ---- CPU-side classification (identical arithmetic to core; CPU tensors
    # only, no GPU sync) ----
    spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
    num_spec_decodes = int(spec_sequence_masks_cpu.sum())
    if num_spec_decodes == 0 or int(
        num_decode_draft_tokens_cpu[spec_sequence_masks_cpu].sum()
    ) == 0:
        stats.record_fallback("zero-spec")
        return self._arc_p10_orig_build(
            common_prefix_len, m, num_accepted_tokens, num_decode_draft_tokens_cpu,
            fast_build,
        )

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    non_spec_lens = query_lens_cpu[~spec_sequence_masks_cpu]
    num_decodes = int((non_spec_lens == 1).sum())
    num_zero_len = int((non_spec_lens == 0).sum())
    num_prefills = non_spec_lens.numel() - num_decodes - num_zero_len
    if num_decodes > 0 or num_prefills > 0:
        # Mixed batch: core reclassifies plain decodes as prefills and takes
        # the token-reordering branch — delegate, correctness over speed.
        stats.record_fallback("mixed")
        return self._arc_p10_orig_build(
            common_prefix_len, m, num_accepted_tokens, num_decode_draft_tokens_cpu,
            fast_build,
        )
    if num_spec_decodes > self.decode_cudagraph_max_bs:
        stats.record_fallback("over-max-bs")
        return self._arc_p10_orig_build(
            common_prefix_len, m, num_accepted_tokens, num_decode_draft_tokens_cpu,
            fast_build,
        )

    num_spec_decode_tokens = int(query_lens_cpu.sum())
    if num_spec_decode_tokens > self.decode_cudagraph_max_bs:
        stats.record_fallback("over-max-tokens")
        return self._arc_p10_orig_build(
            common_prefix_len, m, num_accepted_tokens, num_decode_draft_tokens_cpu,
            fast_build,
        )

    # ---- GPU-side construction: every op below is an async kernel launch or
    # a view. No CPU-mask indexing of device tensors, no H2D at all, and no
    # nonzero() (the p10 probe showed nonzero's count D2H blocks behind the
    # stream exactly like the mask index it would replace). Spec rows are
    # front-contiguous — the same invariant core relies on when it slices
    # query_start_loc[: num_spec_decodes + 1] ("padded sequences are always
    # at the back") — so row selection is a slice. Guarded; falls back if
    # the invariant ever breaks. ----
    if not bool(spec_sequence_masks_cpu[:num_spec_decodes].all()):
        stats.record_fallback("non-contiguous")
        return self._arc_p10_orig_build(
            common_prefix_len, m, num_accepted_tokens, num_decode_draft_tokens_cpu,
            fast_build,
        )

    block_table_tensor = mamba_get_block_table_tensor(
        m.block_table_tensor,
        m.seq_lens,
        self.kv_cache_spec,
        self.vllm_config.cache_config.mamba_cache_mode,
    )
    spec_state_indices_tensor = block_table_tensor[:num_spec_decodes, : self.num_spec + 1]
    num_accepted = num_accepted_tokens[:num_spec_decodes]

    spec_token_size = min(num_spec_decodes * (self.num_spec + 1), int(query_start_loc_cpu[-1]))
    arange_buf = getattr(self, "_arc_p10_arange", None)
    if arange_buf is None or arange_buf.numel() < spec_token_size:
        arange_buf = torch.arange(
            self.decode_cudagraph_max_bs * (self.num_spec + 1),
            dtype=torch.int32,
            device=query_start_loc.device,
        )
        self._arc_p10_arange = arange_buf
    spec_token_indx = arange_buf[:spec_token_size]
    non_spec_token_indx = self.non_spec_token_indx[:0]
    spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]

    # ---- full-cudagraph static-buffer staging (core's, with every fill
    # scalar CPU-known: in this branch query_start_loc[num_spec_decodes]
    # equals num_spec_decode_tokens — the sum of the front-contiguous spec
    # rows' lengths — so no 0-dim GPU tensor is passed to fill_ (which
    # would .item() and sync). ----
    batch_size = m.num_reqs
    self.spec_state_indices_tensor[:num_spec_decodes].copy_(
        spec_state_indices_tensor, non_blocking=True
    )
    spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
    spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

    self.spec_sequence_masks[:num_spec_decodes].fill_(True)
    spec_sequence_masks = self.spec_sequence_masks[:batch_size]
    spec_sequence_masks[num_spec_decodes:].fill_(False)

    self.spec_token_indx[: spec_token_indx.size(0)].copy_(
        spec_token_indx, non_blocking=True
    )
    spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

    self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
        spec_query_start_loc, non_blocking=True
    )
    spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
    spec_query_start_loc[num_spec_decodes + 1 :].fill_(num_spec_decode_tokens)

    self.num_accepted_tokens[:num_spec_decodes].copy_(num_accepted, non_blocking=True)
    num_accepted_tokens = self.num_accepted_tokens[:batch_size]
    num_accepted_tokens[num_spec_decodes:].fill_(1)

    stats.record_fast((time.perf_counter() - t0) * 1e3)
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_spec_decodes,
        num_spec_decode_tokens=num_spec_decode_tokens,
        num_actual_tokens=m.num_actual_tokens,
        has_initial_state=None,
        chunk_indices=None,
        chunk_offsets=None,
        prefill_query_start_loc=None,
        prefill_state_indices=None,
        prefill_has_initial_state=None,
        spec_query_start_loc=spec_query_start_loc,
        non_spec_query_start_loc=None,
        spec_state_indices_tensor=spec_state_indices_tensor,
        non_spec_state_indices_tensor=None,
        spec_sequence_masks=spec_sequence_masks,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=num_accepted_tokens,
        nums_dict=None,
        batch_ptr=None,
        token_chunk_offset_ptr=None,
    )


def _install() -> bool:
    """Replace GDNAttentionMetadataBuilder.build with the fast path.

    Idempotent; a no-op when ``VLLM_ARC_GDN_FASTPATH_DISABLE`` is set (leg
    A/B control). Falls back to the stock build for every shape the fast
    path does not own.
    """
    global _installed
    if _installed:
        return True
    if os.environ.get("VLLM_ARC_GDN_FASTPATH_DISABLE", "") not in ("", "0"):
        logger.info("[arc-p10] gdn metadata fastpath DISABLED by env")
        return False
    if getattr(GDNAttentionMetadataBuilder, "_arc_p10_stats", None) is not None:
        _installed = True
        return True

    GDNAttentionMetadataBuilder._arc_p10_stats = _BuildStats()
    GDNAttentionMetadataBuilder._arc_p10_orig_build = GDNAttentionMetadataBuilder.build
    GDNAttentionMetadataBuilder.build = _fast_build
    _installed = True
    logger.info(
        "[arc-p10] gdn metadata fastpath installed "
        "(steady spec-decode builds; stock build kept for prefill/mixed)"
    )
    return True
