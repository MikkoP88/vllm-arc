# SPDX-License-Identifier: Apache-2.0
"""sched_tail_split.py — r35 fork patch (vllm_gaudi/xpu/patches/).

Engine-core scheduler delta (v4, scoped): when the eagle backoff would
leave NOTHING bankable, plan the mamba-aligned chunk end at the deepest
full block anyway — while the eagle lookup-side pop (the coordinator's
bit, handed over at scheduler init) stays fully active.

The composition-coherence law (r35, probe-proven A/B + control):
  A warm resume is byte-coherent only at depths that are COLD chunk-end
  states, and the drop-backoff pins chunk ends to popped depth. Chunking
  deeper than the pop (the r35 v3 delta, applied to all prompts) changes
  the cold's own numeric path — every warm that does not follow the new
  composition (local pop-hit recomputes, unbanked spans) flips
  marginally-sensitive prompts: golden 3/5, N4 warm[0] PREFILL_FLIP, cold
  shas re-valued to the r34-fix-boot values (d9f6019...). Control knob-off:
  golden 5/5, all canonical. Banked in
  results/2026-10-29-p3-r34-eagle-block-drop-falsified (r35 addendum).
  THE 8064-depth warm on 8k prompts is therefore structurally unreachable
  without breaking 0-drift: the state@8064 the upstream cold computes is
  mid-chunk (never a discrete checkpoint — the p41c exact-boundary law).

The one coherent exception (this patch): prompts where upstream banks
NOTHING — last_cache_position with the backoff == 0 while full blocks
exist (token counts in (block, 2*block], e.g. the 1k face at (896,1792]).
There the delta ADDS the deepest-full-block chunk end instead of moving an
existing one: cold chunks [0,B) + [B,n), banks B (p41c snap + adapter
store at the aligned end), and the warm = external load B + residual
[B,n) — the residual composition EQUALS the cold's chunk2, carried from
the same frozen state, so within-run cold==warm holds (N3 cell: sha
8a5edab2 stable across cold + 5 warms, warm med 740.4 ms <= 750 GATE PASS
vs 9.2 s with nothing banked). Prompts outside the class keep upstream's
split bit-for-bit (wrapper is a passthrough).

Scope of the flip (audit, upstream-0.30 scheduler.py): use_eagle_block_drop
is read at runtime only inside _mamba_block_aligned_split (:444 the backoff,
:451 checkpoint position, :487 tail-boundary shift). :451 is inert without
mamba prefill checkpoint blocks (not configured) and :487 without
mamba_partial_cache_hit (needs hash_block_size < block_size; ours are equal)
— the first split call logs both bits once to prove it. The coordinator's
eagle bit (:304, constructor) and every spec_config read elsewhere are
outside the flip window.

Known accepted trade: inside the class, the cold's chunk composition
changes (upstream ran one exempt chunk); within-run determinism holds
(measured), cross-posture marginal drift on sensitive prompts is possible
in principle (same class as the documented cross-boot marginal flips).

Install: sitecustomize exec_module hook on vllm.v1.core.sched.scheduler
(r30_sitecustomize pattern) calls _install(module).
"""

import logging
import os

logger = logging.getLogger(__name__)

# r35 knob: 0 restores upstream's backoff exactly (A/B / falsification).
_R35_TAIL_SPLIT = os.environ.get("VLLM_ARC_R35_TAIL_SPLIT", "1") != "0"


def _install(mod) -> None:
    """Wrap Scheduler._mamba_block_aligned_split. Idempotent."""
    S = mod.Scheduler
    if getattr(S._mamba_block_aligned_split, "_arc_r35_tail_split", False):
        return
    orig = S._mamba_block_aligned_split

    def _split_tail_deep(self, *args, **kwargs):
        if not _R35_TAIL_SPLIT or not self.use_eagle_block_drop:
            return orig(self, *args, **kwargs)
        try:
            req = args[0] if args else kwargs.get("request")
        except Exception:  # noqa: BLE001
            req = None
        bs = self.cache_config.block_size
        deepest_full = 0
        banked_upstream = 0
        if req is not None:
            n = req.num_tokens
            deepest_full = n - n % bs
            banked_upstream = max(deepest_full - bs, 0)
        # v4 scope: only where the backoff banks nothing but a full block
        # exists (the wrapper is otherwise a pure passthrough).
        if not (deepest_full > 0 and banked_upstream == 0):
            return orig(self, *args, **kwargs)
        prev = self.use_eagle_block_drop
        self.use_eagle_block_drop = False
        try:
            out = orig(self, *args, **kwargs)
        finally:
            self.use_eagle_block_drop = prev
        if os.environ.get("ARC_SNAP_PROBE"):  # r35 diagnostic (silent off)
            st = globals().setdefault("_arc_r35_probe", 0) + 1
            globals()["_arc_r35_probe"] = st
            if st <= 8:
                _rid = getattr(req, "request_id", "?")
                print(
                    f"[r35probe split#{st} req={_rid} n={req.num_tokens} "
                    f"deepest_full={deepest_full} "
                    f"ckpt_blocks={self.mamba_has_prefill_checkpoint_blocks} "
                    f"partial_hit={self.mamba_partial_cache_hit} "
                    f"new={out}]",
                    flush=True,
                )
        return out

    _split_tail_deep._arc_r35_tail_split = True
    S._mamba_block_aligned_split = _split_tail_deep
    logger.info(
        "arc-r35 sched tail-split v4 installed (no-backoff only when "
        "upstream banks nothing; lookup pop untouched, enabled=%s)",
        _R35_TAIL_SPLIT,
    )
