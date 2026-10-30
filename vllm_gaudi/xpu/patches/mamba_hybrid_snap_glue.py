# SPDX-License-Identifier: Apache-2.0
"""mamba_hybrid_snap_glue.py — p41c fork patch (vllm_gaudi/xpu/patches/).

p41c PROBLEM (r31, 2026-10-28): p41's v136 snap lives in the carry branch of
``vllm.v1.worker.mamba_utils.preprocess_mamba`` — and under our product
posture (MTP k4 + hybrid GDN + mamba_cache_mode=align, vllm 0.30) that
function NEVER RUNS. r31 probe evidence (results/2026-10-28-p3-r31):
0 ppm calls across a 5-face x 4-request ladder (prefill AND decode), while
``[r31probe mh.preprocess_state#N]`` / ``prepare_attn`` / ``postprocess_state``
fire every step in both TP workers. 0.30 runs the mamba state machinery
through the model-states V2 path (``MambaHybridModelState``): the carries
are issued by the fused pre-copy kernel launched inside
``preprocess_state`` (mamba_hybrid.py:212), so there is no host carry
branch anywhere in mamba_utils. Net effect before p41c: the boundary
registry stayed empty forever -> the adapter's wait_for_save mamba branch
found nothing to store -> every m-lookup returned 0 -> the joint clamp
killed every external hit ("LMCache hit tokens: 0" on every face; ALL warm
speedup was vllm APC alone). The C champion's own wrap (same md5-1a6ae0ec
lineage) is equally dead under specdec — its 750 ms warm bar was APC-only.

p41c FIX (the v136 law re-anchored onto the live site, provenance:
llm-scaler v136 semantics, line C md5 1a6ae0ec / line D 0.26-re-anchor
md5 24e734b0 — see external_patches/lmcache/README.md row e02):
wrap ``MambaHybridModelState.preprocess_state``. Per step:

  BEFORE the fused kernel launch (insertions 1+2 of the v136 port):
  - consume a retrieved boundary seed the adapter's retrieve parked in
    ``_LMC_SEED_OVERRIDE`` and write it into the carry-from page
    (block B//bs - 1; ``add_request`` seeds the GPU state idx from the
    same num_computed, so the kernel carries the seed into the running
    block — exactly the dead path's contract), then run the v135 fp16
    repair over that page.
  AFTER the kernel launch (insertions 3+4):
  - for each row at an EXACT chunk-end boundary (ncc % bs == 0 — r31
    root cause #2: a boundary crossed mid-chunk never materializes in
    the block table; the pre-copy carries bt[src] -> bt[dst] directly,
    skipping the crossed column, and the postprocess migration only
    fires on chunk-end-aligned boundaries, so bt[ncc//bs - 1] is a
    stale/foreign page for mid-chunk crossings — banking it poisoned
    every warm resume; probe-proven 2026-10-29: ncc 7168 -> 8192 with
    no 8064 step, snap B=8064 stored a page no write ever touched),
    D2H-freeze B = ncc into the registry via the p41 glue's
    ``_lmc_snap_boundary`` (same-B re-snaps replace; registry caps
    8/req + FIFO 48 global as before). The snap's device sync is
    stream-ordered after the just-launched kernel. Decode-step
    crossings migrate correctly (postprocess fires on the aligned
    running state), so the exact-boundary rule is uniform over
    prefill and decode.

Everything else (registry contract, adapter pack order group-major/
layer/conv-ssm, m-store tokens[:deepest]) is the C adapter's byte-preserved
protocol — unchanged.

r31 deploy-defect chain (fixed 2026-10-29, each found via the [p41c]
stdout diagnostics added after the first silent failure — vllm_gaudi.*
logger warnings are INVISIBLE in engine logs, root-not-configured law):
the first three revisions reached for V1 gpu_input_batch attribute
names that do not exist on the V2-path InputBatch
(v1/worker/gpu/input_batch.py) — ``len(input_batch.block_table)``
(MultiGroupBlockTable has __getitem__ but NO __len__), then
``num_computed_tokens_cpu`` (V2 name: ``num_computed_tokens_np``), then
``input_batch.block_table`` itself (V2 InputBatch carries NO block
table; per-group ids live in the ``block_tables`` ARGUMENT, a tuple of
[num_reqs_padded, max_num_blocks] GPU tensors from
model_runner.prepare_attn). Each defect threw straight into this
wrap's own except -> zero snaps, zero mamba stores, every joint lookup
mamba_hit=0, warm times APC-only. Symptom-to-fix turnaround for all
three: one ladder + one grep. ncc source stays
``input_batch.num_computed_tokens_np`` — the scheduler-truth mirror
(exact under our sync-scheduling posture; the docstring's "optimistic
mirror" caveat applies to async scheduling only; the GPU
``num_computed_tokens`` argument is a StagedWriteTensor buffer that is
all-zeros until apply_write later in the step — never read it here).

Disable knob: VLLM_ARC_LMC_MAMBA_SNAP=0 (whole wrap off; back to the
r31 zero-snap posture).

Install: sitecustomize exec_module hook on
``vllm.v1.worker.gpu.model_states.mamba_hybrid`` calls _install(module).
The p41 glue symbols (registry / snap / seed-write / fp16 repair) are
late-imported from the arc_p41_patch module at first call — mamba_utils
and mamba_hybrid import order is not guaranteed.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_MAMBA_SNAP = os.environ.get("VLLM_ARC_LMC_MAMBA_SNAP", "1") != "0"


class _RowState:
    """Shim giving _lmc_snap_boundary/_lmc_seed_write a req_state-shaped
    .block_ids view resolved from one row of the preprocess_state
    ``block_tables`` argument (V2-path: tuple of per-group GPU tensors,
    each [num_reqs_padded, max_num_blocks] — model_runner.prepare_attn
    gather; the V2 InputBatch itself carries NO block_table attribute,
    r31 third-deploy defect: first revision used the V1 names
    ``input_batch.block_table``/``num_computed_tokens_cpu`` and threw
    invisibly every step). D2H row-slice only runs at snap/seed events
    (once per 896-token boundary), never per step."""

    __slots__ = ("block_ids", )

    def __init__(self, block_tables, row, nblocks):
        self.block_ids = {
            gid: block_tables[gid][row, :nblocks].detach().cpu().tolist()
            for gid in range(len(block_tables))
        }


def _glue():
    """The p41 glue module (arc_p41_patch), imported lazily."""
    import sys

    m = sys.modules.get("arc_p41_patch")
    if m is None:
        import vllm.v1.worker.mamba_utils as _mu  # noqa: F401 (hook fires here)

        m = sys.modules.get("arc_p41_patch")
    if m is None:
        raise RuntimeError("arc_p41_patch (p41 mamba glue) not installed")
    return m


def _bmd5(fc, kv_cache_config, gids, row_state, blk):
    """Probe-only (ARC_SNAP_PROBE) md5 over one boundary block's pages —
    the SAME group→layer→st iteration order as the glue's snap pack and
    seed write, so the hashes compare across requests/boots. One 41 MB
    D2H per call; only used at deep boundaries in probe runs."""
    import hashlib

    h = hashlib.md5()
    for gid in gids:
        bid = row_state.block_ids[gid][blk]
        for lname in kv_cache_config.kv_cache_groups[gid].layer_names:
            for st in fc[lname].kv_cache:
                h.update(st[bid].reshape(-1).view(torch.uint8).detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _install(mod) -> None:
    """Wrap MambaHybridModelState.preprocess_state with the v136 blocks.
    Idempotent."""
    cls = getattr(mod, "MambaHybridModelState", None)
    if cls is None or getattr(cls, "_arc_p41c", False):
        return
    if not _MAMBA_SNAP:
        cls._arc_p41c = True
        return

    orig = cls.preprocess_state
    state = {
        "last_b": {},
        "fc": None,
        "gids": None,
        "bs": None,
        "warn": {},
        "live": False,
    }

    def _warn(tag, msg):
        """logger.warning + first-3 stdout prints (engine logs swallow
        vllm_gaudi.* logger output — root not configured; r31 law)."""
        logger.warning("arc-p41c %s: %s", tag, msg)
        n = state["warn"].get(tag, 0)
        if n < 3:
            state["warn"][tag] = n + 1
            print(f"[p41c {tag}#{n + 1}] {msg}", flush=True)

    def preprocess_state(self, input_batch, block_tables, kv_cache_config, num_computed_tokens):
        g = None
        try:
            g = _glue()
        except Exception as e:  # noqa: BLE001
            _warn("glue-unavailable", repr(e)[:160])
            return orig(self, input_batch, block_tables, kv_cache_config, num_computed_tokens)

        try:
            fc = self.vllm_config.compilation_config.static_forward_context
            gids, spec = self._get_mamba_group_info(kv_cache_config)
            bs = spec.block_size
            state.update(fc=fc, gids=gids, bs=bs)
        except Exception as e:  # noqa: BLE001
            _warn("ctx-resolve-failed", repr(e)[:160])
            return orig(self, input_batch, block_tables, kv_cache_config, num_computed_tokens)

        n = input_batch.num_reqs
        # --- (0) r31 probe (root cause #2): pre-orig md5 of the boundary and
        # running blocks for deep rows. Paired with the post-orig snapmd5
        # print it shows, per step, whether the fused pre-copy wrote the
        # running block and what the decision inputs were. ARC_SNAP_PROBE
        # runs only (diagnostic boots; 2x41 MB D2H per deep row per step).
        _chain_rows = []
        if os.environ.get("ARC_SNAP_PROBE"):
            for i in range(n):
                ncc = int(input_batch.num_computed_tokens_np[i])
                j = ncc // bs - 1
                if j < 6:
                    continue
                _chain_rows.append((i, j))
            for i, j in _chain_rows:
                try:
                    _rs = _RowState(block_tables, i, j + 2)
                    _slot = int(input_batch.idx_mapping[i])
                    print(
                        f"[r31probe p41c-pre req={input_batch.req_ids[i]} "
                        f"ncc={int(input_batch.num_computed_tokens_np[i])} "
                        f"sidx={int(self._mamba_state_idx_gpu[_slot].item())} "
                        f"nacc={int(self.num_accepted_tokens_gpu[_slot].item())} "
                        f"blk{j}={_bmd5(state['fc'], kv_cache_config, state['gids'], _rs, j)} "
                        f"blk{j + 1}={_bmd5(state['fc'], kv_cache_config, state['gids'], _rs, j + 1)}]",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[r31probe p41c-pre-ERR {repr(e)[:80]}]", flush=True)

        # --- (1)+(2) v136 load half: seed write + v135 fp16 repair into the
        # carry-from page, BEFORE the fused kernel launch below.
        for i in range(n):
            req_id = input_batch.req_ids[i]
            seed = g._LMC_SEED_OVERRIDE.pop(req_id, None)
            if seed is None:
                continue
            ncc = int(input_batch.num_computed_tokens_np[i])
            prev = ncc // bs - 1  # carry-from block at the booked depth
            if prev < 0:
                _warn("seed-dropped", f"req={req_id} ncc={ncc} B={seed[0]} (no carry-from block)")
                continue
            row_state = _RowState(block_tables, i, prev + 2)
            if seed[0] // bs - 1 != prev:
                _warn("seed-mismatch", f"req={req_id} B={seed[0]} prev={prev}")
                continue
            # r31 root-cause fix (2026-10-29): upstream add_request seeds the
            # running-state column as (num_computed-1)//cache_config.block_size
            # — the ATTENTION block size (64) — but block-table columns are
            # mamba blocks (bs=896). A remote-loaded resume (B=6272) lands at
            # col 97 instead of 6, so the fused pre-copy carries a FOREIGN
            # page and the seed written into block prev is never read
            # (integrity 0/5, golden len=4096 drift — the only live path that
            # ever calls add_request with num_computed>0; stock fresh reqs
            # seed -1 and vllm-APC resumes reattach cached mamba blocks via
            # the hybrid KV-cache manager). Correct the seeded column for
            # seed rows before the fused kernel reads it.
            try:
                slot = int(input_batch.idx_mapping[i])
                self._mamba_state_idx_gpu[slot] = prev
                if os.environ.get("ARC_SNAP_PROBE"):
                    print(
                        f"[r31probe p41c-sidx req={req_id} slot={slot}"
                        f" prev={prev}]",
                        flush=True,
                    )
            except Exception as e:  # noqa: BLE001
                _warn("state-idx-fix-failed", f"req={req_id} {repr(e)[:120]}")
                continue
            g._lmc_seed_write(
                req_id,
                seed[0],
                prev,
                bs,
                row_state,
                kv_cache_config,
                state["gids"],
                state["fc"],
                seed[1],
            )
            if os.environ.get("ARC_SNAP_PROBE"):
                import hashlib

                _smd5 = hashlib.md5(seed[1].cpu().numpy().tobytes()).hexdigest()[:16]
                _bids = {gid: row_state.block_ids[gid][prev:prev + 2] for gid in state["gids"]}
                print(
                    f"[r31probe p41c-seedmd5 req={req_id} B={seed[0]} "
                    f"prev={prev} seed_bytes={_smd5} "
                    f"postwrite={_bmd5(state['fc'], kv_cache_config, state['gids'], row_state, prev)} "
                    f"bids={_bids}]",
                    flush=True,
                )
                state.setdefault("seed_pending", {})[req_id] = (i, prev)
            if g._FP16_SEED_REPAIR:
                try:
                    for gid in state["gids"]:
                        lname = kv_cache_config.kv_cache_groups[gid].layer_names[0]
                        bids = row_state.block_ids[gid]
                        for st in state["fc"][lname].kv_cache:
                            if st.dtype == torch.float16:
                                pg = st[bids[prev]]
                                pg.copy_(torch.nan_to_num(
                                    pg,
                                    nan=0.0,
                                    posinf=65504.0,
                                    neginf=-65504.0,
                                ))
                except Exception as e:  # noqa: BLE001
                    _warn("fp16-repair-failed", repr(e)[:160])

        r = orig(self, input_batch, block_tables, kv_cache_config, num_computed_tokens)

        # --- r31 probe: the fused kernel's actual copy decision for seed
        # rows (post-orig: state_idx advanced, src_col/src_off emitted).
        if os.environ.get("ARC_SNAP_PROBE"):
            for _rid, (_i, _prev) in state.pop("seed_pending", {}).items():
                try:
                    _slot = int(input_batch.idx_mapping[_i])
                    print(
                        f"[r31probe p41c-postseed req={_rid} "
                        f"sidx={int(self._mamba_state_idx_gpu[_slot].item())} "
                        f"src_col={int(self._mamba_src_col_gpu[_slot].item())} "
                        f"src_off={int(self._mamba_src_off_gpu[_slot].item())}]",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[r31probe p41c-postseed-ERR {repr(e)[:80]}]", flush=True)

        # --- (3)+(4) v136 store half: freeze the deepest COMPLETE boundary
        # per row, after the fused kernel (sync ordered behind its writes).
        try:
            last_b = state["last_b"]
            if len(last_b) > 8192:  # bounded leak: finished reqs never pop
                for k in list(last_b)[:len(last_b) // 2]:
                    del last_b[k]
            state["calls"] = state.get("calls", 0) + 1
            _probe = bool(os.environ.get("ARC_SNAP_PROBE")) and state["calls"] <= 12
            if _probe and n:
                try:
                    _gpu = num_computed_tokens.cpu()[:n].tolist()
                except Exception:  # noqa: BLE001
                    _gpu = None
                print(
                    f"[r31probe p41c-ncc#{state['calls']} n={n} "
                    f"np={input_batch.num_computed_tokens_np[:n].tolist()} "
                    f"gpu={_gpu}]",
                    flush=True,
                )
            for i in range(n):
                req_id = input_batch.req_ids[i]
                ncc = int(input_batch.num_computed_tokens_np[i])
                if ncc % bs:
                    # r31 root cause #2 (2026-10-29, probe-proven): a boundary
                    # crossed MID-CHUNK never materializes in the block table.
                    # The fused pre-copy carries bt[src] -> bt[dst] directly
                    # (dst = ceil(computed_after/bs)-1, skipping the crossed
                    # column) and the postprocess migration only fires when
                    # aligned_new_computed >= num_tokens_running_state (chunk
                    # end aligned). bt[j] for such a boundary is a stale/
                    # foreign page no write ever touched (probe: ncc 7168 ->
                    # 8192 with no 8064 step; snap B=8064 banked garbage;
                    # warm resume re-seeded it -> integrity 0/5, golden 4096
                    # drift). Only exact chunk-end boundaries are present —
                    # decode crossings migrate correctly, so the rule is
                    # uniform: snap iff ncc % bs == 0.
                    continue
                j = ncc // bs - 1
                if j < 0:
                    continue
                b = ncc
                if b <= last_b.get(req_id, 0):
                    continue
                row_state = _RowState(block_tables, i, j + 2)
                g._lmc_snap_boundary(
                    req_id,
                    b,
                    j,
                    row_state,
                    kv_cache_config,
                    state["gids"],
                    state["fc"],
                )
                last_b[req_id] = b
                if os.environ.get("ARC_SNAP_PROBE"):
                    print(
                        f"[r31probe p41c-snap req={req_id} ncc={ncc} "
                        f"B={b} idx={j} bs={bs}]",
                        flush=True,
                    )
                if os.environ.get("ARC_SNAP_PROBE") and b >= 6272:
                    # md5 chain (r31 root-cause #2 probe): block j = the
                    # state leaving the GPU toward the pool (cold truth);
                    # block j+1 = the running state AFTER this step's
                    # pre-copy (what the forward reads). On a warm resume
                    # the same two prints at the same B give the warm-side
                    # truth — equality across the pair localizes the
                    # corruption to seed-content vs carry vs attn-load.
                    try:
                        print(
                            f"[r31probe p41c-snapmd5 req={req_id} B={b} "
                            f"blk{j}={_bmd5(state['fc'], kv_cache_config, state['gids'], row_state, j)} "
                            f"blk{j + 1}={_bmd5(state['fc'], kv_cache_config, state['gids'], row_state, j + 1)}]",
                            flush=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"[r31probe p41c-snapmd5-ERR {repr(e)[:80]}]", flush=True)
        except Exception as e:  # noqa: BLE001
            _warn("snap-failed", repr(e)[:160])
        if not state["live"]:
            state["live"] = True
            print(
                f"[p41c] wrap live (pid={os.getpid()} bs={bs} "
                f"gids={len(state['gids'])})",
                flush=True,
            )
        return r

    cls.preprocess_state = preprocess_state
    cls._arc_p41c = True
    print(f"[p41c] mamba_hybrid snap glue installed (pid={os.getpid()})", flush=True)
