# SPDX-License-Identifier: Apache-2.0
"""mamba_lmcache_glue.py — p41 fork patch (vllm_gaudi/xpu/patches/).

Worker-side LMCache mamba store/load glue for the hybrid-GDN target,
ported per the fork law from the certified donor lineages (provenance:
external_patches/lmcache/README.md, patch table row e02-vllm-side):

  - registry / seed-override symbols, _lmc_snap_boundary, _lmc_seed_write,
    and the v135 fp16 seed repair: llm-scaler v136/v135 (line C champion
    llm-scaler-exp:v1.2.35, vllm 0.30 mamba_utils, md5 1a6ae0ec) —
    semantically verbatim.
  - the fused-kernel re-anchor strategy (snapshot host site inside
    preprocess_mamba's carry branch): llm-scaler v136 "0.26 re-anchor"
    (line D champion vllm-iexperimental:v1.0.0, vllm 0.26 mamba_utils,
    md5 24e734b0) — upstream 0.21's host postprocess copy loop is fused
    into device kernels at 0.26+, so the snapshot fires where the host
    still knows the truth: when a carry stages, the deepest full
    boundary block is final and visible.

The lmcache-0.5.5+arc1 wheel's adapter (integration/vllm/vllm_adapter
graft) imports BOTH registry symbols from vllm.v1.worker.mamba_utils —
without this module the store path ImportErrors on the first finished
request and the joint (attn AND mamba) lookup clamps every hit to 0.

Not carried (REVAL ledger, external_patches/lmcache/README.md):
  - v138c sanitize-carry (NaN/Inf->0 on fallback-path carried pages):
    the align-mode fused path is the live posture; enable on evidence
    (LMCBOUND telemetry nf>0) if the fp16 pool ever shows poison.
  - LMCDbgFwd / LMCDbgM0 / v52f probe telemetry (debug-only donors).

Install: sitecustomize exec_module hook on vllm.v1.worker.mamba_utils
(the p10 gdn_metadata_fastpath wiring) calls _install(module).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# llm-scaler v136 (LMCache mamba store): worker-side boundary registry.
# Align mode nulls every non-running boundary block, so a historical
# boundary state only exists between the copy that materializes it and
# the next nulling pass -- the adapter's store can never read it back from
# the pool (the null-table law). Snapshot the boundary bytes (one page per
# mamba layer, every group) when a carry proves they are final;
# wait_for_save consumes the registry to bank REAL boundary states into
# the mamba LMCache engine. Layout: group-major, then layer order within
# the group, then conv/ssm per layer -- the same order the adapter's
# MemoryObj pack/unpack uses.
_LMC_BOUNDARY_REGISTRY: dict[str, list[tuple[int, torch.Tensor]]] = {}
_LMC_BOUNDARY_PER_REQ = int(os.environ.get("LMC_BOUNDARY_KEEP", "8"))
# Leak bound for registry entries the finishing store never consumed
# (aborted requests, engine shutdown): retire oldest request first.
_LMC_BOUNDARY_GLOBAL_CAP = int(os.environ.get("LMC_BOUNDARY_CAP", "48"))
# Retrieved boundary states parked by the adapter's retrieve (CPU bytes);
# preprocess_mamba consumes-and-pops the entry for its request. Value:
# (boundary token count B, flat uint8 CPU tensor of all layer pages).
_LMC_SEED_OVERRIDE: dict[str, tuple[int, torch.Tensor]] = {}

# llm-scaler v135 (MambaDType float16 support): fp16 pools can hold inf
# (a pass whose trajectory exceeds 65504 materializes inf, and the next
# pass consuming that seed NaNs the lineage). Repair carried seed pages
# in place at PRE-COPY staging: nan->0, +-inf->+-65504. No-op for finite
# pages; skipped for non-float16 pools. Our posture runs
# --mamba-ssm-cache-dtype float16, so this is live. Disable via
# VLLM_MAMBA_FP16_SEED_REPAIR=0.
_FP16_SEED_REPAIR = os.environ.get("VLLM_MAMBA_FP16_SEED_REPAIR", "1") != "0"


def _lmc_snap_boundary(
    req_id,
    aligned_tokens,
    dest_block_idx,
    req_state,
    kv_cache_config,
    mamba_group_ids,
    forward_context,
) -> None:
    """D2H-freeze the boundary state pages (group-major, layer, conv/ssm)
    into one flat uint8 CPU tensor. Failures degrade to a missing boundary
    (later lookups recompute), never to a wrong one."""
    try:
        pages = []
        pinfo = []
        total = 0
        for gid in mamba_group_ids:
            bid = req_state.block_ids[gid][dest_block_idx]
            for layer_name in kv_cache_config.kv_cache_groups[gid].layer_names:
                for st in forward_context[layer_name].kv_cache:
                    pg = st[bid].reshape(-1).view(torch.uint8).reshape(-1)
                    pages.append(pg)
                    pinfo.append((total, pg.numel(), st.dtype))
                    total += pg.numel()
        if total == 0:
            return
        flat = torch.empty(total, dtype=torch.uint8, device=pages[0].device)
        off = 0
        for pg in pages:
            flat[off:off + pg.numel()].copy_(pg)
            off += pg.numel()
        cpu = flat.cpu()  # one D2H, synced after the materializing copies
        # llm-scaler telemetry: non-finite scan of the just-frozen boundary
        # (the write-side truth for NaN-poison diagnosis; LMC_BOUND_STATS=0
        # skips, LMC_BOUND_MD5=0 drops only the md5).
        if os.environ.get("LMC_BOUND_STATS", "1") != "0":
            try:
                import hashlib as _hl

                nf = 0
                am = 0.0
                for off2, nb2, dt2 in pinfo:
                    if dt2 == torch.uint8:
                        continue
                    f2 = cpu[off2:off2 + nb2].view(dt2).float()
                    nf += int((~torch.isfinite(f2)).sum().item())
                    if f2.numel():
                        am = max(am, float(f2.abs().max().item()))
                hh = "-"
                if os.environ.get("LMC_BOUND_MD5", "1") != "0":
                    hh = _hl.md5(cpu.numpy().tobytes()).hexdigest()[:8]
                logger.debug(
                    "LMCBOUND req=%s B=%d dest=%d %s/n%d/a%.2e/c%d",
                    req_id,
                    aligned_tokens,
                    dest_block_idx,
                    hh,
                    nf,
                    am,
                    total,
                )
                if nf > 0:
                    logger.warning(
                        "LMCBOUND NONFINITE req=%s B=%d nf=%d am=%.2e "
                        "(v138c sanitize-carry REVAL trigger)",
                        req_id,
                        aligned_tokens,
                        nf,
                        am,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("LMCBOUND ERR req=%s %s", req_id, repr(e)[:120])
        lst = _LMC_BOUNDARY_REGISTRY.setdefault(req_id, [])
        if lst and lst[-1][0] == aligned_tokens:
            lst[-1] = (aligned_tokens, cpu)
        else:
            lst.append((aligned_tokens, cpu))
        while len(lst) > _LMC_BOUNDARY_PER_REQ:
            lst.pop(0)
        if os.environ.get("ARC_SNAP_PROBE"):  # r31 diagnostic (silent off)
            print(
                f"[r31probe snap req={req_id} B={aligned_tokens} "
                f"bytes={total} reg_len={len(lst)} "
                f"global={len(_LMC_BOUNDARY_REGISTRY)}]",
                flush=True,
            )
        # Global FIFO bound: the finishing request's store pops its own
        # entry at wait_for_save; anything the store never consumed is
        # retired oldest-first here (a dropped entry only turns a later
        # hit into a clean recompute).
        while len(_LMC_BOUNDARY_REGISTRY) > _LMC_BOUNDARY_GLOBAL_CAP:
            oldest = next(iter(_LMC_BOUNDARY_REGISTRY))
            if oldest == req_id:
                break
            _LMC_BOUNDARY_REGISTRY.pop(oldest)
    except Exception as e:  # noqa: BLE001
        logger.warning("LMC boundary snapshot failed req=%s: %s", req_id, repr(e)[:120])


def _lmc_seed_write(
    req_id,
    seed_tokens,
    prev_state_idx,
    block_size,
    req_state,
    kv_cache_config,
    mamba_group_ids,
    forward_context,
    seed_bytes,
) -> None:
    """Write a retrieved boundary state (seed) into the pages the resume
    will carry FROM. The joint lookup booked exactly B = block_size*j
    tokens as computed, so prev_state_idx = B/block_size - 1 IS the
    boundary's dest block. The normal PRE-COPY then carries the seed into
    the running-state block and the v135 fp16 repair covers this write."""
    want_idx = seed_tokens // block_size - 1
    if want_idx != prev_state_idx or prev_state_idx < 0:
        logger.warning(
            "LMC seed mismatch req=%s B=%d want_idx=%d prev=%d; dropping",
            req_id,
            seed_tokens,
            want_idx,
            prev_state_idx,
        )
        return
    off = 0
    written = 0
    for gid in mamba_group_ids:
        bid = req_state.block_ids[gid][prev_state_idx]
        for layer_name in kv_cache_config.kv_cache_groups[gid].layer_names:
            for st in forward_context[layer_name].kv_cache:
                pg = st[bid].reshape(-1).view(torch.uint8).reshape(-1)
                pg.copy_(seed_bytes[off:off + pg.numel()].to(pg.device, non_blocking=False))
                off += pg.numel()
                written += 1
    logger.debug(
        "LMC seed write req=%s B=%d page_idx=%d pages=%d bytes=%d",
        req_id,
        seed_tokens,
        prev_state_idx,
        written,
        off,
    )


def _arc_p41_preprocess_mamba(
    scheduler_output,
    kv_cache_config,
    cache_config,
    mamba_state_idx,
    input_batch,
    requests,
    forward_context,
    mamba_state_copy_funcs,
    copy_bufs,
    align_ctx=None,
):
    """vllm-0.30 preprocess_mamba + the v136 LMCache glue blocks.

    Upstream body verbatim except the four marked insertions:
    (1) seed consume/write at loop top; (2) v135 fp16 repair before a
    carry stages; (3) boundary-row collection inside the carry branch;
    (4) the snapshot loop after the copies (stream-ordered before the
    D2H sync). Late-binds everything mutable through the mamba_utils
    module so swaps stay coherent."""
    import vllm.v1.worker.mamba_utils as _mu

    fused = _mu._resolve_fused_precopy(align_ctx)

    if os.environ.get("ARC_SNAP_PROBE"):  # r31 diagnostic (silent off)
        _n = globals().get("_arc_p41_calls", 0) + 1
        globals()["_arc_p41_calls"] = _n
        if _n <= 6:
            print(
                f"[r31probe ppm#{_n} reqs={len(input_batch.req_ids)} "
                f"fused={fused is not None} align_ctx={align_ctx is not None}]",
                flush=True,
            )
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    _mu.cleanup_mamba_state_idx(scheduler_output, mamba_state_idx)

    copy_bufs.offset = 0
    num_reqs = len(input_batch.req_ids)
    boundary_rows = []

    if fused is not None:
        if num_reqs == 0:
            return
        if not fused.ctx.is_initialized:
            fused.ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                mamba_state_copy_funcs,
                [input_batch.block_table[gid].get_device_tensor(num_reqs) for gid in fused.ctx.mamba_group_ids],
            )

        fused.src_col.np[:num_reqs] = -1
        fused.token_bias.np[:num_reqs] = 0

    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        # --- (1) v136 load half: consume the retrieved boundary seed the
        # adapter's retrieve parked; write it into the carry-from page
        # BEFORE any PRE-COPY below so the normal machinery carries it.
        _lmc_seed = _mu._LMC_SEED_OVERRIDE.pop(req_id, None)
        if _lmc_seed is not None:
            _lmc_seed_write(
                req_id,
                _lmc_seed[0],
                prev_state_idx,
                block_size,
                req_state,
                kv_cache_config,
                mamba_group_ids,
                forward_context,
                _lmc_seed[1],
            )

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks = (_mu.cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size) +
                      num_speculative_blocks)
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if fused is not None:
            fused.state_idx.np[i] = curr_state_idx

        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            # --- (2) v135: repair the carried seed pages in place BEFORE
            # staging (covers the (1) seed write and any resume carry).
            if _FP16_SEED_REPAIR:
                try:
                    for gid in mamba_group_ids:
                        _lname = kv_cache_config.kv_cache_groups[gid].layer_names[0]
                        _bids = req_state.block_ids[gid]
                        for _st in forward_context[_lname].kv_cache:
                            if _st.dtype == torch.float16:
                                _pg = _st[_bids[prev_state_idx]]
                                _pg.copy_(torch.nan_to_num(
                                    _pg,
                                    nan=0.0,
                                    posinf=65504.0,
                                    neginf=-65504.0,
                                ))
                except Exception as e:  # noqa: BLE001
                    logger.warning("v135 fp16 seed repair failed: %s", repr(e)[:120])
            accept_token_bias = int(input_batch.num_accepted_tokens_cpu[i]) - 1
            if fused is not None:
                assert accept_token_bias >= 0
                fused.src_col.np[i] = prev_state_idx
                fused.token_bias.np[i] = accept_token_bias
            else:
                _mu.collect_mamba_copy_meta(
                    copy_bufs,
                    kv_cache_config,
                    mamba_state_copy_funcs,
                    mamba_group_ids,
                    prev_state_idx,
                    curr_state_idx,
                    accept_token_bias,
                    req_state,
                    forward_context,
                )
            input_batch.num_accepted_tokens_cpu[i] = 1
            # --- (3) v136 (0.26 re-anchor): the carry firing means the
            # running block advanced past the deepest full boundary, so
            # that boundary block's bytes are final (block j holds the
            # state after (j+1)*block_size tokens once written; the fused
            # kernels of past steps are stream-ordered before the (4)
            # D2H sync, and the carry reads prev / writes curr without
            # touching boundary blocks). Same-B re-snaps replace.
            _lmc_b = (req_state.num_computed_tokens // block_size) * block_size
            _lmc_b_idx = req_state.num_computed_tokens // block_size - 1
            if _lmc_b_idx >= 0 and _lmc_b > 0:
                boundary_rows.append((req_id, _lmc_b, _lmc_b_idx, req_state))
                if os.environ.get("ARC_SNAP_PROBE"):  # r31 diagnostic
                    print(
                        f"[r31probe carry req={req_id} ncc="
                        f"{req_state.num_computed_tokens} B={_lmc_b} "
                        f"idx={_lmc_b_idx} prev={prev_state_idx} "
                        f"curr={curr_state_idx} bs={block_size}]",
                        flush=True,
                    )

    if fused is not None:
        fused.state_idx.copy_to_gpu(num_reqs)
        fused.src_col.copy_to_gpu(num_reqs)
        fused.token_bias.copy_to_gpu(num_reqs)
        fused.ctx.run_fused_precopy(
            num_reqs=num_reqs,
            state_idx_gpu=fused.state_idx.gpu,
            src_col_gpu=fused.src_col.gpu,
            token_bias_gpu=fused.token_bias.gpu,
            idx_mapping=None,
        )
    else:
        _mu.do_mamba_copy_block(copy_bufs)

    # --- (4) v136: freeze the boundaries that advanced this step, after
    # the copies (the D2H inside _lmc_snap_boundary is stream-ordered
    # after the precopy kernel above).
    for _rid, _aligned, _dest_idx, _rstate in boundary_rows:
        _lmc_snap_boundary(
            _rid,
            _aligned,
            _dest_idx,
            _rstate,
            kv_cache_config,
            mamba_group_ids,
            forward_context,
        )


def _install(mod) -> None:
    """Attach the registry symbols and swap preprocess_mamba. Idempotent."""
    mod._LMC_BOUNDARY_REGISTRY = _LMC_BOUNDARY_REGISTRY
    mod._LMC_BOUNDARY_PER_REQ = _LMC_BOUNDARY_PER_REQ
    mod._LMC_BOUNDARY_GLOBAL_CAP = _LMC_BOUNDARY_GLOBAL_CAP
    mod._LMC_SEED_OVERRIDE = _LMC_SEED_OVERRIDE
    mod._lmc_snap_boundary = _lmc_snap_boundary
    mod._lmc_seed_write = _lmc_seed_write
    if getattr(mod.preprocess_mamba, "_arc_p41", False):
        return
    _orig_cleanup = mod.cleanup_mamba_state_idx

    def _cleanup_with_seeds(scheduler_output, mamba_state_idx):
        # v136: a preempted/resumed request re-plans from scratch (fresh
        # joint lookup); the one-shot seed is dropped with it. Boundary
        # registry entries are ONLY popped by the finishing store.
        _orig_cleanup(scheduler_output, mamba_state_idx)
        try:
            finished = scheduler_output.finished_req_ids
            preempted = scheduler_output.preempted_req_ids or set()
            resumed = scheduler_output.scheduled_cached_reqs.resumed_req_ids
            for req_id in (*finished, *preempted, *resumed):
                _LMC_SEED_OVERRIDE.pop(req_id, None)
        except Exception as e:  # noqa: BLE001
            logger.warning("p41 cleanup seed pop failed: %s", repr(e)[:120])

    _arc_p41_preprocess_mamba._arc_p41 = True
    mod.preprocess_mamba = _arc_p41_preprocess_mamba
    mod.cleanup_mamba_state_idx = _cleanup_with_seeds

    if os.environ.get("ARC_SNAP_PROBE"):  # r31 diagnostic (silent off)
        # Which specdec-path mamba_utils entry points actually run under
        # MTP k4 + align (the preprocess_mamba carry branch is dead there)?
        for _fname in (
                "preprocess_mamba_all_specdec",
                "postprocess_mamba_align_gpu",
                "stage_postprocess_inputs_to_gpu",
                "postprocess_mamba_all",
        ):
            _orig = getattr(mod, _fname, None)
            if _orig is None:
                continue

            def _mk(_name, _fn):

                def _w(*a, **k):
                    _r = _fn(*a, **k)
                    _c = globals().setdefault("_arc_p41_probe_calls", {})
                    _c[_name] = _c.get(_name, 0) + 1
                    if _c[_name] <= 4:
                        print(f"[r31probe {_name}#{_c[_name]}]", flush=True)
                    return _r

                return _w

            setattr(mod, _fname, _mk(_fname, _orig))

    logger.info(
        "arc-p41 mamba LMCache glue installed (registry caps %d/%d, "
        "fp16_seed_repair=%s)",
        _LMC_BOUNDARY_PER_REQ,
        _LMC_BOUNDARY_GLOBAL_CAP,
        _FP16_SEED_REPAIR,
    )
