"""arc p41b — LMCacheConnectorV1 HMA + lmcache-0.5.5 compat port
+ arc p42 — lookup-elide / lookup-memo (D v139 extyield2 port)
(vllm-side connector glue; installed via sitecustomize exec_module hook
on `vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector`).

p42 (r31, 2026-10-28): the r30 warm-hit gate landed at median 755.1 ms
vs the 750 ms bar with sha-verified full hits — the adder is the same
one D's probe29 closed (results/2026-10-21-probe29-lookupidle): the
per-call ZMQ lookup round trips (attn pair + v136 joint mamba pair,
1.6–3.5 calls/request), paid even when the answer is 0. Port of D's
extyield2 v139 guards onto the p41b wrapper override (the engine-side
attrs all verified in the C vendored adapter: _requests_priority :1338,
load_specs :1295, _lmc_async_pending :1311, async_loading :1139,
kv_role :1011, producer corner :2463): (1) LOOKUP-ELIDE — engine holds
a prefix AND residual <= VLLM_ARC_LOOKUP_ELIDE (default 0 = off, 0
disables) → skip both lookups, natural-yield LoadSpec, discard pending,
return 0 (byte-equivalent to the declined path minus the round trips;
cold faces never elide; saturation form rejected — misses
ncc=5120/5201). DEFAULT FLIPPED 4096→0 (r31 disposition, 2026-10-28):
elide was scoped when every warm answer was provably 0 (APC-only world)
and the round trips were pure loss; with p40d (torch-baseline KV
transfer → real attn stores) + p41c (live-site boundary snaps → real
joint loads) the same residual<=4096 faces now claim real bytes (the
8192 gate face: APC 7168, joint 8064 — residual 1024 would elide and
FORFEIT the load), so the guard only remains as an explicit opt-in;
(2) LOOKUP-MEMO — per-rid post-clamp hit memo replaying
the resolved depth through the same LoadSpec/pending tail (incl.
full-prompt-hit −1), async-mode only, kill-switch
VLLM_ARC_LOOKUP_MEMO=0.

TWO defects cured (r30 first LMCache leg, 2026-10-28), both with the C
champion (llm-scaler-exp v1.2.35 = vllm 0.30 + lmcache 0.5.5) as the
byte-preserved reference:

DEFECT 1 — HMA refusal. With the hybrid model under HMA (the certified
posture — D-line resume13 adjudication REJECTED the flag flip: "certified
memory-manager semantics vs CL-1 HMA-on bar"), the upstream-0.30 factory
(vllm/distributed/kv_transfer/kv_connector/factory.py:53-59) hard-refuses
the connector at initialize_from_config ("does not support HMA but HMA is
enabled"). Root cause: the factory resolves "LMCacheConnectorV1" to
VLLM'S OWN v1/lmcache_connector.py wrapper, and upstream 0.30's wrapper
lacks the SupportsHMA mixin. Fix (C deltas rebased onto pristine 0.30,
reference file md5 cee1a393f02b7b7f562ed70d63d8e4cf — the same reference
D byte-preserved into its 0.26 base at
/root/build/iexperimental/vllm/patches/prod/v1235-certified/boot026/):
  (a) SupportsHMA mixin + request_finished_all_groups (our 0.30
      scheduler dispatches on isinstance at scheduler.py:~2883; engine
      half = adapter request_finished);
  (b) v133 async-gate fix: report the REAL async flag from
      get_num_new_matched_tokens + forward `blocks` in
      update_state_after_alloc (hardcoded False made the scheduler run
      the forward in the SAME step as the external-hit discovery while
      the load ran in the NEXT step's pre_forward — zero pages consumed,
      ~70 ms late, byte-perfect but useless). Rides our 0.30 scheduler's
      native WAITING_FOR_REMOTE_KVS handshake; engine halves = adapter
      `_lmc_async_pending` + 3-arg update_state_after_alloc (verified in
      lmcache-0.5.5+arc1 adapter 2026-10-28);
  (c) KV-events guard: upstream calls engine.get_kv_events() unguarded;
      lmcache 0.5.5 has no KV-events API. Guarded form returns None.
NOT carried: upstream's bind_connector_metadata engine-bind shim is
KEPT (upstream-0.30-only surface; C's older base predates it).

DEFECT 2 — vendored lmcache_integration vs lmcache 0.5.5 skew. Under the
D-certified `use_native: true`, the connector instantiates VLLM'S
VENDED integration package (v1/lmcache_integration/vllm_v1_adapter.
LMCacheConnectorV1Impl — 2902 lines, the mamba-hybrid engine that
late-imports p41's _LMC_SEED_OVERRIDE/_LMC_BOUNDARY_REGISTRY at its
lines 1741/2049), but upstream 0.30's vendored copy is written against
NEWER lmcache main: its __init__ imports multi_process_adapter which
hard-imports CudaIPCWrapper from lmcache.v1.multiprocess.custom_types
(absent in 0.5.5) → ImportError at connector construction. Fix = C's
compat files, source-override (byte-preserved, exec'd in place of the
vendored module source at import time — no installed-tree edits):

  override file (this kit)         target module                          md5 (C pull)
  lmcint/__init__.py               ...v1.lmcache_integration               1f3e97d33cc8c0875c68138ff44b2e4b
  lmcint/vllm_v1_adapter.py        ...v1.lmcache_integration.vllm_v1_adapter  dcb23fbe0e8990a0b35d6367a8542f3f
  lmcint/utils.py                  ...v1.lmcache_integration.utils         b5300ee017ce5e3c833b0d438372066b

  C's __init__ drops multi_process_adapter ("requires lmcache main
  (CudaIPCWrapper) and is NOT importable here" — C's own header); the
  adapter/utils carry 0.5.5-compat imports (config_base validator,
  metadata.LMCacheMetadata alias, gpu_connectors from submodule) + HMA
  per-group block_ids + MambaPageConnector/_init_mamba_engine + the
  create_lmcache_metadata/is_false utils. Only other consumer of the
  dropped symbols is vllm's separate LMCacheMPConnector (a different
  connector name, not configured — imports lmcache-side anyway).

Engine-surface verification for (b): lmcache-0.5.5+arc1 adapter carries
_lmc_async_pending (:1311), get_num_new_matched_tokens (:2446),
3-arg update_state_after_alloc (:2580), request_finished (:2883).

Disable knob: VLLM_ARC_LMC_HMA=0 leaves the stock connector + vendored
integration untouched (boot then dies at the factory gate — knob exists
for A/B evidence only).
"""

from __future__ import annotations

import importlib.util as _iu
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only names (stock connector idiom :25-29; zero runtime)
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (LMCacheKVEvents)
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

_LMCINT = "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
_OVERRIDE_FILES = {
    _LMCINT: "lmcint/__init__.py",
    f"{_LMCINT}.vllm_v1_adapter": "lmcint/vllm_v1_adapter.py",
    f"{_LMCINT}.utils": "lmcint/utils.py",
}


class _SrcOverrideFinder:
    """Meta-path finder that swaps in the C-compat source for the vendored
    lmcache_integration modules at exec time (spec/loader otherwise stock,
    so package semantics, __path__, and create_module stay upstream)."""

    _marker = "_arc_p41b_src_override"

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _OVERRIDE_FILES:
            return None
        capdir = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(capdir, _OVERRIDE_FILES[fullname])
        if not os.path.exists(src):
            return None  # kit incomplete — stock source execs (and fails)
        sys.meta_path.remove(self)
        try:
            spec = _iu.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None

        def exec_module(module):
            # C-compat source INSTEAD of the stock file (the stock
            # __init__ would import multi_process_adapter -> ImportError).
            with open(src, "rb") as f:
                code = compile(f.read(), src, "exec")
            exec(code, module.__dict__)
            print(
                f"[{fullname.rsplit('.', 1)[-1]}@lmcint] src-override OK"
                f" (pid={os.getpid()})",
                flush=True,
            )

        spec.loader.exec_module = exec_module
        return spec


def _install_src_overrides() -> None:
    for f in sys.meta_path:
        if getattr(f, "_marker", None) == _SrcOverrideFinder._marker:
            return  # idempotent across sitecustomize + re-imports
    sys.meta_path.insert(0, _SrcOverrideFinder())


def _install(mod) -> None:
    _install_src_overrides()

    _base = getattr(mod, "LMCacheConnectorV1", None)
    if _base is None or getattr(_base, "_arc_p41b", False):
        return  # already patched (or unexpected module shape) — idempotent
    if os.environ.get("VLLM_ARC_LMC_HMA", "1") == "0":
        return

    from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA

    _logger = getattr(mod, "logger", None)

    class LMCacheConnectorV1(_base, SupportsHMA):
        """arc p41b: HMA-capable LMCacheConnectorV1 (C v1.2.35 deltas)
        + arc p42: lookup-elide / lookup-memo (D v139 extyield2 port)."""

        _arc_p41b = True

        def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
            """
            Get number of new tokens that can be loaded from the
            external KV cache beyond the num_computed_tokens.

            llm-scaler v133 (async-gate fix, 2026-10-09): report the real
            async flag. The hardcoded False made the scheduler schedule the
            forward in the SAME step as the external-hit discovery, while the
            worker-side load only ran in the NEXT step's pre_forward — the
            forward consumed zero pages and the load landed ~70ms late
            (byte-perfect but useless). Gated requests now take the
            WAITING_FOR_REMOTE_KVS handshake: a dedicated no-forward step
            performs the load, get_finished reports completion, and only then
            is the forward scheduled.

            arc p42 lookup-elide (D v139 extyield2 port; probe29 law:
            the flat warm/t2 adder is the PER-CALL ZMQ LOOKUP ROUND
            TRIPS — attn pair + v136 joint mamba pair, 1.6–3.5
            calls/request — paid even when the answer is 0 and nothing
            will be claimed; claim == idle == decline). When the engine
            already holds a prefix (num_computed_tokens > 0) and the
            residual (request.num_tokens - num_computed_tokens) is
            within VLLM_ARC_LOOKUP_ELIDE (default 0 = off; r31: real
            joint loads exist, elide would forfeit them — opt-in only),
            skip BOTH lookups entirely, write the natural-yield
            LoadSpec, and return 0 — byte-equivalent outcome to the
            declined path minus the round trips. Cold/evicted faces
            (num_computed_tokens == 0) never elide. Saturation-form
            elision was REJECTED upstream of this port (misses the
            ncc=5120/5201 face); only the residual form catches it.

            arc p42 lookup-memo (same D port; kill-switch
            VLLM_ARC_LOOKUP_MEMO=0): per-request post-clamp hit memo —
            repeat scheduler calls replay the resolved depth through
            the same LoadSpec/pending tail (incl. the full-prompt-hit
            −1) without re-paying the round trips. Async-mode only;
            never returns more than the inner call resolved for the
            same depth.
            """
            eng = self._lmcache_engine

            # kv_producer corner: preserve inner behavior exactly (it
            # returns 0 immediately without lookups when producer reuse
            # is unsupported) — D's wrapper guards this before anything.
            if eng.kv_role == "kv_producer" and not hasattr(eng.lookup_client, "supports_producer_reuse"):
                num_tokens = eng.get_num_new_matched_tokens(request, num_computed_tokens)
                return num_tokens, False

            # inner bookkeeping the elided/replayed paths must keep
            eng._requests_priority[request.request_id] = request.priority

            try:
                _elide_tok = int(os.environ.get("VLLM_ARC_LOOKUP_ELIDE", "0"))
            except ValueError:
                _elide_tok = 0
            if (_elide_tok > 0 and num_computed_tokens > 0 and request.num_tokens - num_computed_tokens <= _elide_tok):
                from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter import (
                    LoadSpec, )

                if _logger is not None:
                    _logger.info(
                        "arc p42 lookup-elide: req=%s tot=%d engine=%d "
                        "residual=%d <= %d -> skip zmq lookups",
                        request.request_id,
                        request.num_tokens,
                        num_computed_tokens,
                        request.num_tokens - num_computed_tokens,
                        _elide_tok,
                    )
                eng.load_specs[request.request_id] = LoadSpec(
                    vllm_cached_tokens=num_computed_tokens,
                    lmcache_cached_tokens=num_computed_tokens,
                    can_load=False,
                )
                eng._lmc_async_pending.discard(request.request_id)
                return 0, False

            _memo_on = (os.environ.get("VLLM_ARC_LOOKUP_MEMO", "1") != "0" and getattr(eng, "async_loading", False))
            _memo = None
            if _memo_on:
                _memo = getattr(self, "_arc_p42_memo", None)
                if _memo is None:
                    _memo = self._arc_p42_memo = {}
                if len(_memo) > 1024:
                    _memo.clear()
                _hit = _memo.get(request.request_id)
                if _hit is not None:
                    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter import (
                        LoadSpec, )

                    _need = _hit - num_computed_tokens
                    if _hit >= request.num_tokens:
                        _need -= 1
                    if _logger is not None:
                        _logger.info(
                            "arc p42 memo-hit: req=%s hit=%d engine=%d need=%d",
                            request.request_id,
                            _hit,
                            num_computed_tokens,
                            _need,
                        )
                    eng.load_specs[request.request_id] = LoadSpec(
                        vllm_cached_tokens=num_computed_tokens,
                        lmcache_cached_tokens=_hit,
                        can_load=False,
                    )
                    if _need <= 0:
                        eng._lmc_async_pending.discard(request.request_id)
                        return 0, False
                    eng._lmc_async_pending.add(request.request_id)
                    num_tokens = _need
                    load_kv_async = True
                    return num_tokens, load_kv_async

            num_tokens = eng.get_num_new_matched_tokens(request, num_computed_tokens)
            if _memo_on and _memo is not None and num_tokens is not None:
                _memo[request.request_id] = num_tokens + num_computed_tokens
            load_kv_async = (num_tokens is not None and num_tokens > 0 and request.request_id in eng._lmc_async_pending)
            return num_tokens, load_kv_async

        def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int):
            """
            Update KVConnector state after block allocation.

            llm-scaler v133 (async-gate fix): forward the allocated blocks so
            the impl can build the load ReqMeta for the gated request (the
            live Request object has no block_ids; only the broadcast
            NewRequestData does).
            """
            self._lmcache_engine.update_state_after_alloc(request, num_external_tokens, blocks)

        def get_kv_connector_kv_cache_events(self) -> LMCacheKVEvents | None:
            """
            Get the KV connector kv cache events collected during the last
            interval (guarded: lmcache 0.5.5 has no KV-events API).
            """
            get_kv_events = getattr(self._lmcache_engine, "get_kv_events", None)
            if not callable(get_kv_events):
                # lmcache 0.5.5 has no KV-events API; nothing to report.
                return None
            events = get_kv_events()
            if not events:
                return None

            blocks = [
                mod.BlockStored(
                    block_hashes=e.block_hashes,
                    parent_block_hash=e.parent_block_hash,
                    token_ids=e.token_ids,
                    lora_id=e.lora_id,
                    block_size=e.block_size,
                    medium=e.medium,
                    lora_name=getattr(e, "lora_name", None),
                ) for e in events
            ]

            lmcache_kv_events = mod.LMCacheKVEvents(num_workers=1)
            lmcache_kv_events.add_events(blocks)
            return lmcache_kv_events

        def request_finished_all_groups(
            self,
            request: Request,
            block_ids: tuple[list[int], ...],
        ) -> tuple[bool, dict | None]:
            """
            HMA variant: block_ids is a per-kv-cache-group tuple. LMCache
            stores synchronously within the forward pass (wait_for_save), so
            blocks can be freed immediately; per-group ids are not needed
            here (the engine tracks allocations via the scheduler metadata
            path).
            """
            flat_ids = block_ids[0] if block_ids else []
            return self._lmcache_engine.request_finished(request, flat_ids)

    mod.LMCacheConnectorV1 = LMCacheConnectorV1
