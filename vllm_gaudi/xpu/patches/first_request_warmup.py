# SPDX-License-Identifier: Apache-2.0
"""First-request warmup interlock for Arc XPU TP2 (defect-b fix, lane W).

Mechanism (r16, results/2026-10-28-p2-graph-p12b-r16-kwcapture): with XPU
graphs live, vllm 0.30 performs decode-graph capture warmup at boot and
reports healthy WITHOUT any real-size eager collective afterwards (no
warmup generate in 0.30). The first real request's embedding allreduce —
the first EAGER large AR — then leaves the device-default stream's SYCL
queue in command-graph recording state with no capture open (banked
bracket: AR entry (832,5120) pre_rec=False -> AR exit pre_rec=True, both
ranks); every later step 500s until the integrity guard kills the engine.
A first request small enough (<= ~512 tok / < 8 MB AR) initializes the
path cleanly and permanently vaccinates the engine for later large
requests (controlled pair, same posture boots: large-as-first flips 2/2
+ 1 with SYCL_CACHE_PERSISTENT=0; small-as-first 200 + the same large
request afterwards 200 with pre_rec=False).

Fix: interlock the FIRST ``AsyncLLM.generate`` — before it runs, push two
internal warmup generations through the same engine (full scheduler +
runner path, which the standalone 2-proc probe proved is required): a
small stage (128 tok, comfortably inside the observed-clean region) and
a max-batch stage (max_num_batched_tokens prompt, chunk 1 = the largest
possible AR). If the wedge ever strikes the warmup itself, the failure
lands at boot/first-traffic time through the existing fail-loud integrity
law — never silently inside user traffic.

Kill-switch for A/B legs: ``VLLM_ARC_WARMUP_DISABLE=1`` (the control leg
with it set reproduces the flip — that is the patch's own counterfactual
evidence, r17).
"""
from __future__ import annotations

import os
import time

_installed = False
_warmed = False

_WARMUP_SMALL_CHARS = 640  # ~128 tokens: inside the clean region
_WARMUP_MAXBATCH_FALLBACK = 8192  # max_num_batched_tokens default


def _max_batched_tokens(engine) -> int:  # noqa: ANN001 — engine is AsyncLLM
    try:
        return int(engine.vllm_config.scheduler_config.max_num_batched_tokens)
    except Exception:  # noqa: BLE001 — fallback is the 0.30 default
        return _WARMUP_MAXBATCH_FALLBACK


async def _run_stage(engine, orig_generate, name: str, text: str, max_tokens: int) -> None:
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True, detokenize=False)
    t0 = time.monotonic()
    n = 0
    async for _ in orig_generate(engine, text, params, f"arc-warmup-{name}"):
        n += 1
    from vllm.logger import init_logger
    init_logger("vllm.platforms.arc").info(
        "ArcXpuPlatform: first-request warmup stage %s done in %.1fs "
        "(%d outputs, %d prompt chars)", name,
        time.monotonic() - t0, n, len(text))


async def _run_warmup(engine, orig_generate) -> None:
    small = ("lorem ipsum dolor sit amet consectetur adipiscing elit "
             "sed do eiusmod tempor incididunt " * 8)[:_WARMUP_SMALL_CHARS]
    await _run_stage(engine, orig_generate, "small", small, 2)
    # "lorem ipsum dolor sit amet " tokenizes to ~6 tokens per repetition
    # (r17a measured: x2 max_num_batched_tokens reps = ~12x overshoot,
    # 70 s warmup). Aim for ~2 full-size chunks — the largest AR, twice.
    reps = max(2, _max_batched_tokens(engine) // 5 + 1)
    big = "lorem ipsum dolor sit amet " * reps
    await _run_stage(engine, orig_generate, "max-batch", big, 2)


def _install() -> None:
    global _installed
    if _installed or os.environ.get("VLLM_ARC_WARMUP_DISABLE", "0") == "1":
        return
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception as e:  # noqa: BLE001 — deferred to the next platform hook
        from vllm.logger import init_logger
        init_logger("vllm.platforms.arc").warning(
            "ArcXpuPlatform: first-request warmup install deferred "
            "(AsyncLLM not importable yet): %r", e)
        return

    orig = AsyncLLM.generate
    if getattr(orig, "_arc_warmup", False):
        _installed = True
        return

    async def generate(self, prompt, sampling_params, request_id, **kw):
        global _warmed
        if not _warmed:
            _warmed = True  # set first: warmup drives orig directly
            from vllm.logger import init_logger
            log = init_logger("vllm.platforms.arc")
            log.info("ArcXpuPlatform: first generate (request %s) — "
                     "running defect-b warmup interlock first", request_id)
            t0 = time.monotonic()
            try:
                await _run_warmup(self, orig)
            except Exception as e:  # noqa: BLE001 — fail-loud law
                log.error(
                    "ArcXpuPlatform: warmup FAILED after %.1fs — "
                    "integrity law applies, refusing to serve on a "
                    "cold engine: %r",
                    time.monotonic() - t0, e)
                raise
            log.info("ArcXpuPlatform: warmup complete in %.1fs — engine "
                     "hot, serving request %s",
                     time.monotonic() - t0, request_id)
        async for out in orig(self, prompt, sampling_params, request_id, **kw):
            yield out

    generate._arc_warmup = True
    AsyncLLM.generate = generate
    _installed = True
    from vllm.logger import init_logger
    init_logger("vllm.platforms.arc").info("ArcXpuPlatform: first-request warmup interlock installed "
                                           "(defect-b lane W, r16 evidence)")
