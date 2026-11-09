# SPDX-License-Identifier: Apache-2.0
"""Skinny-M decode GEMM reroute (p633) -- decode-sized linear-layer
calls go to the arc6 custom kernels within per-lane M caps (evidence:
r126 windows 8b/8c/9), everything else keeps the current route.

Root cause this addresses (r123 kernel matrix, banked under
``results/2026-11-08-p631-mxfp8-lanes/``): at decode Ms (1..8, batch x
spec tokens) every stock oneDNN weight-only route is latency-bound far
above the pure-weights bytes floor -- qwen.up K=5120 N=6912/rank at
608 GB/s floors at 58 us (f8) / 29 us (f4), while the measured medians
are w16pt 179 us, w16pc 218 us, m816 327 us, g16 310 us. The oneDNN
joint-quant primitives lower to M-tiled gemmstone kernels sized for
large M; at M <= 8 the primitive pays full tile-fill traffic for an M
dimension that is mostly padding. The arc6 wheel adds a GEMV-regime
kernel family (``csrc/xpu/skinny_gemm/`` in
``external_patches/vllm-xpu-kernels/arc6_skinny_v017bp/``): one
work-item streams TN weight columns K-contiguously, dequantizes in
registers (f8 bit-identity exponent re-bias, e2m1 value set, e8m0
exp2 group scales), accumulates fp32, and reduces split-K in fixed
order (deterministic). Same op signatures and PWAL layouts as the
routes it replaces, so this wrapper only swaps the call.

Lanes (detected from the layer, not the route that produced it):
  * weight f8 + fp32/absent scale  -> ``skinny_fp8_gemm_w8a16``
    (checkpoint-fp8 champion classes w16pt/w16pc)
  * weight f8 + e8m0 group scale   -> ``skinny_fp8_gemm_w8a16_mxfp8``
    (online MXFP8; identical tensor forms as the p631 reroute)
  * packed f4 weight + e8m0 scale  -> ``skinny_fp4_gemm_w4a16``
    (online MXFP4; identical tensor forms as the p630 reroute)

This module chains AFTER mxfp4_w4a16/mxfp8_w8a16 (it is last in
``_XPU_PATCH_MODULES``): it captures whatever ``apply_weights`` is
installed at import time and only intercepts calls under each lane's
M cap, so larger-M behavior is bit-identical and each lane A/B-runs via ``ARC_SKINNY_M``.

Gate: ``ARC_SKINNY_M`` -- unset/``off`` keeps the current routes (the
certified default: r127 e2e falsified the scalar reroute -- see below),
``on`` installs the arc6 scalar lanes, ``jm`` installs the arc7
joint-matrix lanes (P6.34; repack-cached per weight, M<=8), any other
value WARNs and keeps. On wheels without the ops (pre-arc6/arc7) the
install WARNs and keeps -- a silent non-install is indistinguishable
from the deferred patch (visibility law). A routed op that throws at
runtime WARNs once and that lane falls back permanently.

Why OFF is the default (r127 e2e battery, 2026-11-09, results/
2026-11-08-p633-skinny-m/): the serve decode path replays FULL_DECODE_ONLY
XPU graphs, and under graph replay stock oneDNN runs its tensor-core
gemmstone paths at 420-680 GB/s device-side (r127f 50-call amortized
graph probe: gemma m816 M=2 stock 0.043-0.113 ms/call) -- the ~197 GB/s
"stock level" that motivated the kernel was an eager-launch-bound
artifact (host dispatch ~0.15-0.3 ms/call caps every op eagerly; the
round-interleaved kbench compared host rates, not device compute). The
skinny scalar-FMA streaming design tops at ~300 GB/s, so recorded into
the replayed decode graph it regressed gemma solo decode -38% (m816,
17.1 vs 27.7 tps) / -42% (g16, 23.3 vs 40.5 tps) while every over-cap
path (qwen MTP verify M=4, agg8 M=8, prefill) stayed exactly on ref.
The size-1 decode graph pads the token dim to 2, so even M=1-capped
routing executes as M=2 work inside the graph. The kernels stay
correct (r126 GATE-1) and available opt-in for eager/host-bound use.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.platforms.arc")

_MODE_ENV = "ARC_SKINNY_M"
# Per-lane route caps (r126 windows 8b/8c/9, round-interleaved paired
# medians; kernel itself accepts M <= 8 -- csrc instantiation bound):
#   * w16 (checkpoint-fp8, the qwen champion lane): M=1 only. M=1 wins
#     10/12 w16pt+w16pc cells (0.83-0.91x; qwen.o w16pt 1.06 and gemma.o
#     w16pt 2.17 -- stock's scalar-scale gemv special -- are the residue);
#     at M=2 w16pc drops to 2/6 and w16pt 1/6, M>=4 loses everywhere.
#   * m816 (online MXFP8, the gemma champion lane): M<=2. M=1 4/6
#     (losses 1.03 qwen.up/qwen.down -- qwen does not use this lane),
#     M=2 5/6 (worst 1.045), M=4 1.02-1.31 loses the spec-verify decode.
#   * g16 (online MXFP4): M<=2. M=1 and M=2 both 6/6 wins (0.81-0.93x);
#     M=4 1.03-1.40, M=8 up to 4.5x (gemma.down g16 M=8: the scalar-FMA
#     ALU wall -- decode ops per weight byte scale with M; stock's
#     tensor-core path wins the M>=4 verify batches).
_LANE_M_MAX = {"w16": 1, "m816": 2, "g16": 2}

# JM (arc7 joint-matrix/DPAS family, P6.34) lane caps: the kernel
# instantiates M <= 8 (csrc bound); over-cap M defers to the wrapped
# route exactly as the scalar lanes do. Route contract differs from the
# scalar lanes: the jm ops take COLUMN-REPACKED weight tiles
# ([N/16, K/16, 256] f8 / [N/16, K/16, 128] f4) and N-padded scales, and
# return [M, N_pad] (real columns sliced). The repack + scale-pad runs
# ONCE per weight (cache on the layer, keyed by data_ptr -- weights are
# static after load), so XPU-graph capture records only the jm kernel +
# a view; the one-time eager repack lands in the pre-capture warmup.
_JM_LANE_M_MAX = {"w16": 8, "m816": 8, "g16": 8}

# lane -> (routed once?, fallen back?) -- first-route marker telemetry +
# permanent per-lane fallback on any runtime failure
_state: dict[str, tuple[bool, bool]] = {
    "w16": (False, False),
    "m816": (False, False),
    "g16": (False, False),
}


def _jm_pad_n(t: torch.Tensor, npad: int, dim: int = -1) -> torch.Tensor:
    """Pad tensor dim to npad (no-copy when already padded). Weights pad
    dim 0 ([N, K] -> N_pad rows), scales/bias pad the last dim ([.., N]
    -> [.., N_pad]). r128m caught the i == dim comprehension form: i is
    0-based and never equals -1, so dim=-1 silently appended a full-width
    zero block instead of the pad delta."""
    dim = dim if dim >= 0 else t.dim() + dim
    if t.shape[dim] == npad:
        return t
    pad = torch.zeros(
        [npad - t.shape[dim] if i == dim else s
         for i, s in enumerate(t.shape)],
        dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim).contiguous()


def _jm_repack_f8(w8: torch.Tensor) -> torch.Tensor:
    """[N, K] u8 -> column tiles [N/16, K/16, 256]: tile byte (c, k) =
    16c + k (r128e2 B layout -- lane t's fragment = column n0+t's 16
    consecutive k codes; the jm kernel's repack contract, r128b
    repack_selftest)."""
    n, k = w8.shape
    nt, kt = (n + 15) // 16, k // 16
    w8 = _jm_pad_n(w8, nt * 16, 0)
    return (w8.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
            .contiguous().view(nt, kt, 256))


def _jm_repack_f4(w4: torch.Tensor) -> torch.Tensor:
    """Packed nibbles [N, K/2] u8 -> column tiles [N/16, K/16, 128]:
    nibble (c, k) at byte 8c + (k>>1), slot k&1, LO nibble = even k
    (k-ascending interleave -- the same contract nib_vals/repack_f4
    verify in r128b; a block-concat layout here silently swaps hi/lo
    halves of every 2-k pair)."""
    n, kh = w4.shape
    k = kh * 2
    nt, kt = (n + 15) // 16, k // 16
    lo, hi = w4 & 0xF, w4 >> 4
    wv = torch.stack((lo, hi), dim=2).reshape(n, k)
    wv = _jm_pad_n(wv, nt * 16, 0)
    v = (wv.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
         .contiguous().view(nt, kt, 16, 8, 2))
    return (v[..., 0] | (v[..., 1] << 4)).view(nt, kt, 128)


def _jm_pack(layer: torch.nn.Module, lane: str) -> tuple | None:
    """One-time per-weight repack + scale pad (cache on the layer, keyed
    by weight data_ptr). Returns (b_tiles, b_scale, n_real) or None when
    the lane's tensor forms are unexpected."""
    w = layer.weight
    wsc = getattr(layer, "weight_scale", None)
    key = (lane, w.data_ptr(), w.shape)
    cached = getattr(layer, "_arc_jm_pack", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    n, k = w.shape
    npad = (n + 15) // 16 * 16
    if lane in ("w16", "m816"):
        b = _jm_repack_f8(w.view(torch.uint8))
        if lane == "w16":
            # per-tensor [1] or per-channel [1, N] (XPUW8A16FP8 forms)
            s = wsc
            if s is not None and s.numel() == n:
                s = _jm_pad_n(s.reshape(1, n) if s.dim() == 1 else s,
                              npad).contiguous()
        else:
            # layer scale [N, K/32] -> op form [K/32, N_pad]
            s = _jm_pad_n(wsc.t(), npad).contiguous()
    else:  # g16: packed [N, K/2], scale [N, K/32] -> [K/32, N_pad]
        b = _jm_repack_f4(w.view(torch.uint8) if w.dtype !=
                          torch.uint8 else w)
        s = _jm_pad_n(wsc.t(), npad).contiguous()
    pack = (b, s, n)
    layer._arc_jm_pack = (key, pack)
    return pack


def _flatten_m(x: torch.Tensor) -> int:
    m = 1
    for d in x.shape[:-1]:
        m *= d
    return m


def _try_jm(layer: torch.nn.Module, x: torch.Tensor,
            bias: torch.Tensor | None) -> torch.Tensor | None:
    """Route one call to the arc7 jm (joint-matrix/DPAS) kernels within
    the lane's M cap; None = use the wrapped route. Repack + scale-pad
    are cached per weight (_jm_pack) so graph capture records only the
    jm kernel and the [..., :n] slice view."""
    w = layer.weight
    wsc = getattr(layer, "weight_scale", None)
    ops = torch.ops._xpu_C

    if w.dtype in (torch.uint8, torch.float4_e2m1fn_x2):
        lane = "g16"
    elif w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        lane = ("m816" if wsc is not None
                and wsc.dtype == torch.float8_e8m0fnu else "w16")
    else:
        return None  # unsupported weight format
    if _flatten_m(x) > _JM_LANE_M_MAX[lane] or _state[lane][1]:
        return None
    try:
        b, s, n = _jm_pack(layer, lane)
        # the ops contract against N_pad (tiles*16): bias pads to match
        # (fp32 [N] -> [N_pad], zeros in pad columns); scales were padded
        # at pack time. Output slices back to the real N columns.
        npad = b.shape[0] * 16
        if bias is not None and bias.numel() == n and bias.dim() == 1:
            bias = _jm_pad_n(bias, npad)
        if lane == "w16":
            out = ops.jm_fp8_gemm_w8a16(x, b, s, bias)
        elif lane == "m816":
            out = ops.jm_fp8_gemm_w8a16_mxfp8(x, b, s, x.dtype, bias)
        else:
            out = ops.jm_fp4_gemm_w4a16(x, b, s, x.dtype, bias)
        out = out[..., :n]
    except Exception as e:  # noqa: BLE001 — visibility law: warn + fall back
        _state[lane] = (True, True)
        layer._arc_jm_pack = None  # stale cache can never be trusted
        logger.warning(
            "ArcXpuPlatform: jm %s route FAILED (%r) — lane falls back to "
            "the current route permanently", lane, e)
        return None
    if not _state[lane][0]:
        _state[lane] = (True, False)
        logger.info(
            "ArcXpuPlatform: jm route FIRST %s (M=%d, weight %s) — arc7 "
            "joint-matrix skinny kernel", lane, _flatten_m(x), w.dtype)
    return out


def _try_skinny(layer: torch.nn.Module, x: torch.Tensor,
                bias: torch.Tensor | None) -> torch.Tensor | None:
    """Route one call within the lane's M cap; None = use the wrapped
    route (M over cap, unsupported format, or permanent lane fallback)."""
    w = layer.weight
    wsc = getattr(layer, "weight_scale", None)
    ops = torch.ops._xpu_C

    if w.dtype in (torch.uint8, torch.float4_e2m1fn_x2):
        lane = "g16"
        if _flatten_m(x) > _LANE_M_MAX[lane] or _state[lane][1]:
            return None
        try:
            out = ops.skinny_fp4_gemm_w4a16(x, w, wsc, x.dtype, bias)
        except Exception as e:  # noqa: BLE001 — visibility law: warn + fall back
            _state[lane] = (True, True)
            logger.warning(
                "ArcXpuPlatform: skinny g16 route FAILED (%r) — lane falls "
                "back to the current w4a16 route permanently", e)
            return None
    elif w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if wsc is not None and wsc.dtype == torch.float8_e8m0fnu:
            lane = "m816"
            if _flatten_m(x) > _LANE_M_MAX[lane] or _state[lane][1]:
                return None
            try:
                # identical tensor forms as the p631 reroute: weight
                # [N, K] .t() view, scale [K//32, N] contiguous via .t()
                out = ops.skinny_fp8_gemm_w8a16_mxfp8(
                    x, w.t(), wsc.t(), x.dtype, bias)
            except Exception as e:  # noqa: BLE001
                _state[lane] = (True, True)
                logger.warning(
                    "ArcXpuPlatform: skinny m816 route FAILED (%r) — lane "
                    "falls back to the current w8a16 route permanently", e)
                return None
        else:
            lane = "w16"
            if _flatten_m(x) > _LANE_M_MAX[lane] or _state[lane][1]:
                return None
            try:
                # identical tensor forms as the stock champion caller
                # (XPUW8A16FP8LinearKernel.apply_weights)
                out = ops.skinny_fp8_gemm_w8a16(x, w, wsc, bias)
            except Exception as e:  # noqa: BLE001
                _state[lane] = (True, True)
                logger.warning(
                    "ArcXpuPlatform: skinny w16 route FAILED (%r) — lane "
                    "falls back to the stock w8a16 route permanently", e)
                return None
    else:
        return None  # unsupported weight format (e.g. bf16 lora legs)

    if not _state[lane][0]:
        _state[lane] = (True, False)
        logger.info(
            "ArcXpuPlatform: skinny-M route FIRST %s (M=%d, weight %s) — "
            "arc6 _xpu_C skinny kernel", lane, _flatten_m(x), w.dtype)
    return out


def _install(mod=None) -> None:  # noqa: ARG001 — mod mirrors the patch-module shape
    # Default OFF (r127 e2e falsification — module docstring): unset keeps
    # the stock routes; "on" is the scalar opt-in; "jm" is the arc7
    # joint-matrix family (P6.34).
    mode = os.environ.get(_MODE_ENV, "off").strip().lower()
    if mode == "off":
        logger.info(
            "ArcXpuPlatform: skinny-M reroute OFF (%s unset/off — stock "
            "routes at all Ms; the certified default after r127; set "
            "%s=on or =jm to opt in)", _MODE_ENV, _MODE_ENV)
        return
    if mode not in ("on", "jm"):
        logger.warning(
            "ArcXpuPlatform: unknown %s=%r — keeping current routes (valid: "
            "on, jm, off)", _MODE_ENV, mode)
        return
    jm_mode = mode == "jm"

    # ``vllm_xpu_kernels.__init__`` does NOT dlopen the extension — force
    # the load before probing or the probe false-negatives (arc4 lesson,
    # see mxfp8_w8a16).
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        logger.warning(
            "ArcXpuPlatform: skinny-M reroute SKIPPED — "
            "vllm_xpu_kernels._xpu_C not importable; current routes stay")
        return

    ops = torch.ops._xpu_C
    if jm_mode:
        probe = ("jm_fp8_gemm_w8a16", "jm_fp8_gemm_w8a16_mxfp8",
                 "jm_fp4_gemm_w4a16")
    else:
        probe = ("skinny_fp8_gemm_w8a16", "skinny_fp8_gemm_w8a16_mxfp8",
                 "skinny_fp4_gemm_w4a16")
    missing = [n for n in probe if not hasattr(ops, n)]
    if missing:
        logger.warning(
            "ArcXpuPlatform: skinny-M reroute SKIPPED — ops absent from the "
            "loaded wheel (pre-arc6): %s; current routes stay", missing)
        return

    classes: list[tuple[str, type]] = []
    try:
        from vllm.model_executor.kernels.linear.mxfp8.xpu import (
            XPUMxFp8LinearKernel, )
        classes.append(("mxfp8", XPUMxFp8LinearKernel))
    except ImportError as e:
        logger.warning(
            "ArcXpuPlatform: skinny-M m816 class not importable (%r) — "
            "lane keeps the p631 route", e)
    try:
        from vllm.model_executor.kernels.linear.mxfp4.xpu import (
            XPUMxFp4LinearKernel, )
        classes.append(("mxfp4", XPUMxFp4LinearKernel))
    except ImportError as e:
        logger.warning(
            "ArcXpuPlatform: skinny-M g16 class not importable (%r) — lane "
            "keeps the p630 route", e)
    try:
        from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
            XPUW8A16FP8LinearKernel, )
        classes.append(("w16champ", XPUW8A16FP8LinearKernel))
    except ImportError as e:
        logger.warning(
            "ArcXpuPlatform: skinny-M champion class not importable (%r) — "
            "w16pt/w16pc keep the stock route", e)

    installed = 0
    for name, cls in classes:
        if getattr(cls, "_arc_skinny_patched", False):
            installed += 1
            continue  # idempotent: platform hooks run twice (config + worker)
        # capture the CURRENT apply_weights — post-mxfp4/mxfp8 when those
        # patches installed first (this module is last in
        # _XPU_PATCH_MODULES); the wrapper defers to the active dispatcher
        # (_try_jm for ARC_SKINNY_M=jm, _try_skinny for =on), whose
        # per-lane M caps decide what routes where.
        orig = cls.apply_weights
        dispatch = _try_jm if jm_mode else _try_skinny

        def make_wrapper(orig_fn, dispatch_fn):
            def apply_weights(
                self,
                layer: torch.nn.Module,
                x: torch.Tensor,
                bias: torch.Tensor | None = None,
            ) -> torch.Tensor:
                out = dispatch_fn(layer, x, bias)
                if out is not None:
                    return out
                return orig_fn(self, layer, x, bias)

            return apply_weights

        cls.apply_weights = make_wrapper(orig, dispatch)
        cls._arc_skinny_patched = True
        installed += 1
        logger.info(
            "ArcXpuPlatform: skinny-M wrap installed on %s (mode %s, "
            "per-lane M caps %r, larger M unchanged; %s=off A/Bs)",
            name, mode, _JM_LANE_M_MAX if jm_mode else _LANE_M_MAX,
            _MODE_ENV)

    if installed == 0:
        logger.warning(
            "ArcXpuPlatform: skinny-M reroute installed on ZERO classes — "
            "check the import warnings above")
