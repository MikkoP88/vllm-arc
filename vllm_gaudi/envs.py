# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH: bool = True
    VLLM_HPU_FORCE_CHANNEL_FP8: bool = True
    VLLM_HPU_HETERO_KV_LAYOUT: bool = False
    VLLM_HPU_MULTI_MODEL_CONFIG: Optional[str] = None
    VLLM_NIXL_ABORT_REQUEST_TIMEOUT: float = 300.0
    VLLM_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
    VLLM_NIXL_SIDE_CHANNEL_PORT: int = 5600
    VLLM_HPU_NIXL_JOINT_KV: bool = False
    VLLM_HPU_NIXL_STAGING_SLOTS: int = 0
    VLLM_MINIMAX_M3_MOE_TOKEN_TILE: int = 512
    VLLM_MINIMAX_M3_MOE_DECODE_GATHER: bool = True
    VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS: int = 16
    VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY: bool = False
    VLLM_HPU_MOE_GATHER: bool = False
    VLLM_HPU_MOE_GATHER_RATIO: float = 0.4
    VLLM_HPU_MOE_GATHER_VERIFY: bool = False
    VLLM_COMPACT_GDN: bool = False
    # vllm-arc XPU lane (VLLM_ARC_* namespace; Intel Arc / Xe2 devices).
    # r45d: VLLM_ARC_ASYNC / VLLM_ARC_YIELD_TOKENS / VLLM_ARC_COMPACT_GDN
    # dropped — declared-but-never-consumed (no-unevidenced law, Appendix H
    # #6; audit r45 findings F1-F3). Re-add only WITH a consumer leg.
    VLLM_ARC_LOOKUP_ELIDE: bool = True
    VLLM_ARC_XG_COMPACT: bool = False
    VLLM_ARC_GMU_RELAX: float = 0.0

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition
environment_variables: dict[str, Callable[[], Any]] = {
    # Contiguous cache fetching to avoid using costly gather operation on
    # Gaudi3. This is only applicable to HPU contiguous cache. If set to true,
    # contiguous cache fetch will be used.
    "VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH":
    lambda: os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() in ("1", "true"),

    # Convert block fp8 to channel fp8 for HPU
    # If `QUANT_CONFIG` is set, this will be forced to false.
    "VLLM_HPU_FORCE_CHANNEL_FP8":
    lambda: os.environ.get("VLLM_HPU_FORCE_CHANNEL_FP8", "true").lower() in
    ("1", "true") and os.environ.get("QUANT_CONFIG", None) is None,

    # Enable prefill side kv_layout and block_size for heterogeneous run.
    "VLLM_HPU_HETERO_KV_LAYOUT":
    lambda: os.environ.get("VLLM_HPU_HETERO_KV_LAYOUT", "false").lower() in ("0", "false"),

    # Path to a YAML config file describing the multi-model setup
    # (model names, weights, tensor-parallel config, etc.).
    # When unset, multi-model mode is disabled.
    "VLLM_HPU_MULTI_MODEL_CONFIG":
    lambda: os.environ.get("VLLM_HPU_MULTI_MODEL_CONFIG", None),

    # Timeout in seconds for NIXL abort request handling
    "VLLM_NIXL_ABORT_REQUEST_TIMEOUT":
    lambda: float(os.environ.get("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", "300")),

    # NIXL side channel host and port for KV transfer
    "VLLM_NIXL_SIDE_CHANNEL_HOST":
    lambda: os.environ.get("VLLM_NIXL_SIDE_CHANNEL_HOST", "localhost"),
    "VLLM_NIXL_SIDE_CHANNEL_PORT":
    lambda: int(os.environ.get("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600")),

    # Enable joint-KV staging so an HPU prefill instance can serve a GPU
    # (FLASH_ATTN, blocks-first) decode instance over NIXL. When the remote
    # decode registers ONE joint [K|V] region per layer (block_len covering
    # both K and V, V at block_len // 2), the HPU must advertise the same
    # region model. Leave false for HPU<->HPU disagg (separate K/V regions).
    "VLLM_HPU_NIXL_JOINT_KV":
    lambda: os.environ.get("VLLM_HPU_NIXL_JOINT_KV", "false").lower() in ("1", "true"),

    # Number of joint-KV staging slots for heterogeneous (HPU prefill ->
    # GPU decode) NIXL transfer. Each slot holds one on-save block for all
    # layers laid out blocks-first [2, n_kv_heads, block_size_on_save,
    # head_size] so the GPU can read K and V from one contiguous region
    # (V at block_len // 2). 0 (default) auto-sizes to the full concurrent
    # workload: max_num_seqs * ceil(max_model_len / block_size_on_save), which
    # guarantees every schedulable request gets a reservation. Registration
    # fails fast if that pool does not fit device memory. Only set a smaller
    # value if you are certain peak concurrent transfer demand stays under it.
    "VLLM_HPU_NIXL_STAGING_SLOTS":
    lambda: int(os.environ.get("VLLM_HPU_NIXL_STAGING_SLOTS", "0")),

    # MiniMax-M3 SwiGLU-OAI expert execution tuning.
    "VLLM_MINIMAX_M3_MOE_TOKEN_TILE":
    lambda: int(os.environ.get("VLLM_MINIMAX_M3_MOE_TOKEN_TILE", "512")),
    "VLLM_MINIMAX_M3_MOE_DECODE_GATHER":
    lambda: os.environ.get("VLLM_MINIMAX_M3_MOE_DECODE_GATHER", "1").lower() in ("1", "true"),
    "VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS":
    lambda: int(os.environ.get("VLLM_MINIMAX_M3_MOE_GATHER_MAX_TOKENS", "16")),

    # Run multimodal warmup outside PT_COMPILE_ONLY_MODE for models with
    # data-dependent output shapes that must be materialized during warmup.
    "VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY":
    lambda: os.environ.get("VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY", "false").strip().lower() in ("1", "true"),

    # EXPERIMENTAL custom gathered-expert FP8 MoE combine (silu only).
    # Off by default; falls back to the stock Habana fused op when disabled or
    # when the number of gathered experts (min(local_experts, tokens*top_k))
    # would exceed this fraction of this rank's local_experts. Parameterizing on
    # the ratio to local_experts (rather than an absolute count) keeps the gate
    # sane across EP configs, which change local_experts.
    #
    # Default rationale (0.4): this optimization has only been observed to help
    # the Qwen 3.5 MoE family. A crossover sweep across Qwen3.5 35B/122B/397B at
    # several EP levels shows the custom-gather win/loss cutoff tracks
    # g/local_experts most closely, and it lands around this ratio.
    "VLLM_HPU_MOE_GATHER":
    lambda: os.environ.get("VLLM_HPU_MOE_GATHER", "0").lower() in ("1", "true"),
    "VLLM_HPU_MOE_GATHER_RATIO":
    lambda: float(os.environ.get("VLLM_HPU_MOE_GATHER_RATIO", "0.4")),
    "VLLM_HPU_MOE_GATHER_VERIFY":
    lambda: os.environ.get("VLLM_HPU_MOE_GATHER_VERIFY", "0").lower() in ("1", "true"),

    # Use the compact recurrent-state (conv/ssm) layout for gated delta net
    # models. The model runner auto-detects and sets this during init, so read
    # it lazily rather than caching it at import time.
    "VLLM_COMPACT_GDN":
    lambda: os.environ.get("VLLM_COMPACT_GDN", "0").strip().lower() in ("1", "true"),

    # --------------------- vllm-arc XPU lane (Intel Arc / Xe2) ------------
    # r45d: VLLM_ARC_ASYNC (p50 closed at P2 — core 0.30 auto-enables async
    # scheduling), VLLM_ARC_YIELD_TOKENS (p51 never built), and
    # VLLM_ARC_COMPACT_GDN (dead twin of the live VLLM_COMPACT_GDN above,
    # auto-set by the model runner) dropped — audit r45 findings F1-F3,
    # no-unevidenced law.

    # Skip the LMCache lookup when the engine prefix cache already covers
    # the request span (port of D champion's extyield2 / IEXP_LOOKUP_MEMO=1).
    # Consumed by p42.
    "VLLM_ARC_LOOKUP_ELIDE":
    lambda: os.environ.get("VLLM_ARC_LOOKUP_ELIDE", "true").strip().lower() in ("1", "true"),

    # Compact-JSON structured-output posture for the xgrammar backend (port
    # of the C/D V1227_XGCOMPACT boot-script edit). Consumed by p06.
    "VLLM_ARC_XG_COMPACT":
    lambda: os.environ.get("VLLM_ARC_XG_COMPACT", "false").strip().lower() in ("1", "true"),

    # gpu-memory-utilization conditional relax: when free device memory at
    # engine start is below the requested utilization, relax the utilization
    # to this fraction instead of aborting the boot (0 disables; 0.8 is the
    # sanctioned B70 rig posture — standing directive Appendix I #11).
    # Consumed by p05.
    "VLLM_ARC_GMU_RELAX":
    lambda: float(os.environ.get("VLLM_ARC_GMU_RELAX", "0")),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
