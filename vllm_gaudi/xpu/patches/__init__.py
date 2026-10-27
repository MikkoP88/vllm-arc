# SPDX-License-Identifier: Apache-2.0
"""In-place patches over core vLLM for XPU graph safety (plan Appendix C).

Each module here patches core classes at runtime and is installed from the
platform hooks (``vllm_gaudi.xpu.__init__:register`` and the
``ArcXpuPlatform`` methods) — idempotent, per-process, and visible in serve
logs under the ``vllm.platforms.arc`` logger.
"""
