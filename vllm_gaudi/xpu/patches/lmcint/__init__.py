# SPDX-License-Identifier: Apache-2.0
# llm-scaler: single-process LMCache entry for lmcache 0.5.5.
# multi_process_adapter requires lmcache main (CudaIPCWrapper) and is NOT importable here.
from . import vllm_v1_adapter

__all__ = ["vllm_v1_adapter"]
