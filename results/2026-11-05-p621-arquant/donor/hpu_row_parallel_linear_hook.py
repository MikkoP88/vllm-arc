# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Union

import os

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm_gaudi.extension.runtime import get_config

try:
    from vllm_gaudi.ops.ar_quant import ar_quant_all_reduce as _ar_quant_fn
    _ar_quant = (_ar_quant_fn
                 if os.environ.get("IEXP_AR_QUANT", "0") == "1" else None)
except Exception:  # noqa: BLE001 - module optional, plain AR is correct
    _ar_quant = None

_registered = False


def register():
    """Conditionally register HPURowParallelLinear as OOT replacement.

...[AR call site]...
            bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
            output_parallel = self.quant_method.apply(self, input_parallel, bias_)

            if self.reduce_results and self.tp_size > 1:
                if _ar_quant is not None:
                    # W13-AR (IEXP_AR_QUANT=1): quantized-transport AR for
                    # prefill-class payloads (peer partial e4m3 over the
                    # dtype-agnostic all_gather, local partial exact);
                    # falls back to the plain collective when off, small,
                    # or capture/TP-shape mismatched.
                    output = _ar_quant(output_parallel)
                else:
                    output = tensor_model_parallel_all_reduce(output_parallel)
            else:
                output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
