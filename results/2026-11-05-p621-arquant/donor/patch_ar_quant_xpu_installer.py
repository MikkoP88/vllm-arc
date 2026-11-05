def patch_ar_quant_xpu() -> None:
    """W13-AR (IEXP_AR_QUANT=1): quantized-transport TP2 all-reduce for
    prefill-class payloads.

    Wraps vllm.distributed.tensor_model_parallel_all_reduce with
    vllm_gaudi.ops.ar_quant.ar_quant_all_reduce (peer partial as per-row
    scaled e4m3 over the dtype-agnostic all_gather, local partial exact
    bf16 -- the only byte-cut left on a link whose oneCCL AR already runs
    at the P2P ceiling; see ar_quant.py).  The original is stashed as
    vllm.distributed._iexp_ar_quant_orig for the lane's fallbacks, and
    rebound in every module that had bound it by from-import before this
    patch ran (later importers bind the wrapped name directly).  Inert
    unless IEXP_AR_QUANT=1."""
    import logging

    if os.environ.get("IEXP_AR_QUANT", "0") != "1":
        return
    import vllm.distributed as _vd
    from vllm_gaudi.ops.ar_quant import ar_quant_all_reduce as _lane

    _orig = _vd.tensor_model_parallel_all_reduce
    if getattr(_orig, "_iexp_ar_quant", False):
        return
    setattr(_vd, "_iexp_ar_quant_orig", _orig)

    def _arq(x):
        return _lane(x)

    _arq._iexp_ar_quant = True  # noqa: B010
    _vd.tensor_model_parallel_all_reduce = _arq
    for _m in list(sys.modules.values()):
        if _m is None:
            continue
        if getattr(_m, "tensor_model_parallel_all_reduce", None) is _orig:
            setattr(_m, "tensor_model_parallel_all_reduce", _arq)
    logging.getLogger(__name__).warning(
        "ar_quant: quantized-transport all-reduce patch installed "
        "(IEXP_AR_QUANT=1)")
