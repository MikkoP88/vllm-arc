# p33 sitecustomize — deploys, via PYTHONPATH meta-path hooks (r17 fork-safe
# pattern; closure var renamed mod_target — the find_spec `target` PARAM
# SHADOWS same-named closures, r25 law):
#   1. vllm.v1.attention.backends.gdn_attn            -> p10 gdn fastpath
#   2. vllm.v1.worker.gpu.model_runner                -> r25 v2 phase timer
#   3. vllm.v1.worker.gpu.spec_decode.autoregressive.speculator
#        -> p33 probe (speculator entry)
#   4. vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils
#        -> p33 probe (manager entry; fires while the speculator module is
#           still mid-import, so it must not import speculator names)
# The p33 patch file is loaded ONCE per process under a fixed module name so
# both installs share state.
import importlib.util as _iu
import sys

_CAP = "/root/p33cap"

_HOOKS = [
    (
        "vllm.v1.attention.backends.gdn_attn",
        f"{_CAP}/gdn_metadata_fastpath.py",
        "arc_p10_patch",
        "_install",
    ),
    (
        "vllm.v1.worker.gpu.model_runner",
        f"{_CAP}/r25_phase_timer.py",
        "arc_r25_patch",
        "_install",
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator",
        f"{_CAP}/p33_probe.py",
        "arc_p33_probe",
        "install_for_speculator",
    ),
    (
        "vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils",
        f"{_CAP}/p33_probe.py",
        "arc_p33_probe",
        "install_for_cudagraph_utils",
    ),
]


def _make_hook(mod_target, mod_path, patch_name, entry):
    class _Hook:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != mod_target:
                return None
            sys.meta_path.remove(self)
            try:
                spec = _iu.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    if patch_name in sys.modules:
                        m = sys.modules[patch_name]
                    else:
                        s = _iu.spec_from_file_location(patch_name, mod_path)
                        m = _iu.module_from_spec(s)
                        sys.modules[patch_name] = m
                        s.loader.exec_module(m)
                    getattr(m, entry)()
                    print(
                        f"[{mod_target.rsplit('.', 1)[-1]}] {entry} OK "
                        f"(pid={__import__('os').getpid()})",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[{mod_target}] {entry} FAILED: {e!r}", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    return _Hook()


for _t, _p, _n, _e in _HOOKS:
    sys.meta_path.insert(0, _make_hook(_t, _p, _n, _e))
