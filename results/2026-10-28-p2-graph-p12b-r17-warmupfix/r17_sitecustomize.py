# r17 sitecustomize — installs the repo's first_request_warmup patch
# (vllm_gaudi/xpu/patches/first_request_warmup.py, deployed as a file per
# fork law) into the running arc image without an image rebuild. Loads the
# module from /root/r17patch/ at AsyncLLM import time (fork-safe meta-path
# pattern, r16 lesson) and calls _install().
import importlib.util as _iu
import sys

_MOD = "/root/r17patch/first_request_warmup.py"
_TARGET = "vllm.v1.engine.async_llm"


class _InstallWarmup:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
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
                s = _iu.spec_from_file_location("arc_warmup_patch", _MOD)
                m = _iu.module_from_spec(s)
                s.loader.exec_module(m)
                m._install()
                print(f"[r17] warmup patch install attempted (pid={module.__name__})",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[r17] warmup patch FAILED: {e!r}", flush=True)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _InstallWarmup())
