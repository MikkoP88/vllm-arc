# p10 sitecustomize — installs the repo's gdn_metadata_fastpath patch
# (vllm_gaudi/xpu/patches/gdn_metadata_fastpath.py, deployed as a file per
# fork law) into the running arc image without an image rebuild. Loads the
# module from /root/p10cap/ right after vllm's gdn_attn module executes
# (fork-safe meta-path pattern, r17 lesson) and calls _install(). Runs in
# every python process via PYTHONPATH (APIServer + EngineCore + workers).
import importlib.util as _iu
import sys

_MOD = "/root/p10cap/gdn_metadata_fastpath.py"
_TARGET = "vllm.v1.attention.backends.gdn_attn"


class _InstallP10:
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
                s = _iu.spec_from_file_location("arc_p10_patch", _MOD)
                m = _iu.module_from_spec(s)
                s.loader.exec_module(m)
                ok = m._install()
                print(f"[p10] gdn fastpath install={'OK' if ok else 'SKIPPED'} "
                      f"(pid={__import__('os').getpid()})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[p10] gdn fastpath FAILED: {e!r}", flush=True)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _InstallP10())
