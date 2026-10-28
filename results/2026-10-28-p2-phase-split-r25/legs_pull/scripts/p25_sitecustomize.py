# p25 sitecustomize — installs (1) the repo's gdn_metadata_fastpath patch
# (product posture intact) and (2) the r25 DIAGNOSTIC phase timer, into the
# running arc image without rebuild. Fork-safe meta-path pattern (r17):
# hooks fire right after the target vllm module executes. Runs in every
# python process via PYTHONPATH (APIServer + EngineCore + workers).
# NOTE: find_spec's `target` PARAM is part of the finder protocol and would
# shadow a same-named closure var — the module name must live in `mod_target`
# (bug caught on r25 boot 1: hook matched nothing, no installs fired).
import importlib.util as _iu
import sys


def _make_hook(mod_target, mod_path, tag):
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
                    s = _iu.spec_from_file_location(tag.replace("-", "_"),
                                                    mod_path)
                    m = _iu.module_from_spec(s)
                    s.loader.exec_module(m)
                    ok = m._install()
                    print(f"[{tag}] install={'OK' if ok else 'SKIPPED'} "
                          f"(pid={__import__('os').getpid()})", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[{tag}] FAILED: {e!r}", flush=True)

            spec.loader.exec_module = exec_module
            return spec

    return _Hook()


sys.meta_path.insert(0, _make_hook(
    "vllm.v1.attention.backends.gdn_attn",
    "/root/p25cap/gdn_metadata_fastpath.py", "p10"))
sys.meta_path.insert(0, _make_hook(
    "vllm.v1.worker.gpu.model_runner",
    "/root/p25cap/r25_phase_timer.py", "r25"))
