# r27 sitecustomize — p10 GDN metadata fastpath ONLY (timers-off attribution
# leg: no r25 phase timer, no p33 probe). Same exec_module-wrap meta-path
# pattern as the p33 cap sitecustomize (proven r17 fork-safe wiring).
import importlib.util as _iu
import sys

_CAP = "/root/r27cap"
_HOOKS = [
    (
        "vllm.v1.attention.backends.gdn_attn",
        f"{_CAP}/gdn_metadata_fastpath.py",
        "arc_p10_patch",
        "_install",
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
