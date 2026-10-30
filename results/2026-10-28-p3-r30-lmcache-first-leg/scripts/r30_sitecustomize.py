# r30 sitecustomize — p10 GDN metadata fastpath + p41 mamba LMCache glue
# + p41b LMCacheConnectorV1 HMA mixin (timers-off posture; same
# exec_module-wrap meta-path pattern as the p33 cap sitecustomize, proven
# r17 fork-safe wiring). The p41/p41b hooks pass the just-executed module
# to the patch entry so it can attach/rebind its symbols.
import importlib.util as _iu
import sys

_CAP = "/root/r30cap"
_HOOKS = [
    (
        "vllm.v1.attention.backends.gdn_attn",
        f"{_CAP}/gdn_metadata_fastpath.py",
        "arc_p10_patch",
        "_install",
        False,
    ),
    (
        "vllm.v1.worker.mamba_utils",
        f"{_CAP}/mamba_lmcache_glue.py",
        "arc_p41_patch",
        "_install",
        True,
    ),
    (
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
        f"{_CAP}/lmcache_hma_connector.py",
        "arc_p41b_patch",
        "_install",
        True,
    ),
    (
        "lmcache.v1.platform.xpu.device_ops",
        f"{_CAP}/lmc_xpu_pinned_env.py",
        "arc_p40c_patch",
        "_install",
        True,
    ),
]


def _make_hook(mod_target, mod_path, patch_name, entry, pass_module):
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
                    if pass_module:
                        getattr(m, entry)(module)
                    else:
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


for _t, _p, _n, _e, _pm in _HOOKS:
    sys.meta_path.insert(0, _make_hook(_t, _p, _n, _e, _pm))
