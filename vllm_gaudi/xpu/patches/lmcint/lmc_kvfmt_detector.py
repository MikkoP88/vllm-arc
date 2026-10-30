# arc r43 (2026-10-29): lmcache kv-format detector override.
#
# This engine's hybrid FA-XPU attention pool is a single stacked tensor
# (num_blocks, 2=K/V, block_size, kv_heads*head_size) whose PHYSICAL memory
# order is (num_blocks, block_size tokens, K/V, kvdim) -- per-token K/V
# interleaving (strides (2*bs*kvd, kvd, 2*kvd, 1)).
#
# Upstream lmcache's VLLM detector cannot express that layout: the raw
# rank-4 pool lands on the fused-CS branch whose kernel slot model counts
# K/V as separate slot units, folding every page P onto page P//2 (the
# N2/N4-class warm re-hit integrity break root-caused this leg), and no
# 5-D discover() branch places TWO on axis 3.
#
# The byte-true fix (adapter hands a contiguous zero-copy 5-D view
# (NB, BS, 1, TWO, kvdim), same storage/data_ptr as the pool): map it to
# NL_X_NB_BS_NH_TWO_HS with NH=1, HS=kvdim. That format's kernel
# addressing -- slot s -> block s//BS, within token nh*TWO*HS + two*HS +
# hs -- is exactly the pool's physical stride math, one slot per token,
# both K and V moved per slot. The singleton NH axis may sit on dim 1 or
# dim 2 depending on the contiguity recovery's stride sort, so the guard
# accepts either and canonicalizes via reshape (contiguous => view).
#
# Source-override kit file (p41b exec_module pattern); the installed
# lmcache tree is never edited.

# Third Party
import torch

# First Party
import lmcache.lmcache_native as lmcache_native

_ORIG_DISCOVER = None
_ORIG_IS_STATIC = False


def _arc_discover(self, kv_caches, layout_hints):
    try:
        if (isinstance(kv_caches, list) and kv_caches and isinstance(kv_caches[0], torch.Tensor)):
            t = kv_caches[0]
            if (t.dim() == 5 and t.is_contiguous() and int(t.shape[0]) >= 16 and int(t.shape[3]) == 2
                    and int(t.shape[4]) >= 128):
                if int(t.shape[2]) == 1 and int(t.shape[1]) >= 256:
                    bs = int(t.shape[1])
                elif int(t.shape[1]) == 1 and int(t.shape[2]) >= 256:
                    bs = int(t.shape[2])
                else:
                    bs = 0
                if bs:
                    kv_caches = [
                        x.reshape(x.shape[0], bs, 1, 2, x.shape[4])
                        if int(x.shape[2]) != 1 or int(x.shape[1]) != bs else x for x in kv_caches
                    ]
                    print(
                        "[r43kvfmt] arc interleaved attn pool -> "
                        "NL_X_NB_BS_NH_TWO_HS "
                        f"(NB={int(t.shape[0])} BS={bs} TWO=2 HS={int(t.shape[4])})",
                        flush=True,
                    )
                    return (
                        lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_TWO_HS,
                        kv_caches,
                    )
    except Exception:  # noqa: BLE001
        pass
    if _ORIG_IS_STATIC:
        return _ORIG_DISCOVER(kv_caches, layout_hints)
    return _ORIG_DISCOVER(self, kv_caches, layout_hints)


def _install(mod):
    global _ORIG_DISCOVER, _ORIG_IS_STATIC
    raw = vars(mod.VLLM_Detector)["discover"]
    if isinstance(raw, staticmethod):
        _ORIG_IS_STATIC = True
        _ORIG_DISCOVER = raw.__func__
    else:
        _ORIG_DISCOVER = raw
    mod.VLLM_Detector.discover = _arc_discover
