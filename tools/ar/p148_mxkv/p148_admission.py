# SPDX-License-Identifier: Apache-2.0
"""p148_admission.py — P6.47 leg B part 1: in-container admission +
routing probe for the MX KV dtypes (one-shot, NO engine boot, NO XPU
model load).

Verifies, in order, every wiring point mapped in the leg-B survey:
  1. Platform load: vllm resolves ArcXpuPlatform from the /work tree
     (PYTHONPATH law), wheel hooks install, the CacheDType source
     transform fires (or the documented fallback engages).
  2. Literal admission: "mxfp4"/"mxfp8" are TRUE members of
     vllm.config.cache.CacheDType; CacheConfig(cache_dtype=...) admits.
  3. DTYPE map: STR_DTYPE_TO_TORCH_DTYPE carries MX -> torch.uint8.
  4. Mode alias: get_kv_quant_mode maps MX onto TURBOQUANT members.
  5. Backend-family identity: AttentionBackendEnum carries a true MXKV
     member whose value is the MX backend qualname (MX is its own
     family, not a TURBOQUANT label).
  6. Backend routing: ArcXpuPlatform.get_attn_backend_cls returns the
     MX backend qualname; resolve_obj_by_qualname loads it; the format
     stamp is set; customize_spec produces the mx_layout slot bytes.
  7. tq_spec_fix carry-over: the TurboQuant builder carries the P6.19
     capture-forcing patch (verify-graph safety law, dtype-wide).

Exit 0 = all faces PASS. Any FAIL exits 1 with the face name.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORK = Path("/work")
sys.path.insert(0, str(WORK))

_RESULTS: list[tuple[str, bool, str]] = []


def face(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, ok, detail))
    print(f"[face:{name}] {'ok' if ok else 'FAIL'} {detail}", flush=True)


def main() -> int:
    # ---- 1. platform load (finder install happens inside import vllm) ----
    import vllm  # noqa: F401 — platform plugin resolves ArcXpuPlatform

    from vllm.platforms import current_platform

    cls_name = type(current_platform).__name__
    face("platform", cls_name == "ArcXpuPlatform", f"current_platform={cls_name}")

    import typing

    from vllm.config.cache import CacheConfig, CacheDType

    args = typing.get_args(CacheDType)
    face(
        "literal",
        "mxfp4" in args and "mxfp8" in args,
        f"members={'mxfp4' in args}/{'mxfp8' in args} "
        f"({'source transform' if 'mxfp8' in args else 'fallback path'})",
    )

    # ---- 2. CacheConfig admission (the pydantic gate) ----
    try:
        for dt in ("mxfp4", "mxfp8"):
            c = CacheConfig(cache_dtype=dt)
            if c.cache_dtype != dt:
                face("admit", False, f"round-trip lost dtype: {c.cache_dtype!r}")
                return 1
        face("admit", True, "CacheConfig(cache_dtype=mxfp4/mxfp8) admitted+round-trip")
    except Exception as e:  # noqa: BLE001
        face("admit", False, repr(e))
        return 1

    # ---- 3. dtype map ----
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

    import torch

    face(
        "dtype_map",
        STR_DTYPE_TO_TORCH_DTYPE.get("mxfp4") is torch.uint8
        and STR_DTYPE_TO_TORCH_DTYPE.get("mxfp8") is torch.uint8,
        "mxfp4/mxfp8 -> uint8",
    )

    # ---- 4. mode alias ----
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

    m8 = get_kv_quant_mode("mxfp8")
    m4 = get_kv_quant_mode("mxfp4")
    face(
        "mode_alias",
        m8 is KVQuantMode.TURBOQUANT_K8V4 and m4 is KVQuantMode.TURBOQUANT_4BIT_NC,
        f"mxfp8={m8.name} mxfp4={m4.name} is_turboquant={m8.is_turboquant}",
    )

    # ---- 5. backend-family enum member (MXKV is its OWN family) ----
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    want_path = "vllm_gaudi.xpu.patches.mx_kv.engine.MxAttentionBackend"
    try:
        member = AttentionBackendEnum["MXKV"]
        how = (
            "source transform"
            if "MXKV" in AttentionBackendEnum.__dict__
            else "register-time injection"
        )
        face(
            "enum_member",
            member.value == want_path and member.get_path() == want_path,
            f"MXKV={member.value} ({how})",
        )
    except Exception as e:  # noqa: BLE001
        face("enum_member", False, repr(e))
        return 1

    # ---- 6. backend routing + slot math ----
    from vllm.v1.attention.selector import AttentionSelectorConfig

    cfg = AttentionSelectorConfig(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="mxfp8",
        block_size=64,
    )
    path = current_platform.get_attn_backend_cls(None, attn_selector_config=cfg)
    face("routing", path.endswith("mx_kv.engine.MxAttentionBackend"), path)

    from vllm.utils.import_utils import resolve_obj_by_qualname

    backend = resolve_obj_by_qualname(path)
    face(
        "stamp",
        backend.active_kv_dtype == "mxfp8",
        f"active_kv_dtype={backend.active_kv_dtype}",
    )

    from vllm.v1.kv_cache_interface import FullAttentionSpec

    for dt, want in (("mxfp8", 264), ("mxfp4", 136)):
        backend.active_kv_dtype = dt
        spec = FullAttentionSpec(
            block_size=64,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.uint8,
            kv_quant_mode=get_kv_quant_mode(dt),
        )
        out = backend.customize_spec(spec)
        got = out.state_content_bytes
        face("slot", got == want, f"{dt}: state_content_bytes={got} want={want}")
        # page_size_bytes is the BLOCK-sized page: block_size(64) x
        # num_heads(8) x slot bytes.
        want_page = 64 * 8 * want
        face(
            "page",
            out.page_size_bytes == want_page,
            f"{dt}: page_size_bytes={out.page_size_bytes} want={want_page}",
        )

    # ---- 7. tq_spec_fix carry-over on the builder ----
    builder_cls = backend.get_builder_cls()
    patched = getattr(
        builder_cls.build_for_cudagraph_capture, "_arc_p619_tq_spec_fix", False
    )
    # The marker lives on the wrapped forward; the capture builder is
    # replaced wholesale by the hook — detect via the module attribute.
    import vllm.v1.attention.backends.turboquant_attn as tq_mod

    impl_patched = getattr(tq_mod.TurboQuantAttentionImpl.forward, "_arc_p619_tq_spec_fix", False)
    face(
        "tq_spec_fix",
        patched or impl_patched,
        f"builder_forced={patched} impl_forward_patched={impl_patched}",
    )

    # ---- engine module import smoke (kernel wiring resolves) ----
    try:
        from vllm_gaudi.xpu.patches.mx_kv.engine import (  # noqa: F401
            MxAttentionBackend,
            MxAttentionImpl,
        )
        from vllm_gaudi.xpu.patches.mx_kv.mx_dequant import (  # noqa: F401
            mx_dequant_cache,
        )

        face("engine_import", True, "mx_kv.engine + mx_dequant import OK")
    except Exception as e:  # noqa: BLE001
        face("engine_import", False, repr(e))
        return 1

    fails = [n for n, ok, _ in _RESULTS if not ok]
    print(
        f"VERDICT p148-admission {'PASS' if not fails else 'FAIL'} "
        f"({len(_RESULTS) - len(fails)}/{len(_RESULTS)} faces)"
        + (f" failing={fails}" if fails else ""),
        flush=True,
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
