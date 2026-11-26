#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p247_mx_surface_probe.py -- P-E MX deep-research phase (P6.47,
r247): survey the MX levers the PINNED stack actually ships.

CPU-safe API census (no device init, no engine boot): imports the
image's torch/torchao and records what exists -- the P-E law: every
lever disposition cites an existence proof in the pinned torch
(the oneCCL-pinned build; no-lawless-upgrade law) or a banked
verdict, never an upstream-master assumption.

Surveys:
  torch   version; xpu build; e2m1/e4m3/e5m2 dtypes; scaled_mm
          variants and their signature surfaces (rowwise vs tensor-
          wise scales, bf16 out); public/private entry points.
  torchao version; mx_formats (mx_specs fields: elem formats, group
          sizes, scale dtype; to_mx / mx_quantization entry points);
          float4 weight-only API (e2m1 + scale/zero-point, group
          size reach); float8 dyn API; any prototype/mx layout.

Exit 0 = census complete (existence facts recorded; dispositions
happen in the report, not here).
"""

from __future__ import annotations

import importlib
import inspect
import sys


def _has(mod: object, name: str) -> str:
    return "yes" if getattr(mod, name, None) is not None else "NO"


def _sig(fn, n: int = 400) -> str:
    try:
        s = str(inspect.signature(fn))
    except (TypeError, ValueError):
        s = "<builtin/no-sig>"
    return s if len(s) <= n else s[: n - 3] + "..."


def _dispatch_xpu(op: str) -> None:
    """CPU-safe dispatch-table read: does the op register an XPU kernel?"""
    import torch

    try:
        table = torch._C._dispatch_dump_table(op)
        n = table.count("XPU")
        print(f"dispatch[{op}]: XPU entries={n}")
    except Exception as e:  # noqa: BLE001 -- census records misses
        print(f"dispatch[{op}]: NO TABLE ({e})")


def _aten_scan(needle: str) -> None:
    import torch

    try:
        names = [n for n in dir(torch.ops.aten)
                 if needle in n.lower() and not n.startswith("_")]
    except Exception:  # noqa: BLE001
        names = []
    print(f"aten scan '{needle}': {names[:12] or 'none'}")


def survey_torch() -> None:
    import torch

    print(f"torch.version={torch.__version__}")
    print(f"torch.xpu.build={hasattr(torch, 'xpu')}")
    for dt in ("float4_e2m1fn_x2", "float8_e4m3fn", "float8_e4m3fnuz",
               "float8_e5m2", "float8_e8m0fnu"):
        print(f"torch.dtype.{dt}={_has(torch, dt)}")
    fns = ("_scaled_mm", "scaled_mm", "_scaled_grouped_mm",
           "_scaled_dot_product")
    for f in fns:
        fn = getattr(torch, f, None)
        print(f"torch.{f}={'yes ' + _sig(fn) if fn else 'NO'}")
    fn = getattr(torch.nn.functional, "scaled_mm", None)
    print(f"torch.nn.functional.scaled_mm="
          f"{'yes ' + _sig(fn) if fn else 'NO'}")
    # public swizzle/layout helpers adjacent to scaled_mm rowwise
    for f in ("_rowwise", "swap_rowwise_layout"):
        print(f"torch.{f}={_has(torch, f)}")
    # scale-recipe surface of the public scaled_mm
    recipe_names = [n for n in dir(torch.nn.functional)
                    if "recipe" in n.lower() or "scale" in n.lower()]
    print(f"torch.nn.functional scale/recipe names: "
          f"{recipe_names[:16] or 'none'}")
    # dispatch evidence: XPU kernel registrations for the MX dot path
    for op in ("_scaled_mm", "aten::_scaled_mm"):
        _dispatch_xpu(op)
    # aten op families adjacent to MX: pack/convert/e2m1/e8m0/fp4
    for needle in ("float4", "e2m1", "e8m0", "mx", "scaled", "quantize"):
        _aten_scan(needle)


def survey_ipex() -> None:
    """The xpu C-ext lane the fork's W4A16 reroute actually calls."""
    import torch

    try:
        import intel_extension_for_pytorch as ipex

        print(f"ipex.version={getattr(ipex, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"ipex: IMPORT_FAIL ({e!r})")
        return
    for path in ("torch.xpu._C", "intel_extension_for_pytorch._C"):
        mod = torch
        ok = True
        for part in path.split(".")[1:]:
            mod = getattr(mod, part, None)
            if mod is None:
                ok = False
                break
        if ok:
            names = [n for n in dir(mod) if "fp4" in n or "gemm" in n]
            print(f"{path}: {names[:12] or 'no fp4/gemm names'}")


def survey_torchao() -> None:
    import torchao

    print(f"torchao.version={getattr(torchao, '__version__', '?')}")
    for sub in ("mx_formats", "quantization", "prototype"):
        try:
            m = importlib.import_module(f"torchao.{sub}")
            names = [n for n in dir(m) if not n.startswith("_")]
            print(f"torchao.{sub}: {', '.join(names[:40]) or '(empty)'}")
        except Exception as e:  # noqa: BLE001 -- census records misses
            print(f"torchao.{sub}: IMPORT_FAIL {e!r}")
    try:
        from torchao.mx_formats import mx_specs
        ms = mx_specs.mx_specs()
        print(f"mx_specs fields: {sorted(vars(ms)) or dir(ms)}")
        for field in ("elem_format", "group_size", "scale_dtype",
                      "scale_float_conversion"):
            print(f"mx_specs.{field}={getattr(ms, field, '<absent>')}")
    except Exception as e:  # noqa: BLE001
        print(f"mx_specs: NO ({e!r})")
    for path in ("torchao.mx_formats.to_mx",
                 "torchao.quantization.float4_weight_only",
                 "torchao.quantization.mx_float8"):
        mod_name, attr = path.rsplit(".", 1)
        try:
            m = importlib.import_module(mod_name)
            fn = getattr(m, attr, None)
            print(f"{path}={_sig(fn) if callable(fn) else _has(m, attr)}")
        except Exception as e:  # noqa: BLE001
            print(f"{path}: IMPORT_FAIL {e!r}")


def main() -> int:
    print("== p247 MX surface census (pinned stack, CPU-safe)")
    try:
        survey_torch()
    except Exception as e:  # noqa: BLE001
        print(f"TORCH_CENSUS_FAIL {e!r}")
        return 1
    try:
        survey_torchao()
    except Exception as e:  # noqa: BLE001
        print(f"TORCHAO_CENSUS_FAIL {e!r}")
    try:
        survey_ipex()
    except Exception as e:  # noqa: BLE001
        print(f"IPEX_CENSUS_FAIL {e!r}")
    print("CENSUS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
