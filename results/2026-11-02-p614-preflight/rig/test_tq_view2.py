# arc p614 pre-flight v2 (CPU-only, inside the candidate image): the
# L1 debug finding baked in -- the pool arrives as 128-token KERNEL
# blocks (live shape (6500, 2, 128, 262), strides (67072, 262, 524,
# 1)); the view must merge them back to manager-PAGE granularity
# (NB_pages, 1664, 1, 2, HS) so the kit detector's BS >= 256 bound
# passes. Exercises the EXACT patched _arc_attn_pool_view (AST-extracted)
# on kernel-block pools for all four presets at the LIVE geometry
# (head 256, H_kv 2 @TP2), through the INSTALLED kit detector and the
# INSTALLED twohs transfer branch (multi-block, both directions).
import ast
import importlib.util as ilu
import sys

import torch

ADAPTER = "/p614/vllm_v1_adapter.py"
FAIL = 0


def check(name, ok, extra=""):
    global FAIL
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAIL += 1


src = open(ADAPTER).read()
tree = ast.parse(src)  # 1. syntax
print("[PASS] adapter parses")

fn = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_arc_attn_pool_view":
        fn = ast.get_source_segment(src, node)
assert fn, "view fn not found"
ns = {"torch": torch}
exec(fn, ns)
view_fn = ns["_arc_attn_pool_view"]


class Shim:
    pass


# locate the module defining VLLM_Detector, then install the kit overrides
# exactly like the platform hook does
from lmcache.v1.gpu_connector.kv_format.detection import detect_format  # noqa: E402

det_mod = None
for name, m in list(sys.modules.items()):
    if m is not None and "lmcache" in name and hasattr(m, "VLLM_Detector"):
        det_mod = m
        break
assert det_mod is not None, "VLLM_Detector module not found"

import lmcache.v1.platform.torch_ops as torch_ops  # noqa: E402

for kf, nm, target in (
    (
        "/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/"
        "lmcint/lmc_kvfmt_detector.py",
        "kvfmt",
        det_mod,
    ),
    (
        "/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/"
        "lmcint/lmc_twohs_transfer.py",
        "twohs",
        torch_ops,
    ),
):
    sp = ilu.spec_from_file_location(nm, kf)
    m = ilu.module_from_spec(sp)
    sp.loader.exec_module(m)
    m._install(target)
print(f"[PASS] kit overrides installed (detector host: {det_mod.__name__})")

from lmcache.lmcache_native import EngineKVFormat, TransferDirection  # noqa: E402
from lmcache.utils import EngineType  # noqa: E402

# live geometry: head 256, H_kv 2 per rank @TP2; slots from the wheel
from vllm.model_executor.layers.quantization.turboquant.config import (  # noqa: E402
    TurboQuantConfig,
)

PRESETS = [
    ("turboquant_4bit_nc", 4),
    ("turboquant_k8v4", 8),
    ("turboquant_k3v4_nc", 3),
    ("turboquant_3bit_nc", 3),
]
H_KV, HEAD, KBLK, PAGE, L = 2, 256, 128, 1664, 17
NPAGES = 10          # 130 kernel blocks
NBK = NPAGES * (PAGE // KBLK)

for preset, kbits in PRESETS:
    slot = TurboQuantConfig.from_cache_dtype(preset, HEAD).slot_size_aligned
    torch.manual_seed(7)
    row = H_KV * slot
    # physical (B, N, H*C) with N = 128-token kernel granularity; the
    # logical pool view the connector receives is (NBK, H, KBLK, slot)
    phys = torch.randint(0, 256, (NBK, KBLK, row), dtype=torch.uint8)
    pool = phys.view(NBK, KBLK, H_KV, slot).permute(0, 2, 1, 3)
    assert pool.dtype == torch.uint8
    assert tuple(pool.stride()) == (KBLK * row, slot, row, 1)
    s = Shim()
    s._arc_tq_attn = True
    s._arc_tq_slot_bytes = slot
    s._arc_tq_page_tokens = PAGE
    v = view_fn(s, pool)
    check(
        f"{preset} view (slot={slot})",
        tuple(v.shape) == (NPAGES, PAGE, 1, 2, row // 2)
        and v.is_contiguous()
        and v.data_ptr() == pool.data_ptr(),
        f"shape={tuple(v.shape)}",
    )

    views = [view_fn(s, pool) for _ in range(L)]
    fmt, canon = detect_format(views, EngineType.VLLM)
    check(
        f"{preset} detect",
        int(fmt) == int(EngineKVFormat.NL_X_NB_BS_NH_TWO_HS),
        f"fmt={fmt}",
    )

    # D2H through the installed kit branch; page 0 full + page 5 partial
    part = 777
    kv = torch.zeros(2, L, PAGE + part, row // 2, dtype=torch.uint8)
    slots = torch.cat(
        [
            torch.arange(PAGE, dtype=torch.int64),
            5 * PAGE + torch.arange(part, dtype=torch.int64),
        ]
    )
    torch_ops.multi_layer_kv_transfer(
        kv, views, slots, torch.device("cpu"), NPAGES * PAGE,
        TransferDirection.D2H, fmt, PAGE,
    )
    cat = torch.cat([kv[0], kv[1]], dim=-1)  # (L, PAGE+part, row)
    # physical page p lives at kernel blocks p*13 .. p*13+12
    phys_pages = phys.view(NPAGES, PAGE, row)
    ok = all(
        torch.equal(cat[l, :PAGE], phys_pages[0])
        and torch.equal(cat[l, PAGE:], phys_pages[5, :part])
        for l in range(L)
    )
    check(f"{preset} D2H rows byte-true (kernel-block merge)", ok)

    # H2D back into zeroed storage sharing the same view geometry
    phys2 = torch.zeros_like(phys)
    views2 = [
        view_fn(s, phys2.view(NBK, KBLK, H_KV, slot).permute(0, 2, 1, 3))
        for _ in range(L)
    ]
    torch_ops.multi_layer_kv_transfer(
        kv, views2, slots, torch.device("cpu"), NPAGES * PAGE,
        TransferDirection.H2D, fmt, PAGE,
    )
    phys2_pages = phys2.view(NPAGES, PAGE, row)
    ok = (
        torch.equal(phys2_pages[0], phys_pages[0])
        and torch.equal(phys2_pages[5, :part], phys_pages[5, :part])
        and phys2_pages[5, part:].sum() == 0
        and phys2_pages[1:5].sum() == 0
        and phys2_pages[6:].sum() == 0
    )
    check(f"{preset} H2D roundtrip byte-true", ok)

# fp8-shaped pool unchanged through the first branch (TQ flag off AND on)
for flag in (False, True):
    s_fp8 = Shim()
    s_fp8._arc_tq_attn = flag
    s_fp8._arc_tq_slot_bytes = 0
    s_fp8._arc_tq_page_tokens = 0
    fp8_pool = torch.zeros(476, 2, 896, 512, dtype=torch.float8_e4m3fn)
    v = view_fn(s_fp8, fp8_pool)
    check(
        f"fp8 pool first branch (tq_flag={flag})",
        tuple(v.shape) == (476, 896, 1, 2, 512),
    )

# negative guards: TQ flag on but pool NOT LBNHC-contiguous -> must raise;
# and kernel blocks that do NOT tile the page -> unmerged 128 face
s_bad = Shim()
s_bad._arc_tq_attn = True
s_bad._arc_tq_slot_bytes = 262
s_bad._arc_tq_page_tokens = PAGE
bad = torch.randint(0, 256, (NBK, H_KV, KBLK, 262), dtype=torch.uint8)
raised = False
try:
    view_fn(s_bad, bad)
except ValueError:
    raised = True
check("non-LBNHC TQ pool refuses loudly", raised)

s_km = Shim()
s_km._arc_tq_attn = True
s_km._arc_tq_slot_bytes = 0
s_km._arc_tq_page_tokens = PAGE
km_phys = torch.randint(0, 256, (63, KBLK, 524), dtype=torch.uint8)
km_pool = km_phys.view(63, KBLK, 2, 262).permute(0, 2, 1, 3)
v = view_fn(s_km, km_pool)
check(
    "non-page-tiling kernel count -> unmerged 128 face (loud detect fail)",
    tuple(v.shape) == (63, KBLK, 1, 2, 262),
)

print("PRE_FLIGHT_V2:", "OK" if FAIL == 0 else f"{FAIL} FAILURES")
sys.exit(1 if FAIL else 0)
