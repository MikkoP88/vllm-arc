# arc p614 pre-flight (CPU-only, runs inside the cert5 image):
# 1. syntax-check the patched adapter source
# 2. exercise the EXACT patched _arc_attn_pool_view (AST-extracted) on a
#    synthetic LBNHC TQ pool for all four presets @ TP2 geometry
# 3. feed the views through the INSTALLED lmcache detector (kit override
#    installed the same way the platform hook does it)
# 4. roundtrip D2H->H2D through the INSTALLED kit twohs transfer branch
# 5. fp8-shaped pool still maps through the unchanged first branch
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

PRESETS = {  # slot_size_aligned per the wheel's TurboQuantConfig @ head 128
    "turboquant_4bit_nc": 134,
    "turboquant_k8v4": 196,
    "turboquant_k3v4_nc": 118,
    "turboquant_3bit_nc": 102,
}
H_KV, NB, N, L = 4, 456, 1664, 17  # TP2 per-rank heads; N = forced attn block

for preset, slot in PRESETS.items():
    torch.manual_seed(7)
    row = H_KV * slot
    phys = torch.randint(0, 256, (NB, N, row), dtype=torch.uint8)
    # LBNHC: logical (B, H, N, C) strided view of the physical (B, N, H*C)
    pool = phys.view(NB, N, H_KV, slot).permute(0, 2, 1, 3)
    assert not pool.is_contiguous() and pool.dtype == torch.uint8
    s = Shim()
    s._arc_tq_attn = True
    s._arc_tq_slot_bytes = slot
    v = view_fn(s, pool)
    check(
        f"{preset} view",
        tuple(v.shape) == (NB, N, 1, 2, row // 2)
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

    # D2H through the installed kit branch; per-token row = plane0 ++ plane1
    # slot plan: block 0 full + block 5 partial (multi-block addressing)
    part = 777
    kv = torch.zeros(2, L, N + part, row // 2, dtype=torch.uint8)
    slots = torch.cat(
        [
            torch.arange(N, dtype=torch.int64),
            5 * N + torch.arange(part, dtype=torch.int64),
        ]
    )
    torch_ops.multi_layer_kv_transfer(
        kv, views, slots, torch.device("cpu"), NB * N,
        TransferDirection.D2H, fmt, N,
    )
    cat = torch.cat([kv[0], kv[1]], dim=-1)  # (L, N+part, row)
    ok = all(
        torch.equal(cat[l, :N], phys[0]) and torch.equal(cat[l, N:], phys[5, :part])
        for l in range(L)
    )
    check(f"{preset} D2H rows byte-true", ok)

    # H2D back into zeroed storage sharing the same view geometry
    phys2 = torch.zeros_like(phys)
    views2 = [
        view_fn(s, phys2.view(NB, N, H_KV, slot).permute(0, 2, 1, 3))
        for _ in range(L)
    ]
    torch_ops.multi_layer_kv_transfer(
        kv, views2, slots, torch.device("cpu"), NB * N,
        TransferDirection.H2D, fmt, N,
    )
    ok = (
        torch.equal(phys2[0], phys[0])
        and torch.equal(phys2[5, :part], phys[5, :part])
        and phys2[5, part:].sum() == 0
        and phys2[1:5].sum() == 0
        and phys2[6:].sum() == 0
    )
    check(f"{preset} H2D roundtrip byte-true", ok)

# 5. fp8-shaped pool unchanged through the first branch (TQ flag off AND on:
# with the flag on, an fp8-dtype pool must still take the first branch)
for flag in (False, True):
    s_fp8 = Shim()
    s_fp8._arc_tq_attn = flag
    s_fp8._arc_tq_slot_bytes = 0
    fp8_pool = torch.zeros(476, 2, 896, 512, dtype=torch.float8_e4m3fn)
    v = view_fn(s_fp8, fp8_pool)
    check(
        f"fp8 pool first branch (tq_flag={flag})",
        tuple(v.shape) == (476, 896, 1, 2, 512),
    )

# 6. negative guard: TQ flag on but pool NOT LBNHC-contiguous -> must raise
s_bad = Shim()
s_bad._arc_tq_attn = True
s_bad._arc_tq_slot_bytes = 0
bad = torch.randint(0, 256, (NB, H_KV, N, slot), dtype=torch.uint8)
# ^ contiguous (B,H,N,C): the branch's permute(0,2,1,3) is NON-contiguous
raised = False
try:
    view_fn(s_bad, bad)
except ValueError:
    raised = True
check("non-LBNHC TQ pool refuses loudly", raised)

print("PRE_FLIGHT:", "OK" if FAIL == 0 else f"{FAIL} FAILURES")
sys.exit(1 if FAIL else 0)
