# r128h_poison.py -- P6.34 arc7 JM store-drop vs load-drop discriminator.
# Pinpoint law: halfwords at offset == 3 (mod 4) of the staging scratch
# contribute exactly zero (both sides, every run, clean values -- never
# garbage). Two mechanism classes fit:
#   LOAD-drop:  joint_matrix_load never delivers elements at those offsets
#               -> scratch content there is IRRELEVANT, out stays 24.
#   STORE-drop: the staging u16 stores to those offsets never land ->
#               the offsets keep the allocator block's PRIOR content.
#               Poison the recycled scratch block with 0x3C bytes
#               (halfword 0x3C3C ~= 1.0273): dead k then contributes
#               1.0273 x 1.0 each -> out[0] = 24 + 8*1.0273 ~= 32.22.
# The launcher's scratch is at::empty(subs*768 B) freed back to the pool
# at op end; a same-size torch.full freed between calls recycles it.
import sys
import torch

sys.path.insert(0, "/tmp/arc7wheel")
import vllm_xpu_kernels  # noqa: F401,E402
import vllm_xpu_kernels._xpu_C  # noqa: F401,E402

ops = torch.ops._xpu_C
dev = torch.device("xpu")


def repack_f8(w8):  # [N, K] u8 -> column tiles [N/16, K/16, 256]
    n, k = w8.shape
    nt, kt = (n + 15) // 16, k // 16
    if n < nt * 16:
        w8 = torch.cat(
            [w8, torch.zeros(nt * 16 - n, k, dtype=torch.uint8)], 0)
    return (w8.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
            .contiguous().view(nt, kt, 256))


def run(name, w8, x):
    rp = repack_f8(w8).to(dev)
    xd = x.to(dev)
    out = ops.jm_fp8_gemm_w8a16(xd, rp, None, None)[..., :16].float().cpu()
    print(f"{name}: out[0]={out[0, 0].item():.5f} out[3]={out[0, 3].item():.5f}")


K, N = 32, 16
C1 = 0x38  # e4m3 1.0
x1 = torch.ones(1, 1, K, dtype=torch.bfloat16)
w = torch.full((N, K), C1, dtype=torch.uint8)

# 1) baseline (first op call allocates + frees the 768 B scratch block)
run("1 baseline    ", w, x1)  # expect 24.0 (defect) / 32.0 (fixed)

# 2) poison the freed block, rerun
poison = torch.full((768,), 0x3C, dtype=torch.uint8, device=dev)
assert poison.data_ptr() != 0
del poison
run("2 poisoned    ", w, x1)  # 24.0 => LOAD-drop; ~32.22 => STORE-drop

# 3) poison again with 0xFF (halfword 0xFFFF = -NaN x1.88e19? f16 NaN) --
#    a DIFFERENT poison to confirm the value tracks scratch content
poison = torch.full((768,), 0xFF, dtype=torch.uint8, device=dev)
del poison
run("3 poison=FF   ", w, x1)  # NaN/garbage => STORE-drop; 24.0 => LOAD-drop
print("POISON_DONE")
