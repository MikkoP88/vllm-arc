# r128j4_scratch.py -- P6.34 arc7 JM g16 scratch-garbage dependence test.
# r128j3 mixmatch law: (x_p,rp_p,sc_p) and (x_f,rp_f,sc_f) PASS; every
# mixed combo FAILS rel 1e2..1e4 -- data-identical semantics, so the
# verdict tracks ALLOCATOR HISTORY, i.e. the launcher's at::empty scratch
# (subs*768 B, r128h law) delivers recycled garbage into lanes the f4
# path reads but staging never wrote. Fresh zero pages -> PASS; recycled
# torch-op garbage -> FAIL.
# Cells: P-verbatim 5x (determinism), then poison-recycle tests with
# 0x3C / 0xFF / 0x00 fills sized 63*768=48384 B (covers any subs<=63 via
# front-split), plus heavy-churn first-call in fresh subprocesses.
import subprocess
import sys
import torch

sys.path.insert(0, "/tmp/arc7wheel")
import vllm_xpu_kernels  # noqa: F401,E402
import vllm_xpu_kernels._xpu_C  # noqa: F401,E402

ops = torch.ops._xpu_C
dev = torch.device("xpu")
LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64)
K, N, M = 1024, 1003, 1
npad = ((N + 15) // 16) * 16


def repack_f4(w4v):
    n, k = w4v.shape
    nt, kt = (n + 15) // 16, k // 16
    if n < nt * 16:
        w4v = torch.cat([w4v, torch.zeros(
            nt * 16 - n, k, dtype=w4v.dtype, device=w4v.device)], 0)
    v = (w4v.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
         .contiguous().view(nt, kt, 16, 8, 2))
    return (v[..., 0] | (v[..., 1] << 4)).view(nt, kt, 128)


def build_set(seed, scale_lo=120, scale_hi=136):
    torch.manual_seed(seed)
    x = torch.randn(M, K, dtype=torch.bfloat16).to(dev)
    w4v = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    scb = torch.randint(scale_lo, scale_hi, (K // 32, N), dtype=torch.uint8)
    rp = repack_f4(w4v.to(dev))
    sc = torch.cat([scb.to(dev), torch.zeros(
        (K // 32, npad - N), dtype=torch.uint8, device=dev)], 1
    ).contiguous().view(torch.float8_e8m0fnu)
    w = LUT[w4v.long()].double()
    g = torch.pow(2.0, scb.double() - 127.0)
    ref = x.cpu().double() @ (w.t() * torch.repeat_interleave(g, 32, 0))
    return x, rp, sc, ref


def run(tag, x, rp, sc, ref):
    out = ops.jm_fp4_gemm_w4a16(
        x, rp, sc, torch.bfloat16, None)[..., :N].float().cpu().double()
    bad = ((out - ref).abs() / ref.abs().clamp_min(1.0)).max().item()
    print(f"{tag:46s} rel={bad:.4g} {'OK' if bad < 2e-2 else 'WRONG'}",
          flush=True)
    return bad


x, rp, sc, ref = build_set(3)

# 1) determinism: same tensors, five consecutive calls
for i in range(5):
    run(f"D{i} same tensors call #{i}", x, rp, sc, ref)

# 2) poison-recycle: fill a 48384 B block, free it, call (the launcher's
#    at::empty scratch recycles it; garbage lands in unwritten lanes)
for fill, nm in ((0x3C, "poison3C"), (0xFF, "poisonFF"), (0x00, "zero00"),
                 (0x3C, "poison3C-again")):
    blk = torch.full((63 * 768,), fill, dtype=torch.uint8, device=dev)
    assert blk.data_ptr() != 0
    del blk
    run(f"{nm} recycled-scratch", x, rp, sc, ref)

# 3) heavy churn (many randn/randint frees) then call
for _ in range(8):
    t = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    del t
run("after heavy randn churn", x, rp, sc, ref)
print("SCRATCH_DONE")

# 4) fresh-process first-call verdict vs pre-call churn, via subprocess
FRESH = (
    "import sys,torch;sys.path.insert(0,'/tmp/arc7wheel');"
    "import vllm_xpu_kernels,vllm_xpu_kernels._xpu_C;"
    "ops=torch.ops._xpu_C;dev=torch.device('xpu');"
    "torch.manual_seed(3)"
    ";x=torch.randn(1,1024,dtype=torch.bfloat16).to(dev)"
    ";w=torch.randint(0,16,(1003,1024),dtype=torch.uint8)"
    ";s=torch.randint(120,136,(32,1003),dtype=torch.uint8)"
    ";nt,kt,npad=63,64,1008"
    ";wp=torch.cat([w,torch.zeros(5,1024,dtype=torch.uint8)],0)"
    ";v=(wp.view(nt,16,kt,16).permute(0,2,1,3).contiguous()"
    ".view(nt,kt,16,8,2))"
    ";rp=(v[...,0]|(v[...,1]<<4)).view(nt,kt,128).to(dev)"
    ";sc=torch.cat([s.to(dev),torch.zeros((32,5),dtype=torch.uint8,"
    "device=dev)],1).contiguous().view(torch.float8_e8m0fnu)"
    "%s"
    ";o=ops.jm_fp4_gemm_w4a16(x,rp,sc,torch.bfloat16,None)"
    "[...,:1003].float().cpu().double()"
    ";L=torch.tensor([0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6],"
    "dtype=torch.float64)"
    ";g=torch.pow(2.,s.double()-127.)"
    ";r=x.cpu().double()@(L[w.long()].double().t()*"
    "torch.repeat_interleave(g,32,0))"
    ";bad=((o-r).abs()/r.abs().clamp_min(1.)).max().item()"
    ";print('fresh rel=%.4g %s'%(bad,'OK' if bad<2e-2 else 'WRONG'))"
)
CHURN = (";[torch.randn(512,512,device=dev).cpu() for _ in range(4)];"
         "torch.xpu.synchronize()")
for tag, pre in (("fresh minimal-churn", ""), ("fresh heavy-churn", CHURN)):
    r = subprocess.run([sys.executable, "-c", FRESH % pre],
                       capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
    print(f"{tag:46s} {line}", flush=True)
