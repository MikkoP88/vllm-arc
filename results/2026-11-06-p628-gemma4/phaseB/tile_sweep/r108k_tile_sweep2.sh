#!/bin/bash
# r108k_tile_sweep.sh — P6.28 FIX-DATA window: B70 tile/warp tuning sweep. (v2)
#
# v2: first launch crashed — env dicts passed positionally landed in run()'s
# `timeout` param (TypeError float+dict after the first point). All env dicts
# now env_extra= keyword. Points added from upstream research: M64w4T32
# (vllm#52585 RDNA3 winner shape), M128w8T64 (PR#60418 gfx head-256 high
# band), stage variants, and a full-shape (ntok=8192) validation section —
# PR#60418's own tables key configs per max_seqlen_q band, so the ntok=1024
# winner must be re-timed at the engine's real chunk shape.
#
# r108j verdict (banked micro_disc/): SLOW, not dead — full512-c1 bare
# kernel steady 8.3 s (~1.1 TFLOP = 0.13 TFLOPS vs 43.2 TFLOPS matmul
# ref = 0.3% of achievable); slide256-c2 4.2 s. Engine budget ~= 20 min
# device time per 32k face vs 300 s VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
# kill = the whole "freeze". Launcher today: BLOCK_M=16/BLOCK_Q=8,
# TILE_SIZE_PREFILL=32, warps default(4); tuned_large_head unreachable on
# XPU (get_device_capability -> None) and head-512-excluded anyway.
# Upstream fix pattern exists: PR#60418 (RDNA config tables keyed by padded
# head size x max_seqlen_q band) — this sweep gathers the XPU/B70 values.
set -uo pipefail
R=/root/build
CT=$R/p628/phaseB
IMG=vllm-arc:v0.30.0-b70-cert11
LEG=$CT/tile_sweep
mkdir -p $LEG

if pgrep -f "snapshot_download" >/dev/null 2>&1; then
  echo "FATAL: host model download in flight"; exit 1
fi
uptime >> $LEG/host_load.txt

w14_start() {
  docker rm -f arc-micro >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_serve_only.sh > $R/w14_restore_p628k2.log 2>&1 < /dev/null &
}
trap w14_start EXIT

WH=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health 2>/dev/null || true)
echo "W14 health before stop: $WH"
docker stop radiance-test >/dev/null 2>&1 || true
sleep 3

docker rm -f arc-micro >/dev/null 2>&1 || true
echo "=== [sweep] container boot (NO serve) $(date +%H:%M:%S) ==="
docker run -d --name arc-micro \
  --privileged --network host --ipc host \
  --shm-size 64g \
  -v /root/build/triton_cache:/root/.triton_cache \
  --entrypoint /bin/bash $IMG -c 'sleep infinity' >/dev/null
sleep 5

# --- DIAGNOSTIC launcher hook: env-overridable launch params.
# Inserted right after the tuned_large_head TILE block, BEFORE
# total_num_q_blocks (grid depends on BLOCK_Q) and before the use_td
# clamp (we re-clamp ourselves). Idempotent via marker. NEVER repo.
docker exec -i arc-micro /opt/venv/bin/python3 - < /dev/null > $LEG/hook_install.txt 2>&1 << 'PYHOOK' || true
OP = "/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py"
MARK = "# ARC-P628-SWEEP (diagnostic)"
ANCHOR = """    if tuned_large_head:
        TILE_SIZE_PREFILL = 128
"""
HOOK = MARK + """
    import os as _os
    _m = _os.environ.get("ARC_SWEEP_BLOCK_M")
    if _m:
        BLOCK_M = int(_m)
        BLOCK_Q = BLOCK_M // num_queries_per_kv
    _w = _os.environ.get("ARC_SWEEP_WARPS")
    if _w:
        launch_num_warps = int(_w)
    _s = _os.environ.get("ARC_SWEEP_STAGES")
    if _s:
        launch_num_stages = int(_s)
    _t = _os.environ.get("ARC_SWEEP_TILE")
    if _t:
        TILE_SIZE_PREFILL = int(_t)
    if use_td:
        TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, block_size)
"""
src = open(OP).read()
if MARK in src:
    print("HOOK_ALREADY_PRESENT")
elif ANCHOR not in src:
    print("ANCHOR_NOT_FOUND")
else:
    open(OP, "w").write(src.replace(ANCHOR, ANCHOR + HOOK, 1))
    print("HOOK_INSTALLED")
PYHOOK
grep -q "HOOK_INSTALLED\|HOOK_ALREADY_PRESENT" $LEG/hook_install.txt || { echo "FATAL: hook install failed"; cat $LEG/hook_install.txt; exit 1; }
cat $LEG/hook_install.txt

docker exec -i arc-micro /opt/venv/bin/python3 - < /dev/null > $LEG/sweep_results.txt 2>&1 << 'PYSWEEP'
import json, subprocess, sys, os

PY = sys.executable

CHILD = r'''
import json, sys, os, time, torch
cfg = json.loads(sys.argv[1])
dev = "xpu"
torch.manual_seed(0)
head = cfg["head"]; nqh = cfg["nqh"]; nkvh = cfg["nkvh"]
ntok = cfg["ntok"]; seqlen = cfg["seqlen"]; bs = 64
win = tuple(cfg["win"]); causal = True
nblk = (seqlen + bs - 1) // bs
kdt = torch.float8_e4m3fn
q = torch.randn(ntok, nqh, head, dtype=torch.float16, device=dev)
if cfg.get("dump"):
    torch.manual_seed(1234)
kc = torch.randn(nblk, bs, nkvh, head, dtype=torch.float16, device=dev).to(kdt)
vc = torch.randn(nblk, bs, nkvh, head, dtype=torch.float16, device=dev).to(kdt)
out = torch.empty_like(q)
bt = torch.arange(nblk - 1, -1, -1, dtype=torch.int32, device=dev).reshape(1, nblk)
cuq = torch.tensor([0, ntok], dtype=torch.int32, device=dev)
sq = torch.tensor([seqlen], dtype=torch.int32, device=dev)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
def call():
    unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cuq,
        max_seqlen_q=ntok, seqused_k=sq, max_seqlen_k=seqlen,
        softmax_scale=head ** -0.5, causal=True, alibi_slopes=None,
        window_size=win, block_table=bt, softcap=0.0, q_descale=None,
        kv_quant_mode=1,
        k_descale=torch.ones(1, nkvh, device=dev),
        v_descale=torch.ones(1, nkvh, device=dev),
        seq_threshold_3D=16, num_par_softmax_segments=16,
        softmax_segm_output=None, softmax_segm_max=None,
        softmax_segm_expsum=None, sinks=None, output_scale=None,
        mm_prefix_range=None)
    torch.xpu.synchronize()
n = cfg.get("repeat", 2)
call()
t0 = time.perf_counter()
for _ in range(n):
    call()
dt = (time.perf_counter() - t0) / n
if cfg.get("dump"):
    torch.save(out[:256].to(torch.float32).cpu(), cfg["dump"])
    print("SYNC_OK dump wall=%.4fs" % dt)
else:
    print("SYNC_OK wall=%.4fs x%d" % (dt, n))
'''

def run(name, cfg, timeout=240, env_extra=None):
    env = dict(os.environ)
    for k in ("ARC_SWEEP_BLOCK_M", "ARC_SWEEP_WARPS", "ARC_SWEEP_STAGES", "ARC_SWEEP_TILE"):
        env.pop(k, None)
    if env_extra:
        env.update(env_extra)
    try:
        r = subprocess.run([PY, "-c", CHILD, json.dumps(cfg)],
                           capture_output=True, text=True, timeout=timeout, env=env)
        status = r.stdout.strip().splitlines()[-1] if r.stdout.strip() \
            else f"ERR {r.stdout[-200:]} {r.stderr[-400:]}"
    except subprocess.TimeoutExpired:
        status = "WEDGE (timeout %ds)" % timeout
    print("[%-22s] %s" % (name, status), flush=True)
    return status

F5 = dict(head=512, nqh=16, nkvh=8, ntok=1024, seqlen=8192, win=(-1, -1))
S2 = dict(head=256, nqh=16, nkvh=8, ntok=1024, seqlen=8192, win=(1023, 0))

print("=== head-512 full-attn sweep (ntok=1024) ===", flush=True)
run("base-M16w4T32", F5)
run("M32w4T32", F5, env_extra={"ARC_SWEEP_BLOCK_M": "32"})
run("M32w8T32", F5, env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_WARPS": "8"})
run("M32w8T64", F5, env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64"})
run("M32w4T64", F5, env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_TILE": "64"})
run("M64w8T64", F5, env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64"})
run("M64w8T32", F5, env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8"})
run("M64w4T64", F5, env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_TILE": "64"})
run("M64w4T32", F5, env_extra={"ARC_SWEEP_BLOCK_M": "64"})
run("M32w8T64s2", F5, env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64", "ARC_SWEEP_STAGES": "2"})
run("M64w8T64s2", F5, env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64", "ARC_SWEEP_STAGES": "2"})
run("M64w8T64s1", F5, env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64", "ARC_SWEEP_STAGES": "1"})

print("=== head-256 sliding sweep (ntok=1024) ===", flush=True)
run("base-M16w4T32", S2)
run("M32w8T32", S2, env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_WARPS": "8"})
run("M32w8T64", S2, env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64"})
run("M64w8T64", S2, env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64"})
run("M64w4T64", S2, env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_TILE": "64"})
run("M128w8T64", S2, env_extra={"ARC_SWEEP_BLOCK_M": "128", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64"})

print("=== full-shape validation (ntok=8192, engine chunk band) ===", flush=True)
run("full512-base", {**F5, "ntok": 8192, "repeat": 3}, timeout=240)
run("full512-M64w8T32", {**F5, "ntok": 8192, "repeat": 3}, timeout=240,
    env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8"})
run("full512-M64w4T32", {**F5, "ntok": 8192, "repeat": 3}, timeout=240,
    env_extra={"ARC_SWEEP_BLOCK_M": "64"})
run("full512-M32w8T64", {**F5, "ntok": 8192, "repeat": 3}, timeout=240,
    env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64"})
run("slide256-base", {**S2, "ntok": 8192, "repeat": 3}, timeout=240)
run("slide256-M32w8T64", {**S2, "ntok": 8192, "repeat": 3}, timeout=240,
    env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64"})
run("slide256-M64w8T64", {**S2, "ntok": 8192, "repeat": 3}, timeout=240,
    env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64"})

print("=== numeric sanity: base vs tuned candidates (head 512) ===", flush=True)
run("dump-base", {**F5, "dump": "/root/out_base.pt"}, timeout=240)
run("dump-M32w8T64", {**F5, "dump": "/root/out_a.pt"}, timeout=240,
    env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64"})
run("dump-M64w8T32", {**F5, "dump": "/root/out_b.pt"}, timeout=240,
    env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8"})
import torch
a0 = torch.load("/root/out_base.pt")
for tag, f in (("M32w8T64", "/root/out_a.pt"), ("M64w8T32", "/root/out_b.pt")):
    b = torch.load(f)
    d = (a0 - b).abs()
    rel = d.max() / a0.abs().max().clamp_min(1e-6)
    print("NUMERIC[%s] max_abs=%.5f rel_vs_peak=%.5f allclose(1e-2)=%s"
          % (tag, d.max(), rel, torch.allclose(a0, b, atol=1e-2)), flush=True)
print("SWEEP_DONE", flush=True)
PYSWEEP

cat $LEG/sweep_results.txt
echo "=== [sweep] window done $(date +%H:%M:%S) ==="
