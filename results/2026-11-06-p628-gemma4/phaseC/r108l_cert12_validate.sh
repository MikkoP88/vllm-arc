#!/bin/bash
# r108l_cert12_validate.sh — P6.28 Window A: build cert12 and validate the
# REAL patched launcher (no env hooks) against stock cert11.
#
# Sections:
#   [1] stock same-seed dumps + kv16k risk-point timings from cert11
#       (arc-micro, pristine launcher, NO diagnostic hook)
#   [2] cert12 build (overlay on cert11; context = patcher +
#       cert12.Dockerfile shipped to /root/build/p628/cert12ctx)
#   [3] patched gate asserts + timing points + same-seed numeric
#       compares vs the [1] dumps
# Dumps land in /root/build/p628/phaseC (mounted at /phaseC in both
# containers) so numerics compare across images. Same seeds as r108k
# (q manual_seed(0); kv re-seeded 1234 when dumping). W14 restore trap
# fires on exit; rig never left on an arc image.
set -uo pipefail
R=/root/build
CT=$R/p628/phaseC
CTX=$R/p628/cert12ctx
IMG_NEW=vllm-arc:v0.30.0-b70-cert12
IMG_OLD=vllm-arc:v0.30.0-b70-cert11
LEG=$CT
mkdir -p $LEG $CTX

if pgrep -f "snapshot_download" >/dev/null 2>&1; then
  echo "FATAL: host model download in flight"; exit 1
fi

FORK_SHA="$1"
[ -n "$FORK_SHA" ] || { echo "usage: $0 <fork_sha>"; exit 1; }

w14_start() {
  docker rm -f arc-micro >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_serve_only.sh > $R/w14_restore_p628l.log 2>&1 < /dev/null &
}
trap w14_start EXIT

WH=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health 2>/dev/null || true)
echo "W14 health before stop: $WH"
docker stop radiance-test >/dev/null 2>&1 || true
sleep 3

boot_micro() {  # $1 = image
  docker rm -f arc-micro >/dev/null 2>&1 || true
  docker run -d --name arc-micro \
    --privileged --network host --ipc host \
    --shm-size 64g \
    -v /root/build/triton_cache:/root/.triton_cache \
    -v $CT:/phaseC \
    --entrypoint /bin/bash "$1" -c 'sleep infinity' >/dev/null
  sleep 5
}

# --- [1] STOCK (cert11): dumps + kv16k timings ---------------------------
echo "=== [1] stock cert11 dumps + kv16k risk points $(date +%H:%M:%S) ==="
boot_micro $IMG_OLD
# prove the pristine launcher is REALLY pristine (no sweep hook left)
docker exec arc-micro grep -c "ARC-P628-SWEEP" \
  /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py \
  > $LEG/stock_hook_check.txt 2>&1 || true
cat $LEG/stock_hook_check.txt
grep -q "^0$" $LEG/stock_hook_check.txt \
  || { echo "FATAL: cert11 launcher not pristine"; exit 1; }

docker exec -i arc-micro /opt/venv/bin/python3 - < /dev/null > $LEG/stock_points.txt 2>&1 << 'PY1'
import json, subprocess, sys, os

PY = sys.executable
CHILD = r'''
import json, sys, time, torch
cfg = json.loads(sys.argv[1])
dev = "xpu"
torch.manual_seed(0)
head = cfg["head"]; nqh = cfg["nqh"]; nkvh = cfg["nkvh"]
ntok = cfg["ntok"]; seqlen = cfg["seqlen"]; bs = 64
win = tuple(cfg["win"])
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
call()
if cfg.get("dump"):
    torch.save(out[:256].to(torch.float32).cpu(), cfg["dump"])
    print("DUMP_OK " + cfg["dump"])
else:
    n = cfg.get("repeat", 2)
    t0 = time.perf_counter()
    for _ in range(n):
        call()
    dt = (time.perf_counter() - t0) / n
    print("SYNC_OK wall=%.4fs x%d" % (dt, n))
'''

def run(name, cfg, timeout=240):
    try:
        r = subprocess.run([PY, "-c", CHILD, json.dumps(cfg)],
                           capture_output=True, text=True, timeout=timeout)
        status = r.stdout.strip().splitlines()[-1] if r.stdout.strip() \
            else f"ERR {r.stdout[-200:]} {r.stderr[-400:]}"
    except subprocess.TimeoutExpired:
        status = "WEDGE (timeout %ds)" % timeout
    print("[%s] %s" % (name, status), flush=True)

F5 = dict(head=512, nqh=16, nkvh=8, ntok=1024, seqlen=8192, win=(-1, -1))
S2 = dict(head=256, nqh=16, nkvh=8, ntok=1024, seqlen=8192, win=(1023, 0))
F5H = {**F5, "ntok": 8192}
S2H = {**S2, "ntok": 8192}

run("stock-dump-f5-1k", {**F5, "dump": "/phaseC/stock_f5_1k.pt"})
run("stock-dump-f5-8k", {**F5H, "dump": "/phaseC/stock_f5_8k.pt"})
run("stock-dump-s2-8k", {**S2H, "dump": "/phaseC/stock_s2_8k.pt"})
run("stock-f5-kv16k", {**F5H, "seqlen": 16384, "repeat": 2}, timeout=360)
run("stock-s2-kv16k", {**S2H, "seqlen": 16384, "repeat": 2}, timeout=360)
print("STOCK_DONE", flush=True)
PY1
cat $LEG/stock_points.txt
grep -q "STOCK_DONE" $LEG/stock_points.txt \
  || { echo "FATAL: stock section incomplete"; exit 1; }
docker rm -f arc-micro >/dev/null

# --- [2] cert12 build ------------------------------------------------------
echo "=== [2] cert12 build $(date +%H:%M:%S) ==="
ls -la $CTX
docker build -f $CTX/cert12.Dockerfile \
  --build-arg FORK_SHA=$FORK_SHA \
  --build-arg BUILT=$(date -u +%Y%m%dT%H%M%SZ) \
  -t $IMG_NEW $CTX > $LEG/cert12_build.log 2>&1 \
  || { echo "FATAL: cert12 build failed"; tail -30 $LEG/cert12_build.log; exit 1; }
tail -5 $LEG/cert12_build.log
docker run --rm --entrypoint cat $IMG_NEW /etc/vllm-arc/IMAGE_MANIFEST \
  > $LEG/cert12_manifest.txt
grep -q "^cert12=P6.28" $LEG/cert12_manifest.txt \
  || { echo "FATAL: cert12 manifest row missing"; cat $LEG/cert12_manifest.txt; exit 1; }
echo "MANIFEST_OK"

# --- [3] PATCHED (cert12): gate + timings + numeric compares ---------------
echo "=== [3] patched cert12 gate + points + numerics $(date +%H:%M:%S) ==="
boot_micro $IMG_NEW

docker exec arc-micro /opt/venv/bin/python3 -c "
import vllm.v1.attention.ops.triton_unified_attention as m
src = open(m.__file__, encoding='utf-8').read()
assert src.count('[ARC-P628-XPU-TUNE]') == 3, 'mark count'
assert m._ARC_XPU_PREFILL_CONFIGS[256][2] == (64, 64, 8, None), 't256'
assert m._ARC_XPU_PREFILL_CONFIGS[512][2] == (64, 32, 8, None), 't512'
assert m._arc_xpu_prefill_config(512, 8192) == (64, 32, 8, None)
assert m._arc_xpu_prefill_config(256, 1024) == (64, 64, 8, None)
assert m._arc_xpu_prefill_config(512, 8) is None
assert m._arc_xpu_prefill_config(128, 8192) is None
assert m._ARC_XPU_TUNE is True
print('GATE_OK patch live, table correct, xpu-gated')
" > $LEG/gate.txt 2>&1 || { cat $LEG/gate.txt; echo "FATAL: gate"; exit 1; }
cat $LEG/gate.txt

docker exec -i arc-micro /opt/venv/bin/python3 - < /dev/null > $LEG/patched_points.txt 2>&1 << 'PY2'
import json, subprocess, sys

PY = sys.executable
CHILD = r'''
import json, sys, time, torch
cfg = json.loads(sys.argv[1])
dev = "xpu"
torch.manual_seed(0)
head = cfg["head"]; nqh = cfg["nqh"]; nkvh = cfg["nkvh"]
ntok = cfg["ntok"]; seqlen = cfg["seqlen"]; bs = 64
win = tuple(cfg["win"])
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
call()
if cfg.get("dump"):
    torch.save(out[:256].to(torch.float32).cpu(), cfg["dump"])
    print("DUMP_OK " + cfg["dump"])
else:
    n = cfg.get("repeat", 2)
    t0 = time.perf_counter()
    for _ in range(n):
        call()
    dt = (time.perf_counter() - t0) / n
    print("SYNC_OK wall=%.4fs x%d" % (dt, n))
'''

def run(name, cfg, timeout=240):
    try:
        r = subprocess.run([PY, "-c", CHILD, json.dumps(cfg)],
                           capture_output=True, text=True, timeout=timeout)
        status = r.stdout.strip().splitlines()[-1] if r.stdout.strip() \
            else f"ERR {r.stdout[-200:]} {r.stderr[-400:]}"
    except subprocess.TimeoutExpired:
        status = "WEDGE (timeout %ds)" % timeout
    print("[%s] %s" % (name, status), flush=True)

F5 = dict(head=512, nqh=16, nkvh=8, ntok=1024, seqlen=8192, win=(-1, -1))
S2 = dict(head=256, nqh=16, nkvh=8, ntok=1024, seqlen=8192, win=(1023, 0))
F5H = {**F5, "ntok": 8192}
S2H = {**S2, "ntok": 8192}

run("patched-dump-f5-1k", {**F5, "dump": "/phaseC/patched_f5_1k.pt"})
run("patched-dump-f5-8k", {**F5H, "dump": "/phaseC/patched_f5_8k.pt"})
run("patched-dump-s2-8k", {**S2H, "dump": "/phaseC/patched_s2_8k.pt"})
run("patched-f5-8k", {**F5H, "repeat": 3})
run("patched-s2-8k", {**S2H, "repeat": 3})
run("patched-f5-kv16k", {**F5H, "seqlen": 16384, "repeat": 2}, timeout=360)
run("patched-s2-kv16k", {**S2H, "seqlen": 16384, "repeat": 2}, timeout=360)

import torch
for tag in ("f5_1k", "f5_8k", "s2_8k"):
    a = torch.load("/phaseC/stock_%s.pt" % tag)
    b = torch.load("/phaseC/patched_%s.pt" % tag)
    d = (a - b).abs()
    rel = d.max() / a.abs().max().clamp_min(1e-6)
    print("NUMERIC[%s] max_abs=%.5f rel_vs_peak=%.5f allclose(1e-2)=%s"
          % (tag, d.max(), rel, torch.allclose(a, b, atol=1e-2)), flush=True)
print("PATCHED_DONE", flush=True)
PY2
cat $LEG/patched_points.txt
grep -q "PATCHED_DONE" $LEG/patched_points.txt \
  || { echo "FATAL: patched section incomplete"; exit 1; }
grep -c "NUMERIC\[.*allclose(1e-2)=True" $LEG/patched_points.txt \
  | grep -q "^3$" || { echo "FATAL: numeric compare failed"; exit 1; }

echo "=== [r108l] window done $(date +%H:%M:%S) ==="
