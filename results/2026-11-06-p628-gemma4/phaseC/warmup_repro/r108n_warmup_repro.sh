#!/bin/bash
# r108n_warmup_repro.sh — P6.28 root-cause of the cert12 engine wedge.
#
# r108m: cert12 L1 boot reached health, then the FIRST request
# (arc-warmup-small, prompt_token_ids_len=98) wedged execute_model ->
# "RPC call to sample_tokens timed out" at 300 s. Cert11 same posture
# passed the same warmup (r108 phase-B G1 green). Delta = the patched
# launcher. 98 tokens: max_seqlen_q=98 > 1 -> 2D path (wheel's use_3d
# excludes prefill), band 65-4095 -> table config (BLOCK_M=64,
# BLOCK_Q=32) -> total_num_q_blocks = 98//32+1 = 4 with a PARTIAL TAIL
# (98 = 3x32 + 2). The r108k sweep only tested exact multiples of
# BLOCK_Q (ntok 1024/8192) — the partial-tail path at BLOCK_M=64 was
# never exercised.
#
# [1] cert12 (baked table): exact warmup shape both head classes +
#     controls at 1024. WEDGE (subprocess timeout 120 s) = repro.
# [2] cert11 + r108k env hook: ntok=98 config matrix to find the
#     largest config that survives small partial-tail shapes.
# Trap restores W14. No rebuild in this window — data decides the guard.
set -uo pipefail
R=/root/build
CT=$R/p628/phaseC
LEG=$CT/warmup_repro
mkdir -p $LEG

if pgrep -f "snapshot_download" >/dev/null 2>&1; then
  echo "FATAL: host model download in flight"; exit 1
fi

w14_start() {
  docker rm -f arc-micro >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_serve_only.sh > $R/w14_restore_p628n.log 2>&1 < /dev/null &
}
trap w14_start EXIT

WH=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health 2>/dev/null || true)
echo "W14 health before stop: $WH"
[ "$WH" = "200" ] || { echo "FATAL: W14 not healthy — restore first"; exit 1; }
docker stop radiance-test >/dev/null 2>&1 || true
sleep 3

boot_micro() {  # $1 = image
  docker rm -f arc-micro >/dev/null 2>&1 || true
  docker run -d --name arc-micro \
    --privileged --network host --ipc host \
    --shm-size 64g \
    -v /root/build/triton_cache:/root/.triton_cache \
    --entrypoint /bin/bash "$1" -c 'sleep infinity' >/dev/null
  sleep 5
}

CHILD_BODY='
import json, sys, os, time, torch
cfg = json.loads(sys.argv[1])
dev = "xpu"
torch.manual_seed(0)
head = cfg["head"]; nqh = cfg["nqh"]; nkvh = cfg["nkvh"]
ntok = cfg["ntok"]; seqlen = cfg["seqlen"]; bs = 64
win = tuple(cfg["win"])
nblk = (seqlen + bs - 1) // bs
kdt = torch.float8_e4m3fn
q = torch.randn(ntok, nqh, head, dtype=torch.float16, device=dev)
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
n = cfg.get("repeat", 2)
t0 = time.perf_counter()
for _ in range(n):
    call()
dt = (time.perf_counter() - t0) / n
print("SYNC_OK wall=%.4fs x%d" % (dt, n))
'

run_suite() {  # $1 = outfile; reads point specs from stdin as
               # "name|json|envjson" lines
  docker exec -i arc-micro /opt/venv/bin/python3 - < /dev/null >> "$1" 2>&1 << PYRUN
import json, subprocess, sys, os

PY = sys.executable
CHILD = r'''${CHILD_BODY}'''

def run(name, cfg, timeout=120, env_extra=None):
    env = dict(os.environ)
    for k in ("ARC_SWEEP_BLOCK_M", "ARC_SWEEP_WARPS", "ARC_SWEEP_STAGES", "ARC_SWEEP_TILE"):
        env.pop(k, None)
    if env_extra:
        env.update(env_extra)
    try:
        r = subprocess.run([PY, "-c", CHILD, json.dumps(cfg)],
                           capture_output=True, text=True, timeout=timeout, env=env)
        status = r.stdout.strip().splitlines()[-1] if r.stdout.strip() \
            else "DIED_PRE_PRINT " + (r.stderr[-300:] if r.stderr else "")
    except subprocess.TimeoutExpired:
        status = "WEDGE (timeout %ds)" % timeout
    print("[%s] %s" % (name, status), flush=True)

F5 = dict(head=512, nqh=16, nkvh=8, ntok=1024, seqlen=8192, win=(-1, -1))
S2 = dict(head=256, nqh=16, nkvh=8, ntok=1024, seqlen=8192, win=(1023, 0))

# --- [1] cert12 baked-table: exact warmup shape + controls ---
print("=== cert12 baked table: warmup-98 repro + 1024 controls ===", flush=True)
run("c12-w98-F5", {**F5, "ntok": 98, "seqlen": 98})
run("c12-w98-S2", {**S2, "ntok": 98, "seqlen": 98})
run("c12-ctrl1024-F5", dict(F5))
run("c12-ctrl1024-S2", dict(S2))
run("c12-w8192-F5", {**F5, "ntok": 8192, "repeat": 1}, timeout=120)
print("CERT12_REPRO_DONE", flush=True)
PYRUN
}

echo "=== [1] cert12 repro $(date +%H:%M:%S) ==="
boot_micro vllm-arc:v0.30.0-b70-cert12
run_suite $LEG/cert12_repro.txt
cat $LEG/cert12_repro.txt
grep -q "CERT12_REPRO_DONE" $LEG/cert12_repro.txt || { echo "FATAL: cert12 section incomplete"; exit 1; }
docker rm -f arc-micro >/dev/null

echo "=== [2] cert11 hook matrix at ntok=98 $(date +%H:%M:%S) ==="
boot_micro vllm-arc:v0.30.0-b70-cert11
docker exec -i arc-micro /opt/venv/bin/python3 - < /dev/null > $LEG/hook_install.txt 2>&1 << 'PYHOOK'
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
grep -q "HOOK_INSTALLED\|HOOK_ALREADY_PRESENT" $LEG/hook_install.txt \
  || { echo "FATAL: hook install failed"; cat $LEG/hook_install.txt; exit 1; }

docker exec -i arc-micro /opt/venv/bin/python3 - < /dev/null > $LEG/matrix98.txt 2>&1 << 'PYMAT'
import json, subprocess, sys, os

PY = sys.executable
CHILD = r'''
import json, sys, os, time, torch
cfg = json.loads(sys.argv[1])
dev = "xpu"
torch.manual_seed(0)
head = cfg["head"]; nqh = cfg["nqh"]; nkvh = cfg["nkvh"]
ntok = cfg["ntok"]; seqlen = cfg["seqlen"]; bs = 64
win = tuple(cfg["win"])
nblk = (seqlen + bs - 1) // bs
kdt = torch.float8_e4m3fn
q = torch.randn(ntok, nqh, head, dtype=torch.float16, device=dev)
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
n = cfg.get("repeat", 2)
t0 = time.perf_counter()
for _ in range(n):
    call()
dt = (time.perf_counter() - t0) / n
print("SYNC_OK wall=%.4fs x%d" % (dt, n))
'''

def run(name, cfg, timeout=120, env_extra=None):
    env = dict(os.environ)
    for k in ("ARC_SWEEP_BLOCK_M", "ARC_SWEEP_WARPS", "ARC_SWEEP_STAGES", "ARC_SWEEP_TILE"):
        env.pop(k, None)
    if env_extra:
        env.update(env_extra)
    try:
        r = subprocess.run([PY, "-c", CHILD, json.dumps(cfg)],
                           capture_output=True, text=True, timeout=timeout, env=env)
        status = r.stdout.strip().splitlines()[-1] if r.stdout.strip() \
            else "DIED_PRE_PRINT " + (r.stderr[-300:] if r.stderr else "")
    except subprocess.TimeoutExpired:
        status = "WEDGE (timeout %ds)" % timeout
    print("[%s] %s" % (name, status), flush=True)

F5 = dict(head=512, nqh=16, nkvh=8, ntok=98, seqlen=98, win=(-1, -1))
S2 = dict(head=256, nqh=16, nkvh=8, ntok=98, seqlen=98, win=(1023, 0))

print("=== head-512 @ ntok=98 (tail 98=3x32+2 / 12x8+2) ===", flush=True)
run("base-M16w4T32", dict(F5))
run("M32w4T32", dict(F5), env_extra={"ARC_SWEEP_BLOCK_M": "32"})
run("M32w8T32", dict(F5), env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_WARPS": "8"})
run("M64w8T32", dict(F5), env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8"})
print("=== head-256 @ ntok=98 ===", flush=True)
run("base-M16w4T32", dict(S2))
run("M32w8T32", dict(S2), env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_WARPS": "8"})
run("M64w8T32", dict(S2), env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8"})
run("M32w8T64", dict(S2), env_extra={"ARC_SWEEP_BLOCK_M": "32", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64"})
run("M64w8T64", dict(S2), env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8", "ARC_SWEEP_TILE": "64"})
print("=== tail-shape variants @ head-512 M64w8T32 (is ANY partial tail OK?) ===", flush=True)
run("M64w8T32-ntok97", {**F5, "ntok": 97, "seqlen": 97},
    env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8"})
run("M64w8T32-ntok96", {**F5, "ntok": 96, "seqlen": 96},
    env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8"})
run("M64w8T32-ntok8222", {**F5, "ntok": 8222, "seqlen": 8222, "repeat": 1},
    env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8"}, timeout=180)
run("M64w8T32-tdoff", dict(F5),
    env_extra={"ARC_SWEEP_BLOCK_M": "64", "ARC_SWEEP_WARPS": "8", "VLLM_TRITON_USE_TD": "0"})
print("MATRIX_DONE", flush=True)
PYMAT
cat $LEG/matrix98.txt
grep -q "MATRIX_DONE" $LEG/matrix98.txt || { echo "FATAL: matrix incomplete"; exit 1; }

echo "=== [r108n] window done $(date +%H:%M:%S) ==="
