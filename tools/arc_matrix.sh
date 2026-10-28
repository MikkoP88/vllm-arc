#!/bin/bash
# arc_matrix.sh — vllm-arc leg driver (pattern = proven p2r6_ckptfp8 driver):
# stage -> boot -> gates -> bench -> counts -> VERDICT. One uniform harness
# law (plan §10): every leg archives env.txt + summary.txt + VERDICT.md.
#
# P1 scope (plan §10 P1 row): eager serve of qwen3.8-27b-fp8, no boot death,
# ArcXpuPlatform resolved, mem-info patch live, k1k sanity cells.
#
# Usage (rig): ARC_LEG=p1-platform bash arc_matrix.sh
set -euo pipefail

ARC_LEG=${ARC_LEG:-p1-platform}
ARC_IMAGE=${ARC_IMAGE:-vllm-arc:v0.30.0-b70}
ARC_NAME=${ARC_NAME:-arc-test}
LEG=/root/build/legs/arc-$ARC_LEG-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LEG"
PLEG=$LEG

say() { echo "[arc-$ARC_LEG $(date +%H:%M:%S)] $*"; }
fail() { echo "[arc-$ARC_LEG FAIL] $*"; exit 1; }

# ---------- 0. pre-gates ----------
say "pre-gates: image present, harness present"
docker image inspect "$ARC_IMAGE" >/dev/null 2>&1 || fail "image $ARC_IMAGE missing"
for f in /root/build/arc_matrix.sh /root/build/arc_pybench.py /root/build/bench_run7n.py; do
  [ -f "$f" ] || fail "harness missing ($f)"
done

# ---------- 1. boot (default P1 posture: eager, MTP off, no LMCache) ----------
# ARC_LEG_GRAPH/ARC_LEG_SPEC/ARC_LEG_LMC override the posture for later-phase
# legs (p30: SPEC is a P2-candidate posture — no override needed; a nonzero
# LMC still auto-sets ARC_ALLOW_UNVALIDATED=1 per the repro law: the serve
# script refuses unvalidated cache postures unless repro is explicit).
say "boot: serve qwen3.8-27b-fp8 TP2 (graph=${ARC_LEG_GRAPH:-0} spec=${ARC_LEG_SPEC:-0} lmc=${ARC_LEG_LMC:-0})"
if [ "${ARC_LEG_LMC:-0}" != 0 ]; then
  export ARC_ALLOW_UNVALIDATED=1
fi
ARC_GRAPH=${ARC_LEG_GRAPH:-0} ARC_SPEC=${ARC_LEG_SPEC:-0} ARC_LMC=${ARC_LEG_LMC:-0} bash /root/build/serve_arc_b70.sh 2>&1 | tee "$PLEG/boot.txt"

say "waiting for engine health (max 30 min)"
ok=0
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:8000/health" >/dev/null 2>&1; then ok=1; break; fi
  if docker exec "$ARC_NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    docker exec "$ARC_NAME" cat /root/serve_full.log > "$PLEG/serve_death.log" 2>/dev/null || true
    fail "boot death — engine log (full copy: $PLEG/serve_death.log):"
    docker exec "$ARC_NAME" tail -50 /root/serve_full.log || true
  fi
  sleep 10
done
[ "$ok" = 1 ] || fail "health timeout (30 min)"
say "engine healthy"

docker exec "$ARC_NAME" cp /root/serve_full.log /root/serve_p1.log 2>/dev/null || true

# ---------- 2. gates ----------
say "gates: platform + mem-info + posture + model identity"
PLAT=$(docker exec "$ARC_NAME" /opt/venv/bin/python -c \
  "import vllm_gaudi.xpu.platform as p; print(p.__name__)" 2>&1) || fail "xpu platform module unimportable: $PLAT"

RES=$(docker exec -i "$ARC_NAME" /opt/venv/bin/python - <<'EOF'
import torch, vllm
from vllm.platforms import current_platform
name = type(current_platform).__name__
mod = type(current_platform).__module__
free, total = torch.accelerator.get_memory_info(0)
xe2 = None
try:
    from vllm_gaudi.xpu.platform import is_xe2_arch
    xe2 = is_xe2_arch()
except Exception as e:
    xe2 = f"ERR {e}"
print(f"PLATFORM={mod}.{name}")
print(f"XE2={xe2}")
print(f"MEMINFO free={free/2**30:.2f}GiB total={total/2**30:.2f}GiB nonzero={free>0}")
print(f"DEVICE={torch.xpu.get_device_name(0)} count={torch.xpu.device_count()}")
EOF
) || fail "gate probe crashed: $RES"
echo "$RES" | tee "$PLEG/gates.txt"

echo "$RES" | grep -q "PLATFORM=vllm_gaudi.xpu.platform.ArcXpuPlatform" || fail "ArcXpuPlatform not resolved: $RES"
echo "$RES" | grep -q "XE2=True" || fail "is_xe2_arch returned non-True: $RES"
echo "$RES" | grep -q "MEMINFO.*nonzero=True" || fail "mem-info patch not live (free==0)"

docker exec "$ARC_NAME" sh -c "grep -aq 'ArcXpuPlatform: Intel Arc Xe2 posture active' /root/serve_full.log" \
  || fail "Xe2 posture line absent from serve log"
docker exec "$ARC_NAME" sh -c "grep -aq 'qwen3.8-27b-fp8' /root/serve_full.log" \
  || fail "served-model-name absent from serve log (model identity probe hygiene law)"

# model identity (probe hygiene law: always verify the model inside a probe)
MODEL_ID=$(docker exec "$ARC_NAME" /opt/venv/bin/python -c \
  "import json; c=json.load(open('/models/target/config.json')); print(c.get('architectures'), c.get('quantization_config',{}).get('quant_method'))" 2>&1) \
  || fail "model config unreadable: $MODEL_ID"
echo "MODEL_ID $MODEL_ID" | tee -a "$PLEG/gates.txt"

# ---------- 3. k1k sanity bench (subset; full suite from P2) ----------
say "bench: k1k sanity (1024 ctx, c1/c4/c8 warm pass)"
docker cp /root/build/arc_pybench.py "$ARC_NAME":/root/arc_pybench.py
docker exec "$ARC_NAME" /opt/venv/bin/python /root/arc_pybench.py 1024 1 8 8 2>&1 | tail -20 | tee "$PLEG/k1k_sanity.txt" || true

# ---------- 3b. p31 spec-acceptance gate (every leg gated) ----------
# Spec legs: pos1 hard >= 0.5 (wrong-logits family detector), overall
# acceptance vs 0.85 contract target. Non-spec legs: expect-off posture
# purity (spec counters must be ABSENT). Plan §10 P2 row: "acceptance
# telemetry live (p31 gate wired, every leg gated)".
if [ "${ARC_LEG_SPEC:-0}" = 1 ]; then
  say "p31 gate: spec acceptance (pos1 hard >=0.5, target 0.85)"
  python3 /root/build/arc_spec_gate.py --json-out "$PLEG/spec_gate.json" 2>&1 | tee "$PLEG/spec_gate.txt"
else
  say "p31 gate: expect-off (non-spec leg posture purity)"
  python3 /root/build/arc_spec_gate.py --expect-off 2>&1 | tee "$PLEG/spec_gate.txt"
fi

# ---------- 4. env + summary ----------
docker exec "$ARC_NAME" sh -c "tr '\0' '\n' < /proc/1/environ" > "$PLEG/env.txt" 2>/dev/null || true
docker exec "$ARC_NAME" sh -c "ps aux | grep '[v]llm serve'" > "$PLEG/ps.txt" 2>/dev/null || true
docker exec "$ARC_NAME" sh -c "grep -aE 'ArcXpuPlatform|memory|XPU Graph|cudagraph|speculative|model=' /root/serve_full.log" > "$PLEG/posture.log" 2>/dev/null || true

{
  echo "leg=$ARC_LEG image=$ARC_IMAGE model=qwen3.8-27b-fp8 tp=2 graph=${ARC_LEG_GRAPH:-0} spec=${ARC_LEG_SPEC:-0} lmc=${ARC_LEG_LMC:-0} gmu=0.8 mnbt=8192 seqs=64"
  echo "gates: platform=PASS xe2=PASS meminfo=PASS modelid=$MODEL_ID"
} > "$PLEG/summary.txt"
say "leg artifacts: $PLEG"
echo "ARC_LEG_DONE $PLEG"
