#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# r105b_cert11_final.sh — P6.26 final-image gate: cert11 (OVERLAY on
# cert10 via tools/cert11.Dockerfile; see r105 RUN NOTE for why cert11c
# the full rebuild is the engine-validation vehicle, not the shipped
# artifact). Runs after the tag exists and the overlay image is built:
#
#   env TAG_SHA=<rel/v0.30.0-b70-cert11 commit sha>  (required)
#   image under test: vllm-arc:v0.30.0-b70-cert11
#
#   [0] content gates: manifest cert11_fork_sha == TAG_SHA + oneccl rows
#       carried from cert10; exhaustive vllm_gaudi/ md5 walk vs cert10 =
#       EXACTLY platform.py + its regenerated pyc; oneccl overlay sha ==
#       cert10 pin at both fat-lib paths; thin shim identical; pip freeze
#       diff == 0 (overlay carries the certified venv byte-identical);
#       fix_live import proof (quark + supports_mx).
#   [1] spot boot, default certified posture (fp8 target, knob off):
#       posture lines, flip-first, spec gate, erraudit, serve log bank.
#       (Full release battery lives on cert11c — r105 [1]; this content-
#       identical image needs the direct boot proof only.)
#
# W14 stopped for the window, restored by trap (rig law). r84 kill_arc
# retry law on every container transition.
set -uo pipefail
R=/root/build
CT=$R/p626/r105b
IMG10=vllm-arc:v0.30.0-b70-cert10
IMG=vllm-arc:v0.30.0-b70-cert11
FAT_SHA=3a120e8b352fbd6be45ab61a7f4e927d80c9bc6618ff646e6db68b0263eca4a7
SP=/opt/venv/lib/python3.12/site-packages
GATE_FAIL=0

[ -n "${TAG_SHA:-}" ] || { echo "FATAL: set TAG_SHA=<cert11 tag commit sha>"; exit 1; }

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p626d.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 20

mkdir -p $CT
cd $CT || exit 1

kill_arc() {  # r84 retry law
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1 || true
  sleep 10
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1 || true
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 10
  docker inspect arc-test >/dev/null 2>&1 && { echo "FATAL: arc-test undeletable"; exit 1; }
}

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

echo "=================== P626 R105B CERT11 FINAL GATE START $(date +%F\ %H:%M:%S) TAG_SHA=$TAG_SHA ==================="

echo "=== [0] content-identity gates on $IMG $(date +%H:%M:%S) ==="
docker image inspect $IMG >/dev/null 2>&1 || { echo "FATAL: image $IMG absent"; exit 1; }
docker run --rm --entrypoint cat $IMG /etc/vllm-arc/IMAGE_MANIFEST > manifest_cert11.txt
grep -q "cert11_fork_sha=$TAG_SHA" manifest_cert11.txt || { echo "MANIFEST cert11_fork_sha: FAIL (want $TAG_SHA)"; cat manifest_cert11.txt; GATE_FAIL=1; }
grep -q "oneccl=2022.0.0+arc1" manifest_cert11.txt || { echo "MANIFEST oneccl row (carried): FAIL"; GATE_FAIL=1; }
grep -q "oneccl_lib_sha256=$FAT_SHA" manifest_cert11.txt || { echo "MANIFEST oneccl lib sha row: FAIL"; GATE_FAIL=1; }
echo "MANIFEST: cert11_fork_sha + carried oneccl rows checked"
docker run --rm --entrypoint bash $IMG10 -c \
  "cd $SP && find vllm_gaudi -type f | sort | xargs md5sum" > tree_cert10.md5
docker run --rm --entrypoint bash $IMG   -c \
  "cd $SP && find vllm_gaudi -type f | sort | xargs md5sum" > tree_cert11.md5
CH10=$(wc -l < tree_cert10.md5); CH11=$(wc -l < tree_cert11.md5)
NDEL=$(diff tree_cert10.md5 tree_cert11.md5 | grep -c '^<' || true)
NADD=$(diff tree_cert10.md5 tree_cert11.md5 | grep -c '^>' || true)
CHG=$(diff tree_cert10.md5 tree_cert11.md5 | grep '^>' | awk '{print $2}' | sort | tr '\n' ' ')
echo "TREE vllm_gaudi/: files $CH10 -> $CH11, changed=$NDEL"
echo "  changed set: $CHG"
if [ "$NDEL" = "2" ] && [ "$NADD" = "2" ] && [ "$CH10" = "$CH11" ] && \
   [ "$CHG" = "vllm_gaudi/xpu/__pycache__/platform.cpython-312.pyc vllm_gaudi/xpu/platform.py " ]; then
  echo "TREE IDENTITY: OK (platform.py + its regenerated pyc only — cert10 carry basis)"
else
  echo "TREE IDENTITY: FAIL (unexpected delta)"; diff tree_cert10.md5 tree_cert11.md5 | head; GATE_FAIL=1
fi
docker run --rm --entrypoint python $IMG -c \
  "from vllm_gaudi.xpu.platform import ArcXpuPlatform as P; \
   print('FIX_LIVE quark=%s supports_mx=%s' % ('quark' in P.supported_quantization, P.supports_mx()))" \
  > fix_live.txt 2>&1 || true
grep -q "FIX_LIVE quark=True supports_mx=True" fix_live.txt \
  || { echo "FIX_LIVE: FAIL"; cat fix_live.txt; GATE_FAIL=1; }
for P in /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so.1.0; do
  GOT=$(docker run --rm --entrypoint sha256sum $IMG $P | awk '{print $1}')
  [ "$GOT" = "$FAT_SHA" ] && echo "OVERLAY SHA OK  $P" \
    || { echo "OVERLAY SHA FAIL $P ($GOT)"; GATE_FAIL=1; }
done
docker run --rm --entrypoint bash $IMG10 -c "md5sum /opt/venv/lib/libccl.so.2.0" > thin_cert10.md5
docker run --rm --entrypoint bash $IMG   -c "md5sum /opt/venv/lib/libccl.so.2.0" > thin_cert11.md5
cmp -s thin_cert10.md5 thin_cert11.md5 && echo "THIN SHIM: identical (untouched)" \
  || { echo "THIN SHIM: FAIL (changed)"; GATE_FAIL=1; }
docker run --rm --entrypoint bash $IMG10 -c "/opt/venv/bin/pip freeze" | sort > freeze_cert10.txt
docker run --rm --entrypoint bash $IMG   -c "/opt/venv/bin/pip freeze" | sort > freeze_cert11.txt
NFZ=$(diff freeze_cert10.txt freeze_cert11.txt | wc -l)
echo "PIP FREEZE diff lines vs cert10: $NFZ (expect 0)"
[ "$NFZ" = "0" ] || { echo "FREEZE: FAIL"; diff freeze_cert10.txt freeze_cert11.txt | head; GATE_FAIL=1; }

echo "=== [1] spot boot, default certified posture $(date +%H:%M:%S) ==="
LEG=$CT/spot
mkdir -p $LEG
kill_arc
( cd $LEG && ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=0 \
    bash $R/serve_arc_b70.sh > boot.out 2>&1 )
grep ARC_SERVE_LAUNCHED $LEG/boot.out || { echo "FATAL: launch failed"; cat $LEG/boot.out; exit 1; }
wait_health | tee -a $LEG/boot.out
docker exec arc-test sh -c "grep -E 'KV cache size|Xe2 posture active|graph posture' /root/serve_full.log | head -3" > $LEG/posture.txt || true
cat $LEG/posture.txt
python3 - > $LEG/flip_first.out 2>&1 << 'PYEOF' || true
import json, time, urllib.request
body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
        "max_tokens": 832, "messages": [{"role": "user", "content": "Count from 1 to 20."}]}
t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=600))
c = r["choices"][0]
txt = c["message"].get("content") or c["message"].get("reasoning_content") or ""
print("FLIP_FIRST:", c["finish_reason"], f"{time.perf_counter()-t0:.1f}s", "head:", repr(txt[:100]))
PYEOF
cat $LEG/flip_first.out
python3 $R/arc_spec_gate.py > $LEG/spec_gate.out 2>&1
tail -3 $LEG/spec_gate.out
grep -q "SPEC_GATE PASS" $LEG/spec_gate.out || { echo "SPEC_GATE: FAIL"; GATE_FAIL=1; }
docker exec arc-test sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
  > $LEG/erraudit.txt 2>/dev/null || echo "ERROR=?" > $LEG/erraudit.txt
cat $LEG/erraudit.txt
grep -q "ERROR=0" $LEG/erraudit.txt || { echo "ERRAUDIT: FAIL"; GATE_FAIL=1; }
docker exec arc-test cat /root/serve_full.log > $LEG/serve_full.log 2>/dev/null || true

echo "=================== P626 R105B COMPLETE $(date +%F\ %H:%M:%S) GATE_FAIL=$GATE_FAIL ==================="
exit $GATE_FAIL
