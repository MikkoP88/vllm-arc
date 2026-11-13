#!/bin/bash
# r168_lmc_disc.sh -- P6.47 discriminator: is the m4q essay-rehit
# REP-LOOP an LMCache double-quant artifact or an inherent e2m1-V
# boundary?
#
# r161 banked m4q (mxfp4, ARC_LMC=1) with ZERO-GARBLE FAIL on
# full+rehit1..3 (essay-class repetition to length cap, pre-existing
# across cert24/25/25y, byte-identical rehit 484f71e94d10); m8q
# (e4m3/e4m3) is clean -> the defect isolates to e2m1 V. Hypothesis A:
# the LMCache prefix-reload path re-quantizes V (e2m1 twice = coarse;
# e4m3 tolerates). Hypothesis B: inherent 4-bit-V quality boundary on
# the rehit (cache-hit) decode path.
#
# This leg reboots the EXACT m4q posture with ARC_LMC=0 (native
# prefix-cache only) on the same image (cert25y) and runs the same
# face sequence: longprobe full -> churn 900 -> longprobe rehit.
#   clean  -> Hypothesis A: posture ARC_LMC auto-off for mxfp4/mxk8v4,
#             evidence-cited (this bank).
#   loop   -> Hypothesis B: inherent boundary; m4/mkv4 essay-rehit
#             gates comparatively (like gemma) + banked boundary note.
# Absolute qwen garble gate on BOTH faces (qwen absolute law). trap
# restores W14. Artifacts -> ${P647_BANK:-/root/build/p647/lmcdisc}/.
set -uo pipefail
R=/root/build
P=${P647_BANK:-$R/p647/lmcdisc}
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25y}
CNAME=arc-c24
SERVED=qwen3.8-27b-fp8
mkdir -p "$P" "$R/p647/gatelog"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p168.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

# refuse a stale image (same content markers as r161)
IMG_OK=$(docker run --rm -i --entrypoint /opt/venv/bin/python "$IMG" - <<'PYEOF' 2>/dev/null
import re
from vllm_gaudi.xpu.patches.mx_kv import engine
from vllm_gaudi.xpu.patches.mx_kv import triton_mx_decode as t
tsrc = open(t.__file__).read()
esrc = open(engine.__file__).read()
ok = "tl.maximum(1, n_tiles)" in tsrc
ok = ok and not re.search(r"query\.float\(\)", esrc)
ok = ok and "_split_verify_decode" in esrc
print("IMG_CONTENT_OK" if ok else "IMG_CONTENT_STALE")
PYEOF
) || IMG_OK=""
echo "$IMG_OK" | grep -q "IMG_CONTENT_OK" \
  || { echo "R168_STALE_IMAGE IMG=$IMG"; exit 1; }

# derived longprobe with served-model env override (r161 derivation;
# p618 hardcodes the qwen name -- one-line edit, provenance kept)
mkdir -p $R/p647
python3 - << 'PYEOF'
src = open("/root/build/p618/p618_longprobe.py", encoding="utf-8").read()
if "ARC_PROBE_MODEL" not in src:
    assert src.count('MODEL = "qwen3.8-27b-fp8"') == 1, "p618 model line drifted"
    src = src.replace('MODEL = "qwen3.8-27b-fp8"',
                      'import os\nMODEL = os.environ.get("ARC_PROBE_MODEL", "qwen3.8-27b-fp8")')
    open("/root/build/p647/p647_longprobe.py", "w", encoding="utf-8").write(src)
print("p647_longprobe.py derived OK")
PYEOF

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect $CNAME >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[disc] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(docker exec $CNAME grep -c "Traceback (most recent call last)" /root/serve_full.log 2>/dev/null)
      [ "${tb:-0}" -gt 0 ] && { echo "[disc] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[disc] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

garble_count() {  # $1 = longprobe out: essay-class garble dict emptiness
  local lines empty
  lines=$(grep -cE "^\[(essay|code|multi|simple)\]" "$1" || true)
  empty=$(( $(grep -cF "garble={'cold': [], 'rehit': [], 'rehit2': []}" "$1" || true)
          + $(grep -cF "garble={'r1': [], 'r2': []}" "$1" || true) ))
  echo "GARBLE: class_lines=$lines empty_garble=$empty in $(basename "$1")"
  [ "$lines" -ge 4 ] && [ "$lines" = "$empty" ]
}

echo "=== [r168] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
docker rm -f $CNAME >/dev/null 2>&1 || true
sleep 5

echo "=== [r168] boot m4q ARC_LMC=0 $(date +%H:%M:%S) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=$SERVED \
  ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 ARC_FP8=1 ARC_KVDTYPE=mxfp4 \
  bash $R/serve_arc_b70.sh > "$P/boot.txt" 2>&1 &
if ! boot_wait; then
  echo "R168_BOOT_FAIL"
  docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME tail -40 /root/serve_full.log 2>/dev/null
  exit 1
fi
echo "HEALTH_OK_disc $(date +%H:%M:%S)"
{
  echo "lmcache_lines=$(docker exec $CNAME grep -c 'LMCacheConnector' /root/serve_full.log 2>/dev/null || echo 0)"
  echo "mx_backend_lines=$(docker exec $CNAME grep -c 'Using MX KV attention backend' /root/serve_full.log 2>/dev/null || echo 0)"
  echo "tracebacks=$(docker exec $CNAME grep -c 'Traceback (most recent call last)' /root/serve_full.log 2>/dev/null || echo 0)"
} > "$P/markers.txt"
cat "$P/markers.txt"
grep -qx "lmcache_lines=0" "$P/markers.txt" \
  || echo "R168_WARN_LMC_STILL_PRESENT (discriminator invalid)"

curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
sleep 3

echo "=== [r168] longprobe full $(date +%H:%M:%S) ==="
( cd "$P" && ARC_PROBE_MODEL=$SERVED python3 $R/p647/p647_longprobe.py r168disc full > longprobe_full.out 2>&1 )
grep -E "^\[|LONGPROBE" "$P/longprobe_full.out"
V1=0; garble_count "$P/longprobe_full.out" || V1=1

echo "=== [r168] churn 900 $(date +%H:%M:%S) ==="
python3 - "$SERVED" > "$P/churn_r168.out" 2>&1 << 'PYEOF' || true
import json, sys, time, urllib.request
m = sys.argv[1]
def post(p, mt=256):
    d = json.dumps({"model": m, "prompt": p, "max_tokens": mt,
                    "temperature": 0}).encode()
    r = urllib.request.Request("http://127.0.0.1:8000/v1/completions", d,
                               {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=600) as f:
        json.load(f)
    return time.time() - t0
t_end = time.time() + 900
i = 0
while time.time() < t_end:
    i += 1
    post(f"churn fill {i} " + "x" * 400)
print(f"churn rounds={i}")
PYEOF
tail -2 "$P/churn_r168.out"

echo "=== [r168] longprobe rehit $(date +%H:%M:%S) ==="
( cd "$P" && ARC_PROBE_MODEL=$SERVED python3 $R/p647/p647_longprobe.py r168disc rehit > longprobe_rehit1.out 2>&1 )
grep -E "^\[|LONGPROBE" "$P/longprobe_rehit1.out"
V2=0; garble_count "$P/longprobe_rehit1.out" || V2=1

docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 || echo "serve.log MISSING"
docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 8
docker rm -f $CNAME >/dev/null 2>&1

cd "$P"
find . -type f -exec sha256sum {} \; > SHA256SUMS 2>/dev/null || true
if [ "$V1" = 0 ] && [ "$V2" = 0 ]; then
  echo "LMC_DISC_CLEAN -- LMCache double-quant implicated; posture ARC_LMC auto-off for mxfp4/mxk8v4 (cite this bank)"
  exit 0
else
  echo "LMC_DISC_LOOP -- inherent e2m1-V boundary; m4/mkv4 essay-rehit gates comparatively (cite this bank)"
  exit 2
fi
