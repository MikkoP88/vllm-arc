#!/bin/bash
# r212i3_poison.sh -- arc7 Phase-8 r212i3: settle the forced-plan
# numerics with the NaN-poison + row-permutation probes (the
# stale-buffer canary law, applied instrumentally). r212i/r212i2
# established: forced-split cells canary > 2e-3 at EVERY mk/planlen
# combination, yet EXECUTE with real cost (s16/s64 == flat floor;
# 1K 301 / 8K 204 / 32K 706 us vs auto 598/423/1039 — forced-16
# beats the engine's auto heuristic 1.5-2.1x at every ctx). Canary
# moved with plan lens (0.109192@real -> 0.069855@cap at 32K) so
# outputs ARE written. Three surviving hypotheses:
#   H1 never-written/partial-write (NaN survives poison)
#   H2 row permutation (rowmatch off-diagonal 0)
#   H3 wrong values (no clean row anywhere)
# Cells P1-P3: ctx {1024,8192,32768}, SPLITS=auto,16, MAXSK=cap,
# PLANLEN=cap, POISON=1, ROWPROBE=1. Cell E: default-env oracle —
# must reproduce r212i_8192.json (canaries identical, timings
# within 15%) proving the env-gated edits changed nothing.
# Artifacts -> /root/build/p199/out/p212i3_<TAG>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212i3

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212I3_NO_WHEEL"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212I3_NO_HARNESS"; exit 1; }
grep -q "P204_POISON" "$HARNESS" || { echo "R212I3_STALE_HARNESS (no poison knob)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212I3_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212I3_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212I3_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212I3_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212i3.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212i3] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {  # tag extra_env...
  local TAG=$1; shift
  local extra=()
  local a
  for a in "$@"; do extra+=(-e "$a"); done
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r212i3:$TAG] $* $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    "${extra[@]}" \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212i3_$TAG.log 2>&1
  local rc=$?
  echo "=== [r212i3:$TAG] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime.json" ]; then
    mv "$P/out/p204_graphtime.json" "$P/out/p212i3_$TAG.json"
  else
    echo "R212I3_MISSING_ARTIFACT_$TAG"
  fi
}

run_cell P1_1k   P204_CTX=1024  P204_SPLITS=auto,16 P204_MAXSK=cap P204_PLANLEN=cap P204_POISON=1 P204_ROWPROBE=1
run_cell P2_8k   P204_CTX=8192  P204_SPLITS=auto,16 P204_MAXSK=cap P204_PLANLEN=cap P204_POISON=1 P204_ROWPROBE=1
run_cell P3_32k  P204_CTX=32768 P204_SPLITS=auto,16 P204_MAXSK=cap P204_PLANLEN=cap P204_POISON=1 P204_ROWPROBE=1
run_cell E_orc   P204_CTX=8192  P204_SPLITS=auto,5,16 P204_MAXSK=cap

echo "=== [r212i3] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0

def load(tag):
    f = os.path.join(out, f"p212i3_{tag}.json")
    return json.load(open(f)).get("mxfp4", {}) if os.path.exists(f) else None

print("r212i3 — poison/rowprobe on forced-16 (auto must stay clean)")
for tag, ctx in (("P1_1k", 1024), ("P2_8k", 8192), ("P3_32k", 32768)):
    d = load(tag)
    if d is None:
        print(f" {tag}: MISSING"); fail = 1; continue
    a, s = d.get("native_auto") or {}, d.get("native_16") or {}
    print(f" {tag} ctx={ctx}: auto {a.get('dev_us')} us canary {a.get('canary')}"
          f" | s16 {s.get('dev_us')} us canary {s.get('canary')}")
    if a.get("canary") is None or a["canary"] > 2e-3:
        print("  AUTO_CANARY_FAIL"); fail = 1
    nan = s.get("nan_left")
    refmx = s.get("ref_absmax")
    rm = s.get("rowmatch")
    verdict = "AUTO_ONLY"
    if nan is not None and nan > 0:
        verdict = f"NEVER_WRITTEN(nan_left={nan}/{2}*12*256)"
    elif rm is not None:
        n = len(rm)
        clean_off = any(rm[i][j] <= 2e-3 for i in range(n) for j in range(n) if i != j)
        verdict = "PERMUTED" if clean_off else "WRONG_VALUES"
    print(f"  s16 nan_left={nan} ref_absmax={refmx} rowmatch={rm} -> {verdict}")

e = load("E_orc")
o = json.load(open(os.path.join(out, "p212i_8192.json"))).get("mxfp4", {})
if e is None:
    print(" E_orc: MISSING"); fail = 1
else:
    ok = True
    for cname in ("v2", "native_auto", "native_5", "native_16", "v2_es", "native_es"):
        c, r = e.get(cname) or {}, o.get(cname) or {}
        if abs((c.get("canary") or 1) - (r.get("canary") or 1)) > 1e-6:
            print(f" E_orc {cname}: canary drift {c.get('canary')} vs {r.get('canary')}"); ok = False
        if c.get("dev_us") and r.get("dev_us") and abs(c["dev_us"] - r["dev_us"]) / r["dev_us"] > 0.15:
            print(f" E_orc {cname}: timing drift {c['dev_us']} vs {r['dev_us']}"); ok = False
    print(f" E_orc oracle vs r212i_8192: {'REPRODUCED' if ok else 'DRIFT'}")
    if not ok:
        fail = 1
print("R212I3_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212I3_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
