#!/bin/bash
# p620 r84 — cert9 release gate on the tag-built image vllm-arc:v0.30.0-b70-cert9
# (fork aa2ee17d, P6.20 prefill host-path: L3 off-thread LMCache store +
# L6 dispatch-late + step_trace instrument + registry rows). Legs:
#   [0] manifest fork_sha + p620 code IN image + registry rows + baked-file
#       md5 vs the cert9 tar + image-native verify (p48 battery + p73 slice,
#       throwaway cert9-verify container)
#   [1] default fp8_e4m3 posture (NO env overrides — proves default-ON L3):
#       boot + store-worker-ACTIVE posture (post first request) + env-absent
#       + trace-silent-off + flip-first + integrity/det + run7n t0 +
#       t1.0-xhigh (in-family vs cert8) + spec gate + erraudit
#   [2] fp8 LONG-RETURN GATE (owner hard gate): longprobe full + 3x
#       churn->rehit; corruption bar = garble-empty + finish stop + t0
#       cross-cycle single-hash; t1-seed sporadic miss -> annex
#       re-characterization (cert7/cert8 precedent); WARN class-set diff
#   [3] p65 full faces + tps (image-native perf record for docs)
#   [4] TQ ALL-DTYPE GATE (P6.19 owner bar carried on cert9): k8v4 full
#       depth (longprobe + 3 churn cycles + split tps), k3v4/q4nc/q3nc
#       boot+posture+flip+longprobe; gate = v6 install line (mode=split)
#       + ZERO garble + erraudit 0
#   [5] mq annex: k8v4 + VLLM_ARC_TQ_VERIFY_MODE=mq, posture + tps spot
#       (experimental knob number for docs; not a gate)
#   [6] W14 restore + port-owner verify (radiance-test container is GONE —
#       w14_boot.sh fullres RE-CREATES it + REAL FRANCE generation probe)
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert9
FORK_SHA=aa2ee17d11612d9292684cbcbd2825bde4c7a65d
CTROOT=$R/p620/cert9/gate
mkdir -p "$CTROOT"
cd "$CTROOT"
GATE_FAIL=0

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  (nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p620.log 2>&1 &)
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1
sleep 20

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

kill_arc() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 10
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 10
}

churn() {  # $1 = seed base (distinct per cycle)
  local base=$1
  python3 > "$CTROOT/churn_$base.out" 2>&1 << PYEOF || true
import json, threading, time, urllib.request
def one(i, out):
    body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
            "max_tokens": 256, "seed": $base + i,
            "messages": [{"role": "user", "content":
                f"Essay {i}: explain cache coherence in distributed systems, 200 words."}]}
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=600))
    out.append((time.perf_counter() - t0, r["usage"]["completion_tokens"]))
res = []
ts = [threading.Thread(target=one, args=(i, res)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
print("CHURN 4x256 done:", [f"{d:.1f}s/{n}tok" for d, n in res])
PYEOF
  tail -1 "$CTROOT/churn_$base.out"
}

flip_first() {  # $1 = outfile
  local out=$1
  python3 - > "$out" 2>&1 << 'PYEOF' || true
import json, sys, time, urllib.request
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
}

tps_probe() {  # $1 = outfile  (t6p/t8g-format: solo x2 + agg8, 512 tok)
  local out=$1
  python3 - > "$out" 2>&1 << 'PYEOF' || true
import json, threading, time, urllib.request
URL = "http://127.0.0.1:8000/v1/chat/completions"
def call(max_tok, seed=None):
    body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
            "max_tokens": max_tok, "messages": [{"role": "user", "content":
            "Write a detailed technical essay about GPU memory hierarchies."}]}
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=600))
    return time.perf_counter() - t0, r["usage"]["completion_tokens"], r["choices"][0]["finish_reason"]
for i in (1, 2):
    dt, n, fin = call(512)
    print(f"solo{i}: tps={n / dt:.2f} tok={n} dt={dt:.1f}s finish={fin}")
res = [None] * 8
def agg(i):
    dt, n, fin = call(512)
    res[i] = (dt, n)
ts = [threading.Thread(target=agg, args=(i,)) for i in range(8)]
t0 = time.perf_counter()
[t.start() for t in ts]; [t.join() for t in ts]
wall = time.perf_counter() - t0
tot = sum(n for _, n in res)
print(f"agg8: tps={tot / wall:.2f} tok={tot} wall={wall:.1f}s per_stream=" +
      str([f"{n / dt:.1f}" for dt, n in res]))
PYEOF
}

erraudit() {  # $1 = leg dir
  docker exec arc-test sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$1/erraudit.txt" 2>/dev/null || echo "ERROR=?" > "$1/erraudit.txt"
  cat "$1/erraudit.txt"
  grep -q "ERROR=0" "$1/erraudit.txt" || { echo "ERRAUDIT: FAIL ($1)"; GATE_FAIL=1; }
}

garble_check() {  # $1 = longprobe out file -> zero-garble gate
  local lines empty
  lines=$(grep -cE "^\[(essay|code|multi|simple)\]" "$1" || true)
  empty=$(grep -c "garble={'cold': \[\], 'rehit': \[\], 'rehit2': \[\]\]}" "$1" || true)
  echo "GARBLE: class_lines=$lines empty_garble=$empty (expect equal, >=4) in $(basename "$1")"
  { [ "$lines" -ge 4 ] && [ "$lines" = "$empty" ]; } || { echo "ZERO-GARBLE: FAIL ($1)"; GATE_FAIL=1; }
}

boot_dtype() {  # $1=dtype $2=tag $3=modeenv(yes|no)
  local DTYPE=$1 TAG=$2 MODEENV=$3 CT
  CT=$CTROOT/$TAG
  mkdir -p "$CT"
  echo "=================== [boot] $TAG ($DTYPE modeenv=$MODEENV) $(date +%H:%M:%S) ==================="
  kill_arc
  cp $R/serve_arc_b70.sh $R/serve_p620_$TAG.sh
  sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype $DTYPE/" $R/serve_p620_$TAG.sh
  if [ "$MODEENV" = "yes" ]; then
    sed -i "s|-e VLLM_XPU_ENABLE_XPU_GRAPH=1|-e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e VLLM_ARC_TQ_VERIFY_MODE=mq|" $R/serve_p620_$TAG.sh
    grep -q "VLLM_ARC_TQ_VERIFY_MODE=mq" $R/serve_p620_$TAG.sh || { echo "FATAL: mode env not injected"; exit 1; }
  else
    grep -q "VLLM_ARC_TQ_VERIFY_MODE" $R/serve_p620_$TAG.sh && { echo "FATAL: mode env leaked"; exit 1; }
  fi
  local n
  n=$(grep -c -- "--kv-cache-dtype $DTYPE" $R/serve_p620_$TAG.sh)
  [ "$n" = "1" ] || { echo "FATAL: dtype substitution wrong ($n)"; exit 1; }
  cp $R/serve_p620_$TAG.sh "$CT/serve_used.sh"
  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
    bash $R/serve_p620_$TAG.sh > "$CT/boot.out" 2>&1
  grep ARC_SERVE_LAUNCHED "$CT/boot.out" || { echo "FATAL: launch failed"; cat "$CT/boot.out"; exit 1; }
  wait_health | tee "$CT/health.txt"
  docker exec arc-test sh -c "grep -E 'KV cache size|v6 installed' /root/serve_full.log | head -2" > "$CT/posture.txt" || true
  cat "$CT/posture.txt"
}

echo "=================== P620 R84 CERT9 GATE START $(date +%F\ %H:%M:%S) ==================="

echo "=== [0] manifest + p620 code + registry + baked md5 + image-native verify $(date +%H:%M:%S) ==="
docker run --rm --entrypoint cat "$IMG" /etc/vllm-arc/IMAGE_MANIFEST > manifest.txt 2>&1
grep -q "fork_sha=$FORK_SHA" manifest.txt && echo "MANIFEST fork_sha: PASS ($FORK_SHA)" \
  || { echo "MANIFEST fork_sha: FAIL"; cat manifest.txt; exit 1; }
SP=/opt/venv/lib/python3.12/site-packages
FIX_LS=$(docker run --rm --entrypoint ls "$IMG" "$SP/vllm_gaudi/xpu/patches/")
echo "patches/: $FIX_LS"
echo "$FIX_LS" | grep -q "step_trace.py" || { echo "P620 CODE: step_trace.py missing"; exit 1; }
echo "$FIX_LS" | grep -q "lmcint" || { echo "P620 CODE: lmcint/ missing"; exit 1; }
HOOK=$(docker run --rm --entrypoint grep "$IMG" -c "step_trace" "$SP/vllm_gaudi/xpu/platform.py")
echo "platform.py step_trace entries=$HOOK (expect >=1)"
[ "$HOOK" -ge 1 ] || { echo "PLATFORM HOOK: FAIL"; exit 1; }
ADP=$SP/vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py
L3=$(docker run --rm --entrypoint grep "$IMG" -c "_lmc_store_loop" "$ADP")
L6=$(docker run --rm --entrypoint grep "$IMG" -c "_late_pending" "$ADP")
echo "adapter: _lmc_store_loop=$L3 _late_pending=$L6 (expect >=1 each)"
{ [ "$L3" -ge 1 ] && [ "$L6" -ge 1 ]; } || { echo "P620 ADAPTER CODE: FAIL"; exit 1; }
REG=$(docker run --rm --entrypoint grep "$IMG" -c "VLLM_ARC_LMC_ASYNC_STORE" "$SP/vllm_gaudi/envs.py")
echo "envs.py registry rows (ASYNC_STORE mentions)=$REG (expect >=2)"
[ "$REG" -ge 2 ] || { echo "REGISTRY: FAIL"; exit 1; }

echo "--- baked-file md5 vs cert9 tar ---"
MD5FAIL=0
for f in vllm_gaudi/xpu/patches/step_trace.py \
         vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py \
         vllm_gaudi/xpu/platform.py \
         vllm_gaudi/envs.py; do
  IMG_MD5=$(docker run --rm --entrypoint md5sum "$IMG" "$SP/$f" | awk '{print $1}')
  TAR_MD5=$(tar -xOf $R/cert9_p620.tar "$f" | md5sum | awk '{print $1}')
  if [ "$IMG_MD5" = "$TAR_MD5" ]; then
    echo "MD5 OK  $f  $IMG_MD5"
  else
    echo "MD5 FAIL $f image=$IMG_MD5 tar=$TAR_MD5"
    MD5FAIL=1
  fi
done
[ "$MD5FAIL" = "0" ] || { echo "BAKED MD5: FAIL"; exit 1; }

docker rm -f cert9-verify >/dev/null 2>&1
docker run -d --name cert9-verify --privileged --ipc host \
  --entrypoint /bin/bash "$IMG" -c 'sleep 900' >/dev/null
docker cp $R/r47c/p48_verify.py cert9-verify:/tmp/p48_verify.py
docker cp $R/p65/cert5/p73_verify.py cert9-verify:/tmp/p73_verify.py
docker exec -e LMCACHE_TRACK_USAGE=false cert9-verify /opt/venv/bin/python /tmp/p48_verify.py \
  > p48_verify.out 2>&1
docker exec -e LMCACHE_TRACK_USAGE=false cert9-verify /opt/venv/bin/python /tmp/p73_verify.py \
  > p73_verify.out 2>&1
docker rm -f cert9-verify >/dev/null
grep "P48_VERIFY" p48_verify.out || { echo "P48 BATTERY: FAIL"; tail -20 p48_verify.out; exit 1; }
echo "p48: $(grep 'P48_VERIFY' p48_verify.out)"
grep "P73_CHECKS" p73_verify.out || { echo "P73 SLICE: FAIL"; tail -20 p73_verify.out; exit 1; }
echo "p73: $(grep 'P73_CHECKS' p73_verify.out)"

echo "=== [1] default fp8_e4m3 posture — NO env overrides $(date +%H:%M:%S) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_arc_b70.sh > fp8_boot.out 2>&1
grep ARC_SERVE_LAUNCHED fp8_boot.out || { echo "FATAL: launch failed"; cat fp8_boot.out; exit 1; }
wait_health | tee fp8_health.txt
docker exec arc-test sh -c 'env | grep -E "^VLLM_ARC_LMC_(ASYNC_STORE|EVSYNC)"' > lmc_env_present.txt 2>/dev/null || true
LMCENV=$(wc -l < lmc_env_present.txt)
echo "LMC_ENV_SET=$LMCENV (expect 0 — default-ON must need NO env)"
[ "$LMCENV" = "0" ] || { echo "DEFAULT-ON POSTURE: FAIL (env present)"; GATE_FAIL=1; }

flip_first fp8_flip_first.out
cat fp8_flip_first.out

docker exec arc-test sh -c 'grep -c "LMC store worker ACTIVE" /root/serve_full.log' > store_worker.txt 2>/dev/null || echo 0 > store_worker.txt
SW=$(cat store_worker.txt)
echo "STORE_WORKER_ACTIVE=$SW (expect >=1 — L3 default ON, fires at first wait_for_save)"
[ "$SW" -ge 1 ] || { echo "L3 DEFAULT-ON: FAIL"; GATE_FAIL=1; }

docker exec arc-test sh -c 'ls /tmp/arc_step_trace_* 2>/dev/null | wc -l' > trace_silent.txt 2>/dev/null || echo 1 > trace_silent.txt
TS=$(cat trace_silent.txt)
echo "TRACE_SILENT_OFF=$TS (expect 0 — ARC_STEP_TRACE unset => no files)"
[ "$TS" = "0" ] || { echo "TRACE SILENT-OFF: FAIL"; GATE_FAIL=1; }

python3 $R/r42/r42_integrity.py > integrity.out 2>&1
I5=$(grep -c "INTEGRITY 5/5 PASS" integrity.out || true)
D3=$(grep -c "DETERMINISM 3/3 PASS" integrity.out || true)
echo "A-gates: integrity_5of5=$I5 det_3of3=$D3"
{ [ "$I5" = "1" ] && [ "$D3" = "1" ]; } || { echo "A-GATES: FAIL"; cat integrity.out; GATE_FAIL=1; }

python3 $R/r46t/bench_run7n_t.py --temperature 0 > fp8_t0_run7n.out 2>&1
grep -h 'aggregate=\|solo200' fp8_t0_run7n.out
python3 $R/r46t/bench_run7n_t.py --temperature 1.0 --thinking xhigh > fp8_t1.0_run7n.out 2>&1
grep -h 'aggregate=\|solo200' fp8_t1.0_run7n.out

python3 $R/arc_spec_gate.py > spec_gate.out 2>&1
tail -5 spec_gate.out

docker exec arc-test cat /root/serve_full.log > $CTROOT/fp8_serve_leg1.log 2>/dev/null || true
erraudit $CTROOT

echo "=== [2] fp8 LONG-RETURN GATE (owner hard gate) $(date +%H:%M:%S) ==="
python3 $R/p618/p618_longprobe.py cert9fp8 full > fp8_longprobe_full.out 2>&1
grep -E "^\[|LONGPROBE" fp8_longprobe_full.out
garble_check fp8_longprobe_full.out
for cyc in 1 2 3; do
  churn $((100 * cyc))
  python3 $R/p618/p618_longprobe.py cert9fp8r$cyc rehit > fp8_longprobe_rehit$cyc.out 2>&1
  grep -E "^\[|LONGPROBE" fp8_longprobe_rehit$cyc.out
  garble_check fp8_longprobe_rehit$cyc.out
done

python3 - > fp8_longreturn_summary.txt 2>&1 << 'PYEOF'
import json
fail = []
def load(fn):
    with open(fn) as f:
        return json.load(f)
full = load("longprobe_cert9fp8_full.json")
rehits = [load(f"longprobe_cert9fp8r{c}_rehit.json") for c in (1, 2, 3)]
classes = ["essay", "code", "multi", "simple"]
t1_miss = []
for cls in classes:
    v = full.get(cls, {})
    hashes = set(v.get("hashes", []))
    garble_all = list(v.get("garble", {}).values()) + [v.get("t1_garble", [])]
    for r in rehits:
        rv = r.get(cls, {})
        hashes |= set(rv.get("hashes", []))
        garble_all += list(rv.get("garble", {}).values())
    flat = [g for g in garble_all if g]
    if len(hashes) != 1:
        fail.append(f"{cls}: cross-cycle hash drift {sorted(hashes)}")
    if flat:
        fail.append(f"{cls}: garble flags {flat}")
    if any(fin != "stop" for fin in v.get("finish", [])):
        fail.append(f"{cls}: truncation finish={v.get('finish')}")
    if not v.get("t1_seed_identical"):
        t1_miss.append(cls)
    print(f"[{cls}] hashes={sorted(hashes)} finish={v.get('finish')} "
          f"chars={v.get('chars')} t1_seed_identical={v.get('t1_seed_identical')} "
          f"garble_empty={not flat}")
print("T1_SEED_MISS_CLASSES:", t1_miss)
print("FP8_LONGRETURN_CORRUPTION_BAR:", "PASS" if not fail else f"FAIL {fail}")
PYEOF
cat fp8_longreturn_summary.txt
grep -q "FP8_LONGRETURN_CORRUPTION_BAR: PASS" fp8_longreturn_summary.txt \
  || { echo "FP8 LONG-RETURN: FAIL"; GATE_FAIL=1; }
if grep -q "T1_SEED_MISS_CLASSES: \['" fp8_longreturn_summary.txt; then
  echo "--- t1-seed annex re-characterization (cert7/cert8 precedent) ---"
  python3 $R/p618/p618d_t1seed_char.py > fp8_t1seed_char.out 2>&1
  tail -8 fp8_t1seed_char.out
fi

docker exec arc-test sh -c 'grep WARNING /root/serve_full.log' | sed -E 's/[0-9]+/N/g' | sort -u > warn_classes.txt
wc -l warn_classes.txt
if [ -f "$R/p618/cert8/gate/warn_classes.txt" ]; then
  only9=$(comm -23 warn_classes.txt "$R/p618/cert8/gate/warn_classes.txt" | wc -l)
  only8=$(comm -13 warn_classes.txt "$R/p618/cert8/gate/warn_classes.txt" | wc -l)
  echo "WARN class-set diff vs cert8: +$only9 -$only8 (expect 0 0)"
fi

echo "=== [3] p65 full faces + tps (image-native perf record) $(date +%H:%M:%S) ==="
python3 $R/p65/p65_timeline.py cert9_timeline.json > cert9_p65_stdout.txt 2>&1 || echo "timeline exit=$?"
grep -A5 'long-stream faces' cert9_p65_stdout.txt || tail -20 cert9_p65_stdout.txt
tps_probe fp8_tps.out
cat fp8_tps.out

echo "=== [4] TQ ALL-DTYPE GATE (P6.19 owner bar carried) $(date +%H:%M:%S) ==="
boot_dtype turboquant_k8v4 k8v4 no
grep -q "v6 installed (active=True mode=split mq_kernel=True)" $CTROOT/k8v4/posture.txt \
  || { echo "K8V4 POSTURE: FAIL (v6 split install line absent)"; GATE_FAIL=1; }
flip_first $CTROOT/k8v4/flip_first.out
cat $CTROOT/k8v4/flip_first.out
python3 $R/p618/p618_longprobe.py cert9k8v4 full > $CTROOT/k8v4/longprobe_full.out 2>&1
grep -E "^\[|LONGPROBE" $CTROOT/k8v4/longprobe_full.out
garble_check $CTROOT/k8v4/longprobe_full.out
for cyc in 1 2 3; do
  churn $((200 * cyc))
  python3 $R/p618/p618_longprobe.py cert9k8v4r$cyc rehit > $CTROOT/k8v4/longprobe_rehit$cyc.out 2>&1
  grep -E "^\[|LONGPROBE" $CTROOT/k8v4/longprobe_rehit$cyc.out
  garble_check $CTROOT/k8v4/longprobe_rehit$cyc.out
done
python3 - > $CTROOT/k8v4/longreturn_summary.txt 2>&1 << 'PYEOF'
import json
fail = []
def load(fn):
    with open(fn) as f:
        return json.load(f)
full = load("longprobe_cert9k8v4_full.json")
rehits = [load(f"longprobe_cert9k8v4r{c}_rehit.json") for c in (1, 2, 3)]
for cls in ["essay", "code", "multi", "simple"]:
    v = full.get(cls, {})
    hashes = set(v.get("hashes", []))
    garble_all = list(v.get("garble", {}).values()) + [v.get("t1_garble", [])]
    for r in rehits:
        rv = r.get(cls, {})
        hashes |= set(rv.get("hashes", []))
        garble_all += list(rv.get("garble", {}).values())
    flat = [g for g in garble_all if g]
    if flat:
        fail.append(f"{cls}: garble {flat}")
    if any(fin != "stop" for fin in v.get("finish", [])):
        fail.append(f"{cls}: truncation {v.get('finish')}")
    drift = " (t0 drift: adjacent coherent variants, Class-A)" if len(hashes) > 1 else ""
    print(f"[{cls}] hashes={sorted(hashes)} finish={v.get('finish')}{drift} garble_empty={not flat}")
print("K8V4_LONGRETURN_CORRUPTION_BAR:", "PASS" if not fail else f"FAIL {fail}")
PYEOF
cat $CTROOT/k8v4/longreturn_summary.txt
grep -q "K8V4_LONGRETURN_CORRUPTION_BAR: PASS" $CTROOT/k8v4/longreturn_summary.txt \
  || { echo "K8V4 LONG-RETURN: FAIL"; GATE_FAIL=1; }
tps_probe $CTROOT/k8v4/tps.out
cat $CTROOT/k8v4/tps.out
erraudit $CTROOT/k8v4
docker exec arc-test cat /root/serve_full.log > $CTROOT/k8v4/serve_full.log 2>/dev/null || true

for D in "turboquant_k3v4_nc k3v4" "turboquant_4bit_nc q4nc" "turboquant_3bit_nc q3nc"; do
  set -- $D
  boot_dtype $1 $2 no
  grep -q "v6 installed (active=True mode=split mq_kernel=True)" $CTROOT/$2/posture.txt \
    || { echo "$2 POSTURE: FAIL (v6 split install line absent)"; GATE_FAIL=1; }
  flip_first $CTROOT/$2/flip_first.out
  cat $CTROOT/$2/flip_first.out
  python3 $R/p618/p618_longprobe.py cert9$2 full > $CTROOT/$2/longprobe_full.out 2>&1
  grep -E "^\[|LONGPROBE" $CTROOT/$2/longprobe_full.out
  garble_check $CTROOT/$2/longprobe_full.out
  erraudit $CTROOT/$2
  docker exec arc-test cat /root/serve_full.log > $CTROOT/$2/serve_full.log 2>/dev/null || true
done

echo "=== [5] mq annex (experimental knob number) $(date +%H:%M:%S) ==="
boot_dtype turboquant_k8v4 mqannex yes
grep -q "v6 installed (active=True mode=mq mq_kernel=True)" $CTROOT/mqannex/posture.txt \
  || { echo "MQANNEX POSTURE: note (mode=mq line absent)"; }
flip_first $CTROOT/mqannex/flip_first.out
cat $CTROOT/mqannex/flip_first.out
tps_probe $CTROOT/mqannex/tps.out
cat $CTROOT/mqannex/tps.out
erraudit $CTROOT/mqannex
docker exec arc-test cat /root/serve_full.log > $CTROOT/mqannex/serve_full.log 2>/dev/null || true

echo "=== [6] W14 restore + port-owner verify (w14_boot fullres + REAL probe) $(date +%H:%M:%S) ==="
kill_arc
(nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p620_final.log 2>&1 &)
ok=""
for i in $(seq 1 180); do
  grep -q "W13H_BOOT_DONE" $R/w14_restore_p620_final.log 2>/dev/null && { ok=1; break; }
  sleep 10
done
echo "W14_BOOT_DONE=$ok after ~$((i*10))s"
grep -E "HEALTH_OK|FRANCE|BOOT_DONE" $R/w14_restore_p620_final.log || tail -15 $R/w14_restore_p620_final.log
PO=$(docker ps --format '{{.Names}} {{.Ports}}' | grep 8000 || echo NONE)
echo "PORT_8000_OWNER=$PO (expect radiance-test, arc-test absent)"
docker inspect arc-test >/dev/null 2>&1 && { echo "ARC-TEST STILL UP (BAD)"; GATE_FAIL=1; } || echo "arc-test removed OK"
grep -q "FRANCE:" $R/w14_restore_p620_final.log || { echo "W14 REAL GENERATION PROBE: FAIL"; GATE_FAIL=1; }

echo "=================== P620 R84 CERT9 GATE COMPLETE $(date +%F\ %H:%M:%S) GATE_FAIL=$GATE_FAIL ==================="
exit $GATE_FAIL
