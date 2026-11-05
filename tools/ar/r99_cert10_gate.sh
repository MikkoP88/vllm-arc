#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# r99_cert10_gate.sh — P6.24 phase 3: cert10 image + release battery +
# FP8 serve A/B (plan P6.24 r99 row).
#
# Preconditions: r98 quality ladder PASS (task-level parity law); stage-arc
# fat sha == pinned v4 (3a120e8b...).
#
# Legs:
#   [0] BUILD cert10 = cert9 + cert10.Dockerfile overlay (two files + manifest
#       rows); post-build gates: manifest fork_sha == cert9's aa2ee17d, oneccl
#       rows present, lib sha256 == pin BOTH paths, EXHAUSTIVE vllm_gaudi/
#       md5 walk identical to cert9 (tree-identity proof — the basis for
#       carrying cert9's TQ/all-dtype certification), wheels md5 vs pins,
#       thin .so.2 identical.
#   [1] DEFAULT-POSTURE release battery on cert10 (NO fp8 env — the shipped
#       posture; r84 [1]+[2] pattern): boot, env-absent proof, flip-first,
#       L3 store-worker ACTIVE post-traffic, trace silent-off, integrity/det,
#       run7n t0 + t1.0-xhigh, spec gate, tps solo+agg8, LONG-RETURN full +
#       3 churn cycles (owner hard gate), erraudit.
#   [2] TQ carry block: k8v4 boot + posture + flip + longprobe full +
#       garble + erraudit (tree-identity carries the remaining dtype faces).
#   [3] FP8 knob block ON cert10 (the shipped image with the knob ON):
#       FP8=1 boot, flip-first, run7n t0 spot, spec, STATS-once probe (long
#       prefill -> in-situ wire proof on the certified image), erraudit.
#   [4] A/B/A/B perf interleave (r81 pattern): p65_timeline v5.1 full faces
#       (U32k/C32k/U8k/U64k + bystander waves) x off1/on1/off2/on2.
#       Pre-registered bars: U32k + U64k TTFT on/off ratio in [0.80, 1.02]
#       (expected ~0.90: prefill AR slice 27.3%, dominant class 88% of it at
#       1.65x -> ~9.5% prefill-time cut); C32k/U8k + bystander decode tps
#       REPORTED (near-zero expected delta — decode ARs stay stock under
#       MIN_COUNT=2097152).
#   [V] verdict tables. W14 restored by trap (rig law).
#
# RUN NOTE (2026-11-05, post-run — r98 precedent: the runner documents its
# own defects; measured data was never affected):
#   - r99 executed [0]+[1] 18:56-19:10. [0] ALL identity gates PASS;
#     [1] battery green BUT longreturn_summary threw FileNotFoundError —
#     longprobe writes its jsons to CWD (== $CT), the summary ran inside
#     $LEG. All data present; corrected verdict recomputed from the banked
#     jsons by r99def_reconstruct.py (added the r98 raw-bytes ghost fence).
#     That recompute surfaced ONE t1 garble flag (code, REP-LOOP) —
#     dispositioned by r99c_annex.sh (stock basis + cross-boot repro +
#     verbatim loop capture), see VERDICT.md.
#   - [2] died at the k8v4 boot on a transient "OCI runtime exec failed:
#     setns ... exit status 1" — this script lacked the r84 kill_arc retry
#     loop on container transitions. [2]+[3]+[4] completed same-day by
#     r99b_cert10_cont.sh with the retry law; results land in the same
#     $CT tree. Fixes applied in-place below (kill_arc + summary cwd) so
#     the runner is correct for any future re-run.
set -uo pipefail
R=/root/build
CT=$R/p624/r99
STAGE=$R/p624/stage-arc
IMG9=vllm-arc:v0.30.0-b70-cert9
IMG=vllm-arc:v0.30.0-b70-cert10
FAT_SHA=3a120e8b352fbd6be45ab61a7f4e927d80c9bc6618ff646e6db68b0263eca4a7
PATCH_SHA=effe9a27fae37ff596a43f25cfb9cc504f8e4ab5889bebcb0a92fdce7e14a07d
FORK_SHA=aa2ee17d11612d9292684cbcbd2825bde4c7a65d
SP=/opt/venv/lib/python3.12/site-packages
GATE_FAIL=0

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p624d.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 20

mkdir -p $CT
cd $CT

kill_arc() {  # r84 retry law: rm + inspect-wait (the OCI setns guard)
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

flip_first() {  # $1 = outfile
  python3 - > "$1" 2>&1 << 'PYEOF' || true
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
}

churn() {  # $1 = seed base
  local base=$1
  python3 > "$CT/churn_$base.out" 2>&1 << PYEOF || true
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
  tail -1 "$CT/churn_$base.out"
}

tps_probe() {  # $1 = outfile
  python3 - > "$1" 2>&1 << 'PYEOF' || true
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

garble_check() {  # $1 = longprobe out file — r98 law: grep -F (the r84-era
  # BRE "\[\]" escapes never matched — empty_garble was always 0 and every
  # gate carried a false ZERO-GARBLE FAIL line dispositioned by the summary
  # PASS; fixed here, do not reintroduce)
  local lines empty
  lines=$(grep -cE "^\[(essay|code|multi|simple)\]" "$1" || true)
  empty=$(( $(grep -cF "garble={'cold': [], 'rehit': [], 'rehit2': []}" "$1" || true)
          + $(grep -cF "garble={'r1': [], 'r2': []}" "$1" || true) ))
  echo "GARBLE: class_lines=$lines empty_garble=$empty (expect equal, >=4) in $(basename "$1")"
  { [ "$lines" -ge 4 ] && [ "$lines" = "$empty" ]; } || { echo "ZERO-GARBLE: FAIL ($1)"; GATE_FAIL=1; }
}

longreturn_summary() {  # $1 = label
  python3 - "$1" > longreturn_summary.txt 2>&1 << 'PYEOF' || true
import json, sys
tag = sys.argv[1]
def load(fn):
    with open(fn) as f:
        return json.load(f)
full = load(f"longprobe_{tag}_full.json")
rehits = [load(f"longprobe_{tag}r{c}_rehit.json") for c in (1, 2, 3)]
fail = []
t1_miss = []
for cls in ["essay", "code", "multi", "simple"]:
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
print(f"LONGRETURN_BAR_{tag.upper()}:",
      "PASS" if not fail else f"FAIL {fail}")
PYEOF
  cat longreturn_summary.txt
  grep -q "LONGRETURN_BAR.*: PASS" longreturn_summary.txt \
    || { echo "LONGRETURN: FAIL ($1)"; GATE_FAIL=1; }
}

boot_cert10() {  # $1 = leg dir, $2 = extra serve env injection sed ("" = none), $3 = serve tag
  local LEG=$1 INJ="$2" STAG=$3 SCRIPT
  mkdir -p $LEG
  if [ -z "$INJ" ]; then
    SCRIPT=$R/serve_arc_b70.sh
    cp $SCRIPT $LEG/serve_used.sh
  else
    SCRIPT=$R/serve_p624_r99_$STAG.sh
    cp $R/serve_arc_b70.sh $SCRIPT
    sed -i "s|export CCL_SYCL_ALLREDUCE_TMP_BUF=0|export CCL_SYCL_ALLREDUCE_TMP_BUF=0; $INJ|" $SCRIPT
    NINJ=$(grep -c "$INJ" $SCRIPT || true)
    [ "$NINJ" = "1" ] || { echo "FATAL: env inject wrong ($NINJ) for $INJ"; exit 1; }
    cp $SCRIPT $LEG/serve_used.sh
  fi
  kill_arc
  ( cd $LEG && ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
      bash $SCRIPT > boot.out 2>&1 )
  grep ARC_SERVE_LAUNCHED $LEG/boot.out || { echo "FATAL: launch failed ($STAG)"; cat $LEG/boot.out; exit 1; }
  wait_health | tee -a $LEG/boot.out
  docker exec arc-test sh -c "grep -E 'KV cache size|v6 installed' /root/serve_full.log | head -2" > $LEG/posture.txt || true
  cat $LEG/posture.txt
}

echo "=================== P624 R99 CERT10 GATE START $(date +%F\ %H:%M:%S) ==================="

echo "=== [0] build cert10 + identity gates $(date +%H:%M:%S) ==="
SHA_NOW=$(sha256sum $STAGE/libccl.so.1.0 | awk '{print $1}')
[ "$SHA_NOW" = "$FAT_SHA" ] || { echo "FATAL: stage fat sha drift ($SHA_NOW)"; exit 1; }
mkdir -p $CT/cert10ctx
cp $STAGE/libccl.so.1.0 $CT/cert10ctx/
cp $R/p624-kit/external_patches/oneccl/cert10.Dockerfile $CT/cert10ctx/ 2>/dev/null \
  || { echo "FATAL: cert10.Dockerfile missing on rig (ship external_patches/oneccl/)"; exit 1; }
docker build -f $CT/cert10ctx/cert10.Dockerfile \
  --build-arg LIB_SHA=$FAT_SHA --build-arg PATCH_SHA=$PATCH_SHA \
  --build-arg BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -t $IMG $CT/cert10ctx > build_cert10.out 2>&1 \
  || { echo "FATAL: cert10 build failed"; tail -30 build_cert10.out; exit 1; }
grep -E "^oneccl=" build_cert10.out | head -4
docker run --rm --entrypoint cat $IMG /etc/vllm-arc/IMAGE_MANIFEST > manifest_cert10.txt
grep -q "fork_sha=$FORK_SHA" manifest_cert10.txt || { echo "MANIFEST fork_sha: FAIL"; GATE_FAIL=1; }
grep -q "oneccl=2022.0.0+arc1" manifest_cert10.txt || { echo "MANIFEST oneccl row: FAIL"; GATE_FAIL=1; }
grep -q "$FAT_SHA" manifest_cert10.txt || { echo "MANIFEST lib sha row: FAIL"; GATE_FAIL=1; }
echo "MANIFEST: OK (fork_sha + oneccl rows)"
# exhaustive tree-identity walk vs cert9 (carries cert9's vllm-layer certs)
docker run --rm --entrypoint bash $IMG9 -c \
  "cd $SP && find vllm_gaudi -type f | sort | xargs md5sum" > tree_cert9.md5
docker run --rm --entrypoint bash $IMG  -c \
  "cd $SP && find vllm_gaudi -type f | sort | xargs md5sum" > tree_cert10.md5
NDIFF=$(diff tree_cert9.md5 tree_cert10.md5 | wc -l)
echo "TREE_IDENTITY vllm_gaudi/: diff_lines=$NDIFF (expect 0)"
[ "$NDIFF" = "0" ] || { echo "TREE IDENTITY: FAIL"; diff tree_cert9.md5 tree_cert10.md5 | head; GATE_FAIL=1; }
docker run --rm --entrypoint bash $IMG9 -c "md5sum /opt/venv/lib/libccl.so.2.0" > thin_cert9.md5
docker run --rm --entrypoint bash $IMG  -c "md5sum /opt/venv/lib/libccl.so.2.0" > thin_cert10.md5
cmp -s thin_cert9.md5 thin_cert10.md5 && echo "THIN SHIM: identical (untouched)" \
  || { echo "THIN SHIM: FAIL (changed)"; GATE_FAIL=1; }
for P in /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so.1.0; do
  GOT=$(docker run --rm --entrypoint sha256sum $IMG $P | awk '{print $1}')
  [ "$GOT" = "$FAT_SHA" ] && echo "OVERLAY SHA OK  $P" \
    || { echo "OVERLAY SHA FAIL $P ($GOT)"; GATE_FAIL=1; }
done

echo "=== [1] DEFAULT-POSTURE release battery (NO fp8 env) $(date +%H:%M:%S) ==="
LEG=$CT/def
boot_cert10 $LEG "" def
docker exec arc-test sh -c 'env | grep -c "^CCL_SYCL_ALLREDUCE_FP8"' > $LEG/env_absent.txt 2>/dev/null || echo 0 > $LEG/env_absent.txt
NA=$(cat $LEG/env_absent.txt)
echo "FP8_ENV_SET=$NA (expect 0 — silent-off default posture)"
[ "$NA" = "0" ] || { echo "DEFAULT-OFF POSTURE: FAIL (env present)"; GATE_FAIL=1; }
flip_first $LEG/flip_first.out; cat $LEG/flip_first.out
python3 $R/r42/r42_integrity.py > $LEG/integrity.out 2>&1
I5=$(grep -c "INTEGRITY 5/5 PASS" $LEG/integrity.out || true)
D3=$(grep -c "DETERMINISM 3/3 PASS" $LEG/integrity.out || true)
echo "A-gates: integrity_5of5=$I5 det_3of3=$D3"
{ [ "$I5" = "1" ] && [ "$D3" = "1" ]; } || { echo "A-GATES: FAIL"; cat $LEG/integrity.out; GATE_FAIL=1; }
python3 $R/r46t/bench_run7n_t.py --temperature 0 > $LEG/run7n_t0.out 2>&1
grep -hE 'aggregate=|solo200|call_FAIL' $LEG/run7n_t0.out
python3 $R/r46t/bench_run7n_t.py --temperature 1.0 --thinking xhigh > $LEG/run7n_t1.out 2>&1
grep -hE 'aggregate=|solo200|call_FAIL' $LEG/run7n_t1.out
python3 $R/arc_spec_gate.py > $LEG/spec_gate.out 2>&1
tail -3 $LEG/spec_gate.out
grep -q "SPEC_GATE PASS" $LEG/spec_gate.out || { echo "SPEC_GATE: FAIL"; GATE_FAIL=1; }
tps_probe $LEG/tps.out; cat $LEG/tps.out
echo "--- LONG-RETURN full + 3 churn cycles (owner hard gate) $(date +%H:%M:%S) ---"
python3 $R/p618/p618_longprobe.py r99def full > $LEG/longprobe_full.out 2>&1
grep -E "^\[|LONGPROBE" $LEG/longprobe_full.out
garble_check $LEG/longprobe_full.out
for cyc in 1 2 3; do
  churn $(( 700 * cyc ))
  python3 $R/p618/p618_longprobe.py r99defr$cyc rehit > $LEG/longprobe_rehit$cyc.out 2>&1
  grep -E "^\[|LONGPROBE" $LEG/longprobe_rehit$cyc.out
  garble_check $LEG/longprobe_rehit$cyc.out
done
( cd $CT && longreturn_summary r99def )
mv -f $CT/longreturn_summary.txt $LEG/longreturn_summary.txt
docker exec arc-test sh -c 'ls /tmp/arc_step_trace_* 2>/dev/null | wc -l' > $LEG/trace_silent.txt 2>/dev/null || echo 1 > $LEG/trace_silent.txt
[ "$(cat $LEG/trace_silent.txt)" = "0" ] || { echo "TRACE SILENT-OFF: FAIL"; GATE_FAIL=1; }
erraudit $LEG
docker exec arc-test cat /root/serve_full.log > $LEG/serve_full.log 2>/dev/null || true

echo "=== [2] TQ carry block: k8v4 boot + longprobe $(date +%H:%M:%S) ==="
LEG=$CT/k8v4
cp $R/serve_arc_b70.sh $R/serve_p624_r99_k8v4.sh
sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_k8v4/" $R/serve_p624_r99_k8v4.sh
NK=$(grep -c -- "--kv-cache-dtype turboquant_k8v4" $R/serve_p624_r99_k8v4.sh)
[ "$NK" = "1" ] || { echo "FATAL: k8v4 dtype substitution wrong ($NK)"; exit 1; }
kill_arc
mkdir -p $LEG
( cd $LEG && ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
    bash $R/serve_p624_r99_k8v4.sh > boot.out 2>&1 )
grep ARC_SERVE_LAUNCHED $LEG/boot.out || { echo "FATAL: k8v4 launch failed"; cat $LEG/boot.out; exit 1; }
wait_health | tee -a $LEG/boot.out
docker exec arc-test sh -c "grep -E 'KV cache size|v6 installed' /root/serve_full.log | head -2" > $LEG/posture.txt || true
cat $LEG/posture.txt
grep -q "v6 installed (active=True mode=split mq_kernel=True)" $LEG/posture.txt \
  || { echo "K8V4 POSTURE: FAIL (v6 split install line absent)"; GATE_FAIL=1; }
flip_first $LEG/flip_first.out; cat $LEG/flip_first.out
python3 $R/p618/p618_longprobe.py r99k8v4 full > $LEG/longprobe_full.out 2>&1
grep -E "^\[|LONGPROBE" $LEG/longprobe_full.out
garble_check $LEG/longprobe_full.out
erraudit $LEG
docker exec arc-test cat /root/serve_full.log > $LEG/serve_full.log 2>/dev/null || true

echo "=== [3] FP8 knob block on cert10 (shipped image, knob ON) $(date +%H:%M:%S) ==="
LEG=$CT/fp8
boot_cert10 $LEG "export CCL_SYCL_ALLREDUCE_FP8=1" fp8
flip_first $LEG/flip_first.out; cat $LEG/flip_first.out
python3 $R/r46t/bench_run7n_t.py --temperature 0 > $LEG/run7n_t0.out 2>&1
grep -hE 'aggregate=|solo200|call_FAIL' $LEG/run7n_t0.out
python3 $R/arc_spec_gate.py > $LEG/spec_gate.out 2>&1
tail -3 $LEG/spec_gate.out
grep -q "SPEC_GATE PASS" $LEG/spec_gate.out || { echo "SPEC_GATE (fp8): FAIL"; GATE_FAIL=1; }
# in-situ wire proof on the certified image (STATS=1 one-shot probe boot)
kill_arc
cp $R/serve_p624_r99_fp8.sh $R/serve_p624_r99_stats.sh
sed -i "s|export CCL_SYCL_ALLREDUCE_FP8=1|export CCL_SYCL_ALLREDUCE_FP8=1; export CCL_SYCL_ALLREDUCE_FP8_STATS=1|" $R/serve_p624_r99_stats.sh
NS=$(grep -c "FP8_STATS=1" $R/serve_p624_r99_stats.sh || true)
[ "$NS" = "1" ] || { echo "FATAL: STATS inject wrong ($NS)"; exit 1; }
( cd $LEG && ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
    bash $R/serve_p624_r99_stats.sh > boot_stats.out 2>&1 )
grep ARC_SERVE_LAUNCHED $LEG/boot_stats.out || { echo "FATAL: stats boot failed"; cat $LEG/boot_stats.out; exit 1; }
wait_health | tee -a $LEG/boot_stats.out
python3 - > $LEG/stats_probe.out 2>&1 << 'PYEOF' || true
import json, time, urllib.request
para = ("Distributed training synchronizes gradient tensors across accelerator "
        "ranks through collective communication libraries, and the wire format "
        "of those collectives determines both the achievable bandwidth and the "
        "numerical envelope of the resulting model updates. ")
body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Summarize the technical context:\n" + para * 420}]}
t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=900))
c = r["choices"][0]
print("STATS_PROBE:", c["finish_reason"], f"{time.perf_counter()-t0:.1f}s",
      "prompt_tokens=", r["usage"]["prompt_tokens"])
PYEOF
cat $LEG/stats_probe.out
docker exec arc-test sh -c "grep -aiE 'CCL fp8 (allreduce|stage)' /root/serve_full.log | tail -40" > $LEG/stats_lines.out || true
echo "fp8 stats lines: $(wc -l < $LEG/stats_lines.out)"; head -14 $LEG/stats_lines.out
[ "$(wc -l < $LEG/stats_lines.out)" -ge 1 ] && echo "FP8_WIRE_ON_CERT10: PROVEN" \
  || { echo "FP8_WIRE_ON_CERT10: FAIL (no stats lines)"; GATE_FAIL=1; }
erraudit $LEG
docker exec arc-test cat /root/serve_full.log > $LEG/serve_full.log 2>/dev/null || true

echo "=== [4] A/B/A/B perf interleave: p65 faces off/on/off/on $(date +%H:%M:%S) ==="
perf_phase() {  # $1 = tag, $2 = inj ("" off | "export CCL_SYCL_ALLREDUCE_FP8=1" on)
  local TAG=$1 INJ="$2" LEG=$CT/perf_$1
  mkdir -p $LEG
  echo "--- perf phase $TAG ($(date +%H:%M:%S)) ---"
  if [ -z "$INJ" ]; then
    SCRIPT=$R/serve_arc_b70.sh
  else
    SCRIPT=$R/serve_p624_r99_perf$TAG.sh
    cp $R/serve_arc_b70.sh $SCRIPT
    sed -i "s|export CCL_SYCL_ALLREDUCE_TMP_BUF=0|export CCL_SYCL_ALLREDUCE_TMP_BUF=0; $INJ|" $SCRIPT
    grep -q "CCL_SYCL_ALLREDUCE_FP8=1" $SCRIPT || { echo "FATAL: perf env inject failed"; exit 1; }
  fi
  kill_arc
  ( cd $LEG && ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
      bash $SCRIPT > boot.out 2>&1 )
  grep ARC_SERVE_LAUNCHED $LEG/boot.out || { echo "FATAL: perf boot failed ($TAG)"; cat $LEG/boot.out; exit 1; }
  wait_health | tee -a $LEG/boot.out
  python3 $R/p65/p65_timeline.py $LEG/timeline.json > $LEG/timeline_stdout.txt 2>&1 || echo "timeline exit=$?"
  grep -A5 'long-stream faces' $LEG/timeline_stdout.txt || tail -15 $LEG/timeline_stdout.txt
}
perf_phase off1 ""
perf_phase on1  "export CCL_SYCL_ALLREDUCE_FP8=1"
perf_phase off2 ""
perf_phase on2  "export CCL_SYCL_ALLREDUCE_FP8=1"

echo "=== [V] verdict $(date +%H:%M:%S) ==="
python3 - > verdict.txt 2>&1 << 'PYEOF' || true
import re, sys

def faces(p):
    d = {}
    for line in open(p, errors="replace"):
        m = re.match(r"\s*(U32k|C32k|U8k|U64k): TTFT=\s*([0-9.]+)s total=\s*([0-9.]+)s", line)
        if m:
            d[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return d

offs = [faces(f"perf_off{i}/timeline_stdout.txt") for i in (1, 2)]
ons = [faces(f"perf_on{i}/timeline_stdout.txt") for i in (1, 2)]
fails = []
for f in ("U32k", "C32k", "U8k", "U64k"):
    o = [x[f] for x in offs if f in x]
    n = [x[f] for x in ons if f in x]
    if not o or not n:
        fails.append(f"{f}: missing face rows")
        continue
    o_ttft = sum(v[0] for v in o) / len(o)
    n_ttft = sum(v[0] for v in n) / len(n)
    r = n_ttft / o_ttft if o_ttft else 0
    spread_o = abs(o[0][0] - o[1][0]) / o_ttft if len(o) == 2 and o_ttft else 0
    line = (f"{f}: TTFT off={o_ttft:7.2f}s on={n_ttft:7.2f}s ratio={r:.3f} "
            f"(off spread {spread_o*100:.1f}%)")
    if f in ("U32k", "U64k") and not 0.80 <= r <= 1.02:
        fails.append(line + " OUT of [0.80,1.02]")
    print(line)
print("R99_CERT10_PERF:", "PASS" if not fails else f"FAIL {fails}")
PYEOF
cat verdict.txt
grep -q "R99_CERT10_PERF: PASS" verdict.txt || GATE_FAIL=1
echo "NOTE: full quality parity on this exact overlay bytes = r98 (GATE PASS);"
echo "      [1] battery above re-proves the release posture on the built image."

echo "=================== P624 R99 COMPLETE $(date +%F\ %H:%M:%S) GATE_FAIL=$GATE_FAIL ==================="
exit $GATE_FAIL
