#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# r99b_cert10_cont.sh — P6.24 phase 3 CONTINUATION after the r99 [2] OCI
# failure (2026-11-05 19:09: the k8v4 boot hit "OCI runtime exec failed ...
# setns exit status 1" — a transient netns/docker-daemon failure the r84
# kill_arc retry loop covers; r99 lacked it. [0] build+identity gates and
# [1] default-posture battery are COMPLETE and banked — not re-run).
#
# Remaining legs (cert10 image, W14 trap):
#   [2] TQ carry block: k8v4 boot + posture + flip + longprobe full
#   [3] FP8 knob block: FP8=1 boot + flip + run7n t0 + spec + STATS probe
#   [4] A/B/A/B perf interleave: p65 faces off1/on1/off2/on2
#   [V] perf verdict (bars pre-registered in r99_cert10_gate.sh)
# Longprobe jsons land in the LEG dir this time (cwd law — r99 [1] wrote
# them to $CT; the reconstruct reads whichever dir holds them).
set -uo pipefail
R=/root/build
CT=$R/p624/r99
IMG=vllm-arc:v0.30.0-b70-cert10
GATE_FAIL=0

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p624e.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 20

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

boot_serve() {  # $1 = leg dir, $2 = serve script, $3 = tag
  local LEG=$1 SCRIPT=$2 STAG=$3
  mkdir -p $LEG
  kill_arc
  ( cd $LEG && ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
      bash $SCRIPT > boot.out 2>&1 )
  grep ARC_SERVE_LAUNCHED $LEG/boot.out || { echo "FATAL: launch failed ($STAG)"; cat $LEG/boot.out; exit 1; }
  local ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health ($STAG)"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK ($STAG) after ~$((i*5))s" | tee -a $LEG/boot.out
  docker exec arc-test sh -c "grep -E 'KV cache size|v6 installed' /root/serve_full.log | head -2" > $LEG/posture.txt || true
  cat $LEG/posture.txt
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

erraudit() {  # $1 = leg dir
  docker exec arc-test sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$1/erraudit.txt" 2>/dev/null || echo "ERROR=?" > "$1/erraudit.txt"
  cat "$1/erraudit.txt"
  grep -q "ERROR=0" "$1/erraudit.txt" || { echo "ERRAUDIT: FAIL ($1)"; GATE_FAIL=1; }
}

garble_check() {  # $1 = longprobe out file — grep -F law (r98)
  local lines empty
  lines=$(grep -cE "^\[(essay|code|multi|simple)\]" "$1" || true)
  empty=$(( $(grep -cF "garble={'cold': [], 'rehit': [], 'rehit2': []}" "$1" || true)
          + $(grep -cF "garble={'r1': [], 'r2': []}" "$1" || true) ))
  echo "GARBLE: class_lines=$lines empty_garble=$empty (expect equal, >=4) in $(basename "$1")"
  { [ "$lines" -ge 4 ] && [ "$lines" = "$empty" ]; } || { echo "ZERO-GARBLE: FAIL ($1)"; GATE_FAIL=1; }
}

echo "=================== P624 R99B CONTINUATION START $(date +%F\ %H:%M:%S) ==================="

echo "=== [2] TQ carry block: k8v4 boot + longprobe $(date +%H:%M:%S) ==="
LEG=$CT/k8v4
cp $R/serve_arc_b70.sh $R/serve_p624_r99_k8v4.sh
sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_k8v4/" $R/serve_p624_r99_k8v4.sh
NK=$(grep -c -- "--kv-cache-dtype turboquant_k8v4" $R/serve_p624_r99_k8v4.sh)
[ "$NK" = "1" ] || { echo "FATAL: k8v4 dtype substitution wrong ($NK)"; exit 1; }
boot_serve $LEG $R/serve_p624_r99_k8v4.sh k8v4
grep -q "v6 installed (active=True mode=split mq_kernel=True)" $LEG/posture.txt \
  || { echo "K8V4 POSTURE: FAIL (v6 split install line absent)"; GATE_FAIL=1; }
flip_first $LEG/flip_first.out; cat $LEG/flip_first.out
( cd $LEG && python3 $R/p618/p618_longprobe.py r99k8v4 full > longprobe_full.out 2>&1 )
grep -E "^\[|LONGPROBE" $LEG/longprobe_full.out
garble_check $LEG/longprobe_full.out
erraudit $LEG
docker exec arc-test cat /root/serve_full.log > $LEG/serve_full.log 2>/dev/null || true

echo "=== [3] FP8 knob block on cert10 (shipped image, knob ON) $(date +%H:%M:%S) ==="
LEG=$CT/fp8
cp $R/serve_arc_b70.sh $R/serve_p624_r99_fp8.sh
sed -i "s|export CCL_SYCL_ALLREDUCE_TMP_BUF=0|export CCL_SYCL_ALLREDUCE_TMP_BUF=0; export CCL_SYCL_ALLREDUCE_FP8=1|" $R/serve_p624_r99_fp8.sh
NF=$(grep -c "CCL_SYCL_ALLREDUCE_FP8=1" $R/serve_p624_r99_fp8.sh || true)
[ "$NF" = "1" ] || { echo "FATAL: FP8 env inject wrong ($NF)"; exit 1; }
boot_serve $LEG $R/serve_p624_r99_fp8.sh fp8
flip_first $LEG/flip_first.out; cat $LEG/flip_first.out
python3 $R/r46t/bench_run7n_t.py --temperature 0 > $LEG/run7n_t0.out 2>&1
grep -hE 'aggregate=|solo200|call_FAIL' $LEG/run7n_t0.out
python3 $R/arc_spec_gate.py > $LEG/spec_gate.out 2>&1
tail -3 $LEG/spec_gate.out
grep -q "SPEC_GATE PASS" $LEG/spec_gate.out || { echo "SPEC_GATE (fp8): FAIL"; GATE_FAIL=1; }
erraudit $LEG
docker exec arc-test cat /root/serve_full.log > $LEG/serve_full.log 2>/dev/null || true
# in-situ wire proof (STATS one-shot boot)
kill_arc
cp $R/serve_p624_r99_fp8.sh $R/serve_p624_r99_stats.sh
sed -i "s|export CCL_SYCL_ALLREDUCE_FP8=1|export CCL_SYCL_ALLREDUCE_FP8=1; export CCL_SYCL_ALLREDUCE_FP8_STATS=1|" $R/serve_p624_r99_stats.sh
NS=$(grep -c "FP8_STATS=1" $R/serve_p624_r99_stats.sh || true)
[ "$NS" = "1" ] || { echo "FATAL: STATS inject wrong ($NS)"; exit 1; }
mkdir -p $CT/statsprobe
boot_serve $CT/statsprobe $R/serve_p624_r99_stats.sh stats
python3 - > $CT/statsprobe/probe.out 2>&1 << 'PYEOF' || true
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
cat $CT/statsprobe/probe.out
docker exec arc-test sh -c "grep -aiE 'CCL fp8 (allreduce|stage)' /root/serve_full.log | tail -40" > $CT/statsprobe/stats_lines.out || true
echo "fp8 stats lines: $(wc -l < $CT/statsprobe/stats_lines.out)"; head -14 $CT/statsprobe/stats_lines.out
[ "$(wc -l < $CT/statsprobe/stats_lines.out)" -ge 1 ] && echo "FP8_WIRE_ON_CERT10: PROVEN" \
  || { echo "FP8_WIRE_ON_CERT10: FAIL (no stats lines)"; GATE_FAIL=1; }
erraudit $CT/statsprobe
docker exec arc-test cat /root/serve_full.log > $CT/statsprobe/serve_stats.log 2>/dev/null || true

echo "=== [4] A/B/A/B perf interleave: p65 faces off/on/off/on $(date +%H:%M:%S) ==="
perf_phase() {  # $1 = tag, $2 = inj ("" off | on)
  local TAG=$1 INJ="$2" LEG=$CT/perf_$1 SCRIPT
  mkdir -p $LEG
  echo "--- perf phase $TAG ($(date +%H:%M:%S)) ---"
  if [ -z "$INJ" ]; then
    SCRIPT=$R/serve_arc_b70.sh
  else
    SCRIPT=$R/serve_p624_r99_perf$TAG.sh
    cp $R/serve_arc_b70.sh $SCRIPT
    sed -i "s|export CCL_SYCL_ALLREDUCE_TMP_BUF=0|export CCL_SYCL_ALLREDUCE_TMP_BUF=0; export CCL_SYCL_ALLREDUCE_FP8=1|" $SCRIPT
    grep -q "CCL_SYCL_ALLREDUCE_FP8=1" $SCRIPT || { echo "FATAL: perf env inject failed"; exit 1; }
  fi
  boot_serve $LEG $SCRIPT perf$TAG
  python3 $R/p65/p65_timeline.py $LEG/timeline.json > $LEG/timeline_stdout.txt 2>&1 || echo "timeline exit=$?"
  grep -A5 'long-stream faces' $LEG/timeline_stdout.txt || tail -15 $LEG/timeline_stdout.txt
}
perf_phase off1 ""
perf_phase on1  "on"
perf_phase off2 ""
perf_phase on2  "on"

echo "=== [V] perf verdict $(date +%H:%M:%S) ==="
python3 - > verdict_perf.txt 2>&1 << 'PYEOF' || true
import re

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
cat verdict_perf.txt
grep -q "R99_CERT10_PERF: PASS" verdict_perf.txt || GATE_FAIL=1

echo "=================== P624 R99B COMPLETE $(date +%F\ %H:%M:%S) GATE_FAIL=$GATE_FAIL ==================="
exit $GATE_FAIL
