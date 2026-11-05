#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# r98_quality.sh — P6.24 phase 2: QUALITY LADDER on the patched lib (FP8=1)
# vs the cert9 baseline (plan P6.24 r98 row; suite = the t5/t9-class
# instruments that certified cert8/cert9: run7n t0 + t1.0-xhigh, spec gate,
# tps solo+agg8, integrity/det A-gates, p618 LONG-RETURN full + 3 churn
# cycles — r84 [1]/[2] byte-pattern).
#
# Legs (ONE GPU window, cert9 image, same container, same caches):
#   [0] preflight: stage-arc fat sha == pinned v4 (3a120e8b...), serve
#       script PRISTINE (no CCL_SYCL_ALLREDUCE_FP8 anywhere — baseline law),
#       harness present, image id banked.
#   [A] BASELINE: stock cert9 bytes (pristine serve_arc_b70.sh, stock
#       libs) — same-day A-leg for the parity table.
#   [B] FP8: identical container + docker-cp overlay of the v4 fat lib to
#       /opt/venv/lib/libccl.so.1 AND .so.1.0 (both regular files on cert9;
#       r96/r97 overlay law), serve bytes differ by EXACTLY ONE line:
#       export CCL_SYCL_ALLREDUCE_FP8=1 (MIN_COUNT default 2097152 keeps
#       every decode/graph-replayed AR on the stock path — only prefill
#       chunk ARs go e4m3; STATS off).
#   [C] stats probe: FP8=1 + STATS=1 boot, ONE long prefill — direct
#       in-situ proof the wire fired (declared perturbation leg; suite
#       already banked, nothing downstream measures on it).
#   [V] pre-registered parity bars (verdict python):
#     HARD  within-leg LONG-RETURN per leg: single hash per class across
#           full + 3 churn/rehit cycles (fp8 wire is deterministic: CAS-amax
#           is order-independent, converters pure), zero garble flags
#           (REP-LOOP/JUNK/BLEED/EMPTY), finish=stop everywhere;
#     HARD  leg-B A-gates: integrity 5/5 + determinism 3/3;
#     HARD  erraudit ERROR=0 TRACEBACK=0 both legs;
#     GATE  spec: leg-B SPEC_GATE PASS and |pos1_B - pos1_A| <= 0.03;
#     GATE  tps: solo1/solo2/agg8 leg-B/leg-A ratio in [0.90, 1.10];
#     GATE  run7n: t0 agg + t0 solo + t1.0-xhigh agg ratio in [0.90, 1.10],
#           ok_streams 4/4 everywhere;
#     PROOF stats probe shows k1..k5 stage lines in-situ (count > 0);
#     NOTE  cross-leg t0 hash drift vs baseline is REPORTED, not failed:
#           e4m3 prefill noise (rel-rms 0.023, r97 noise-floor law) is
#           expected to flip near-tie argmax tokens; coherence is gated by
#           the zero-garble + finish-stop bars above (Class-A precedent:
#           cert7 t1-seed, cert8 identity faces).
# W14 restored by trap (rig law).
set -uo pipefail
R=/root/build
CT=$R/p624/r98
STAGE=$R/p624/stage-arc
IMG=vllm-arc:v0.30.0-b70-cert9
FAT_SHA=3a120e8b352fbd6be45ab61a7f4e927d80c9bc6618ff646e6db68b0263eca4a7
GATE_FAIL=0

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p624c.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 20

mkdir -p $CT
cd $CT

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

tps_probe() {  # $1 = outfile (solo x2 + agg8, 512 tok — r84 format)
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

garble_check() {  # $1 = longprobe out file
  local lines empty
  lines=$(grep -cE "^\[(essay|code|multi|simple)\]" "$1" || true)
  empty=$(grep -c "garble={'cold': \[\], 'rehit': \[\], 'rehit2': \[\]\]}" "$1" || true)
  echo "GARBLE: class_lines=$lines empty_garble=$empty (expect equal, >=4) in $(basename "$1")"
  { [ "$lines" -ge 4 ] && [ "$lines" = "$empty" ]; } || { echo "ZERO-GARBLE: FAIL ($1)"; GATE_FAIL=1; }
}

longreturn_summary() {  # $1 = label (r98a|r98b) -> longreturn_summary.txt in cwd
  python3 - "$1" > longreturn_summary.txt 2>&1 << 'PYEOF' || true
import json, sys
tag = sys.argv[1]
def load(fn):
    with open(fn) as f:
        return json.load(f)
full = load(f"longprobe_{tag}_full.json")
rehits = [load(f"longprobe_{tag}_rehit{c}.json") for c in (1, 2, 3)]
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
print(f"{tag.upper()}_LONGRETURN_BAR:", "PASS" if not fail else f"FAIL {fail}")
PYEOF
  cat longreturn_summary.txt
  grep -q "LONGRETURN_BAR: PASS" longreturn_summary.txt \
    || { echo "LONGRETURN: FAIL ($1)"; GATE_FAIL=1; }
}

suite() {  # $1 = leg dir, $2 = longprobe label, $3 = churn seed base
  local LEG=$1 TAG=$2 SEEDBASE=$3
  mkdir -p $LEG && cd $LEG
  echo "--- suite $TAG: flip + integrity + run7n + spec + tps $(date +%H:%M:%S) ---"
  flip_first flip_first.out; cat flip_first.out
  python3 $R/r42/r42_integrity.py > integrity.out 2>&1
  I5=$(grep -c "INTEGRITY 5/5 PASS" integrity.out || true)
  D3=$(grep -c "DETERMINISM 3/3 PASS" integrity.out || true)
  echo "A-gates: integrity_5of5=$I5 det_3of3=$D3"
  { [ "$I5" = "1" ] && [ "$D3" = "1" ]; } || { echo "A-GATES: FAIL ($TAG)"; cat integrity.out; GATE_FAIL=1; }
  python3 $R/r46t/bench_run7n_t.py --temperature 0 > run7n_t0.out 2>&1
  grep -hE 'aggregate=|solo200|call_FAIL' run7n_t0.out
  python3 $R/r46t/bench_run7n_t.py --temperature 1.0 --thinking xhigh > run7n_t1.out 2>&1
  grep -hE 'aggregate=|solo200|call_FAIL' run7n_t1.out
  python3 $R/arc_spec_gate.py > spec_gate.out 2>&1
  tail -3 spec_gate.out
  grep -q "SPEC_GATE PASS" spec_gate.out || { echo "SPEC_GATE: FAIL ($TAG)"; GATE_FAIL=1; }
  tps_probe tps.out; cat tps.out
  echo "--- suite $TAG: LONG-RETURN full + 3 churn cycles $(date +%H:%M:%S) ---"
  python3 $R/p618/p618_longprobe.py $TAG full > longprobe_full.out 2>&1
  grep -E "^\[|LONGPROBE" longprobe_full.out
  garble_check longprobe_full.out
  local cyc
  for cyc in 1 2 3; do
    churn $(( SEEDBASE * cyc ))
    python3 $R/p618/p618_longprobe.py $TAG rehit > longprobe_rehit$cyc.out 2>&1
    grep -E "^\[|LONGPROBE" longprobe_rehit$cyc.out
    garble_check longprobe_rehit$cyc.out
  done
  longreturn_summary $TAG
  if grep -q "T1_SEED_MISS_CLASSES: \['" longreturn_summary.txt; then
    echo "--- t1-seed annex re-characterization (cert7/8/9 precedent) ---"
    python3 $R/p618/p618d_t1seed_char.py > t1seed_char.out 2>&1
    tail -8 t1seed_char.out
  fi
  erraudit $LEG
  docker exec arc-test cat /root/serve_full.log > serve_full.log 2>/dev/null || true
  cd $CT
}

boot_leg() {  # $1 = serve script path — boot + health + posture
  local SCRIPT=$1
  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
    bash $SCRIPT > boot.out 2>&1
  grep ARC_SERVE_LAUNCHED boot.out || { echo "FATAL: launch failed"; cat boot.out; exit 1; }
  wait_health | tee -a boot.out
  docker exec arc-test sh -c "grep -E 'KV cache size|v6 installed' /root/serve_full.log | head -2" > posture.txt || true
  cat posture.txt
}

echo "=================== P624 R98 QUALITY LADDER START $(date +%F\ %H:%M:%S) ==================="

echo "=== [0] preflight: stage sha + pristine serve + harness + image $(date +%H:%M:%S) ==="
{
  echo "image_id=$(docker image inspect $IMG --format '{{.Id}}')"
  sha256sum $STAGE/libccl.so.1.0
  echo "stage_md5=$(md5sum $STAGE/libccl.so.1.0 | awk '{print $1}')"
  echo "serve_pristine_fp8_mentions=$(grep -c CCL_SYCL_ALLREDUCE_FP8 $R/serve_arc_b70.sh || true)"
  for f in $R/arc_spec_gate.py $R/r46t/bench_run7n_t.py $R/r42/r42_integrity.py \
           $R/p618/p618_longprobe.py $R/p618/p618d_t1seed_char.py $R/w14_boot.sh; do
    [ -f "$f" ] && echo "OK  $f" || echo "MISS $f"
  done
} | tee preflight.txt
SHA_NOW=$(sha256sum $STAGE/libccl.so.1.0 | awk '{print $1}')
[ "$SHA_NOW" = "$FAT_SHA" ] || { echo "FATAL: stage fat sha drift ($SHA_NOW)"; exit 1; }
STAGE_MD5=$(md5sum $STAGE/libccl.so.1.0 | awk '{print $1}')
PRISTINE=$(grep -c CCL_SYCL_ALLREDUCE_FP8 $R/serve_arc_b70.sh || true)
[ "$PRISTINE" = "0" ] || { echo "FATAL: serve_arc_b70.sh not pristine (FP8 env present)"; exit 1; }
grep -q "MISS" preflight.txt && { echo "FATAL: harness missing"; exit 1; }

echo "=== [A] BASELINE leg: stock cert9, pristine bytes $(date +%H:%M:%S) ==="
docker rm -f arc-test >/dev/null 2>&1 || true
SEEDBASE=100
mkdir -p legA legB
( cd legA && boot_leg $R/serve_arc_b70.sh )
docker exec arc-test md5sum /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so.1.0 > legA/lib_md5_stock.txt
cat legA/lib_md5_stock.txt
suite $CT/legA r98a 100

echo "=== [B] FP8 leg: overlay v4 + exactly one env delta $(date +%H:%M:%S) ==="
docker exec arc-test pkill -f "vllm serve" >/dev/null 2>&1 || true
sleep 12
docker cp $STAGE/libccl.so.1.0 arc-test:/opt/venv/lib/libccl.so.1.0
docker cp $STAGE/libccl.so.1.0 arc-test:/opt/venv/lib/libccl.so.1
docker exec arc-test md5sum /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so.1.0 > legB/lib_md5_overlay.txt
cat legB/lib_md5_overlay.txt
docker exec arc-test md5sum /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so.1.0 \
  | awk '{print $1}' | grep -v "$STAGE_MD5" | grep -q . \
  && { echo "FATAL: overlay md5 mismatch"; exit 1; } \
  || echo "OVERLAY_MD5: OK (both paths == stage fat $STAGE_MD5)"
cp $R/serve_arc_b70.sh $R/serve_p624_r98fp8.sh
sed -i "s|export CCL_SYCL_ALLREDUCE_TMP_BUF=0|export CCL_SYCL_ALLREDUCE_TMP_BUF=0; export CCL_SYCL_ALLREDUCE_FP8=1|" $R/serve_p624_r98fp8.sh
ND=$(grep -c "CCL_SYCL_ALLREDUCE_FP8=1" $R/serve_p624_r98fp8.sh || true)
[ "$ND" = "1" ] || { echo "FATAL: FP8 env injection wrong ($ND occurrences)"; exit 1; }
diff <(tr -d '\r' < $R/serve_arc_b70.sh) <(tr -d '\r' < $R/serve_p624_r98fp8.sh) | tee legB/serve_diff.txt
( cd legB && boot_leg $R/serve_p624_r98fp8.sh )
docker exec arc-test sh -c 'grep -m1 "KV cache size" /root/serve_full.log' > legB/posture_kv.txt || true
cat legB/posture_kv.txt
suite $CT/legB r98b 400

echo "=== [C] stats probe: FP8=1 + STATS=1, one long prefill (mechanism proof) $(date +%H:%M:%S) ==="
docker exec arc-test pkill -f "vllm serve" >/dev/null 2>&1 || true
sleep 12
cp $R/serve_p624_r98fp8.sh $R/serve_p624_r98stats.sh
sed -i "s|export CCL_SYCL_ALLREDUCE_FP8=1|export CCL_SYCL_ALLREDUCE_FP8=1; export CCL_SYCL_ALLREDUCE_FP8_STATS=1|" $R/serve_p624_r98stats.sh
NS=$(grep -c "FP8_STATS=1" $R/serve_p624_r98stats.sh || true)
[ "$NS" = "1" ] || { echo "FATAL: STATS env injection wrong ($NS)"; exit 1; }
mkdir -p statsprobe && cd statsprobe
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_p624_r98stats.sh > boot.out 2>&1
grep ARC_SERVE_LAUNCHED boot.out || { echo "FATAL: stats boot failed"; cat boot.out; exit 1; }
wait_health | tee -a boot.out
python3 - > probe.out 2>&1 << 'PYEOF' || true
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
cat probe.out
docker exec arc-test sh -c "grep -aiE 'fp8 (allreduce|stage)' /root/serve_full.log | tail -40" > stats_lines.out || true
echo "fp8 stats lines captured: $(wc -l < stats_lines.out)"
head -14 stats_lines.out
docker exec arc-test sh -c "grep -aicE 'fp8 (allreduce|stage)' /root/serve_full.log" > stats_count.txt || true
[ "$(cat stats_count.txt)" -ge 1 ] \
  && echo "FP8_WIRE_IN_SITU: PROVEN ($(cat stats_count.txt) lines)" \
  || { echo "FP8_WIRE_IN_SITU: FAIL (no stats lines — wire never fired)"; GATE_FAIL=1; }
docker exec arc-test cat /root/serve_full.log > serve_stats.log 2>/dev/null || true
cd $CT

echo "=== [V] parity verdict $(date +%H:%M:%S) ==="
python3 - > verdict.txt 2>&1 << 'PYEOF' || true
import json, re, sys

def rd(p):
    with open(p, errors="replace") as f:
        return f.read()

def tps(p):
    d = {}
    for k, v in re.findall(r"(solo[12]|agg8): tps=([0-9.]+)", rd(p)):
        d[k] = float(v)
    return d

def run7n(p):
    aggs = [float(x) for x in re.findall(r"aggregate=([0-9.]+)", rd(p))]
    solos = [float(x) for x in re.findall(r"solo200 .*tps=([0-9.]+)", rd(p))]
    ok = all(f"ok_streams=4/4" in l for l in rd(p).splitlines() if "aggregate=" in l)
    return (sum(aggs) / len(aggs) if aggs else 0.0,
            sum(solos) / len(solos) if solos else 0.0, ok and len(aggs) == 2)

def spec(p):
    m = re.search(r"SPEC_GATE PASS: pos1=([0-9.]+)", rd(p))
    return float(m.group(1)) if m else None

def lr(tag):
    txt = rd(f"leg{'A' if tag == 'r98a' else 'B'}/longreturn_summary.txt")
    m = re.search(r"LONGRETURN_BAR: (PASS|FAIL)", txt)
    return m.group(1) == "PASS" if m else False

fails, notes = [], []
ta, tb = tps("legA/tps.out"), tps("legB/tps.out")
for k in ("solo1", "solo2", "agg8"):
    if k in ta and k in tb:
        r = tb[k] / ta[k] if ta[k] else 0
        line = f"tps {k}: A={ta[k]:.2f} B={tb[k]:.2f} ratio={r:.3f}"
        if not 0.90 <= r <= 1.10:
            fails.append(line + " OUT of [0.90,1.10]")
        print(line)
for name, fa, fb in (("t0", "legA/run7n_t0.out", "legB/run7n_t0.out"),
                     ("t1.0-xhigh", "legA/run7n_t1.out", "legB/run7n_t1.out")):
    a_agg, a_solo, a_ok = run7n(fa)
    b_agg, b_solo, b_ok = run7n(fb)
    if not (a_ok and b_ok):
        fails.append(f"run7n {name}: stream health bad (A ok={a_ok} B ok={b_ok})")
    if a_agg > 0:
        r = b_agg / a_agg
        line = f"run7n {name} agg: A={a_agg:.2f} B={b_agg:.2f} ratio={r:.3f} (solo A={a_solo:.2f} B={b_solo:.2f})"
        if not 0.90 <= r <= 1.10:
            fails.append(line + " agg OUT of [0.90,1.10]")
        print(line)
pa, pb = spec("legA/spec_gate.out"), spec("legB/spec_gate.out")
print(f"spec pos1: A={pa} B={pb}")
if pb is None:
    fails.append("spec: leg-B SPEC_GATE no PASS line")
elif pa is not None and abs(pb - pa) > 0.03:
    fails.append(f"spec: pos1 delta {pb - pa:+.4f} exceeds 0.03")
print("LONGRETURN within-leg:", "A:" + ("PASS" if lr('r98a') else "FAIL"),
      "B:" + ("PASS" if lr('r98b') else "FAIL"))
if not lr("r98a"):
    fails.append("LONG-RETURN leg-A (baseline!) drift — investigate rig first")
if not lr("r98b"):
    fails.append("LONG-RETURN leg-B within-leg bar")

# cross-leg hash drift: REPORTED only (pre-registered disposition)
def hashes(tag):
    with open(f"leg{'A' if tag == 'r98a' else 'B'}/longprobe_{tag}_full.json") as f:
        return {c: set(json.load(f).get(c, {}).get("hashes", []))
                for c in ("essay", "code", "multi", "simple")}
ha, hb = hashes("r98a"), hashes("r98b")
drift = [c for c in ha if ha[c] and hb[c] and ha[c] != hb[c]]
print("cross-leg t0 hash drift classes:", drift if drift else "none (byte-identical t0)")
if drift:
    notes.append(f"cross-leg drift {drift}: expected e4m3 prefill noise 0.023 rel-rms "
                 "flipping near-tie argmax tokens; coherence gated by zero-garble + "
                 "finish-stop (Class-A precedent)")

for n in notes:
    print("NOTE:", n)
print("R98_QUALITY_LADDER:", "PASS" if not fails else f"FAIL {fails}")
PYEOF
cat verdict.txt
grep -q "R98_QUALITY_LADDER: PASS" verdict.txt || GATE_FAIL=1

echo "=================== P624 R98 COMPLETE $(date +%F\ %H:%M:%S) GATE_FAIL=$GATE_FAIL ==================="
exit $GATE_FAIL
