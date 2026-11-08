#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# r99c_annex.sh — P6.24 r99 t1 REP-LOOP disposition annex (hard-gate
# evidence law: a FAIL face gets a measured disposition, not an argued one).
#
# Facts (2026-11-05): r99 [1] default-posture LONG-RETURN recomputed from
# the banked jsons (r99def_reconstruct.py) shows ONE t1 garble flag:
# code class, t1 (temperature 1.0, seed 1234, both draws hash 41bebba4201b
# — deterministic within the boot). t0 faces are ALL clean (single hash
# across cold/rehit/rehit2 + 3 churn cycles, zero garble, stop finishes,
# t1_seed_identical=True all classes). Banked history: r84's five cert9
# legs = 40 t1 draws with ZERO flags (but t1 seed reproducibility itself
# is known-sporadic on stock — k3v4/q3nc legs), r98 A/B = 16 draws clean.
# 1/8 vs 0/40 (Fisher p~0.28) does not separate cert10-default from stock.
#
# This annex measures the two questions directly:
#   A) cert10 default, FRESH boot, same probe: does 41bebba4201b + loop
#      reproduce cross-boot on the same image? (stable property vs
#      boot-stochastic)
#   B) cert9 STOCK, same day, same probe: does ANY t1 flag appear on the
#      pristine stock image? (stock basis for the phenomenon)
#   Plus verbatim loop-text capture (r99c_probe_t1.py) whenever a flag
#   reproduces. REPORT-only annex: it dispositions, it does not gate.
set -uo pipefail
R=/root/build
CT=$R/p624/r99
IMG10=vllm-arc:v0.30.0-b70-cert10
IMG9=vllm-arc:v0.30.0-b70-cert9

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p624f.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 20

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

boot_serve() {  # $1 = leg dir, $2 = image, $3 = tag
  local LEG=$1 IMAGE=$2 STAG=$3
  mkdir -p $LEG
  kill_arc
  ( cd $LEG && ARC_IMAGE=$IMAGE ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
      bash $R/serve_arc_b70.sh > boot.out 2>&1 )
  grep ARC_SERVE_LAUNCHED $LEG/boot.out || { echo "FATAL: launch failed ($STAG)"; cat $LEG/boot.out; exit 1; }
  local ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health ($STAG)"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK ($STAG) after ~$((i*5))s" | tee -a $LEG/boot.out
}

leg() {  # $1 = leg dir, $2 = image, $3 = label
  local LEG=$1 IMAGE=$2 LABEL=$3
  boot_serve $LEG $IMAGE $LABEL
  ( cd $LEG && python3 $R/p618/p618_longprobe.py $LABEL full > longprobe_full.out 2>&1 )
  grep -E "^\[|LONGPROBE" $LEG/longprobe_full.out
  ( cd $LEG && python3 $R/p624/r99c_probe_t1.py > t1_redraw.out 2>&1 )
  grep -E "^DRAW[12]: (sha|garble)" $LEG/t1_redraw.out
  docker exec arc-test cat /root/serve_full.log > $LEG/serve_full.log 2>/dev/null || true
}

echo "=================== P624 R99C T1 ANNEX START $(date +%F\ %H:%M:%S) ==================="
leg $CT/r99c/c10 $IMG10 r99c10
leg $CT/r99c/c9  $IMG9  r99c9

echo "=== annex summary $(date +%H:%M:%S) ==="
python3 - << 'PYEOF'
import json
CLASSES = ("essay", "code", "multi", "simple")
for name, legdir, tag in (("cert10-default", "r99c/c10", "r99c10"),
                          ("cert9-stock", "r99c/c9", "r99c9")):
    d = json.load(open(f"{legdir}/longprobe_{tag}_full.json"))
    flags = 0
    for cls in CLASSES:
        v = d[cls]
        f = v.get("t1_garble") or []
        flags += bool(f)
        print(f"{name} {cls}: t1={v['t1_hashes']} seed_identical={v['t1_seed_identical']} t1_garble={f}")
    print(f"{name}: flagged_classes={flags}/4")
print("R99C_ANNEX: report-only — disposition in VERDICT.md")
PYEOF
echo "=================== P624 R99C ANNEX COMPLETE $(date +%F\ %H:%M:%S) ==================="
