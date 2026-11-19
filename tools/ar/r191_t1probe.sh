#!/bin/bash
# r191_t1probe.sh -- P6.47 Phase 6 T1 attribution probe (2026-11-19):
# re-run the p172 deepprobe faces on cert25zb (K2/K3 default-on) to bank
# the authoritative per-(G, ctx) kernel/aux/plan tables that the B1
# verdict (r188 m4qv5 e2e) attributes against. The S1 four-lane verdict
# located the goal gap at n1 long-ctx (m4q/mkq/m8q 2.5-2.9x ctrl raw ITL
# at 32K n1, 2.2-2.75x on eff ms/token; ctrl acceptance healthy 78% --
# no acceptance rescue); the ctx-proportional term is the MX decode
# attention chain. Faces needed: A (verify kernel per decoder per
# format: v2 batch + v5 dot, G 1/8 x ctx 8K/32K), B (per-layer aux
# index_select/index_copy_/cast), C (once-per-step plan chain). D/E
# (prefill taxes) ride along -- already dispositioned (p182 Tax-A
# refuted, Phase 5 F1). Probe-only leg: cells bank regardless; exit 0 =
# probe rc 0 + bank present. Runs after r188 completes (its trap owns
# the W14 restore); this leg quiesces W14 itself (port guard BEFORE
# quiesce, r188 law) and restores fullres on exit.
set -uo pipefail
R=/root/build
P=$R/p647/t1A26
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zb}
CNAME=arc-r191
DEEPPROBE=$R/p147/p172_deepprobe.py

[ -f "$DEEPPROBE" ] || { echo "R191_NO_DEEPPROBE $DEEPPROBE"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R191_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R191_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R191_PORT_HELD (unknown holder)"; exit 3
  fi
fi
mkdir -p "$P/kbench"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  docker rm -f arc-p172 >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p647r191.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r191] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R191_WARN_PORT8000_STILL_HELD"
sleep 5

echo "=== [r191] deepprobe in $IMG $(date +%H:%M:%S) ==="
docker rm -f arc-p172 >/dev/null 2>&1 || true
docker run --rm --name arc-p172 \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$R/p147":/p147 -v "$P/kbench":/out \
  --entrypoint /bin/bash \
  $IMG -c "cp /p147/p172_deepprobe.py /tmp/ && cd /tmp && /opt/venv/bin/python p172_deepprobe.py" \
  > "$P/kbench/bench_kernel.txt" 2>&1
KRC=$?
echo "=== [r191] deepprobe rc=$KRC $(date +%H:%M:%S) ==="
tail -8 "$P/kbench/bench_kernel.txt"

cd "$P"
find . -type f -exec sha256sum {} \; > SHA256SUMS 2>/dev/null || true
echo "R191_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $KRC
