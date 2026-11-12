#!/bin/bash
# r153_mxkv_diag.sh -- P6.47 leg B: single SPEC=1 diagnostic boot
# (DUMP=1 TRACE=1, TAG=trace17). trace16 proved: 20 sentinel touches,
# ONE payload, ZERO filter/save failures -- the consumption path is
# invisible. This boot logs MXKVDIAG (every attention call's ord/mql/
# sl0 while a sentinel is pending) + full MXKVTRACE (store/prefill/
# decode paths with impl ids -- the mid-generation re-store census),
# then banks both + the sentinel state.
# Artifacts -> /root/build/p153. Exit 0 = boot + probes PASS.
set -uo pipefail
R=/root/build
P=$R/p153

mkdir -p $P/logs

echo "=== [r153] SPEC=1 diag boot (DUMP+TRACE) $(date +%H:%M:%S) ==="
env FMT=mxfp8 SPEC=1 GRAPH=0 MAXLEN=32768 TRACE=1 DUMP=1 TAG=trace17 \
  bash $R/p149_boot.sh > $P/logs/spec.out 2>&1
rc=$?
grep -E "PROMPT_OK|POOL_DUMP|R149_" $P/logs/spec.out || true

echo "=== [r153] diag greps $(date +%H:%M:%S) ==="
L=$R/p149/trace17/serve_full.log
grep -c "MXKVDIAG" $L > $P/diag_count.txt 2>/dev/null || true
grep "MXKVDIAG" $L | head -80 > $P/diag_head.txt 2>/dev/null || true
grep -E "MXKVTRACE (store|path)" $L | grep -E "58[0-9][0-9]|59[0-9][0-9]" | head -60 > $P/store_slots.txt 2>/dev/null || true
cp $R/p149/trace17/sentinel_state.txt $P/ 2>/dev/null || true
cat $P/diag_count.txt $P/diag_head.txt 2>/dev/null | head -40
echo "R153_DIAG_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
