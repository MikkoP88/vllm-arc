#!/bin/bash
# r174_kbench.sh -- P6.47 A14: r172 phase-0 kernel-probe RERUN (the
# 07:17Z phase-0 died on the parents[3] eager-eval landmine before the
# installed-kernels check could win; probe fixed + reshipped 07:2xZ).
#
# Standalone re-leg of the r172 phase-0 block ONLY (e2e legs q8/m8/m4
# are NOT rerun -- r172 banks those): p172_deepprobe in a bare
# cert25z container loading the INSTALLED shipping kernels (v5 default
# m8/mkv4, v2 batch m4). Faces A-E: verify kernel, per-layer aux,
# plan-build chain, mx_store prefill quant, mx_dequant_cache
# continuation re-dequant. Resolves the m4 contradiction (A13
# kernel-math 129.8 ms > observed step 112 ms -- cert24x-era tree vs
# installed kernels) and feeds the A14 attribution table.
#
# After the probe: refresh deepA14 SHA256SUMS + re-run the telemetry
# report (both derived artifacts -- r172 wrote them before kernel data
# existed). The failed original bench_kernel.txt is preserved as
# bench_kernel.phase0fail.txt (evidence immutability).
#
# w14_restore carries the r170b-incident guard (2026-11-13): skip when
# any arc-* container is Up -- an in-flight leg owns the window and its
# own trap restores. GPU probe -- NEVER concurrent with a serving leg.
# Exit 0 = probe rc 0 (finite outputs; timings bank regardless).
set -uo pipefail
R=/root/build
P=${P647_BANK:-$R/p647/deepA14}
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25z}
PROBE=$R/p147/p172_deepprobe.py
CNAME=arc-p174

[ -f "$PROBE" ] || { echo "R174_STALE_TREE (no probe at $PROBE) -- ship first"; exit 1; }
grep -q "len(_here.parents) > 3" "$PROBE" \
  || { echo "R174_STALE_TREE (probe lacks the parents[3] guard -- pre-fix copy)"; exit 1; }
mkdir -p "$P/kbench"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r174.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r174:A14] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R174_WARN_PORT8000_STILL_HELD"
sleep 5

# preserve the failed original (evidence immutability), then rerun
if [ -f "$P/kbench/bench_kernel.txt" ] \
  && grep -q "IndexError" "$P/kbench/bench_kernel.txt" 2>/dev/null; then
  cp "$P/kbench/bench_kernel.txt" "$P/kbench/bench_kernel.phase0fail.txt"
  echo "R174_ORIGINAL_PRESERVED bench_kernel.phase0fail.txt"
fi

docker rm -f $CNAME >/dev/null 2>&1 || true
echo "=== [r174:A14] kernel probe in $IMG $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$R/p147":/p147 -v "$P/kbench":/out \
  --entrypoint /bin/bash \
  $IMG -c 'cp /p147/p172_deepprobe.py /tmp/ && cd /tmp && /opt/venv/bin/python p172_deepprobe.py' \
  > "$P/kbench/bench_kernel.txt" 2>&1
KRC=$?
echo "=== [r174:A14] kernel probe exit rc=$KRC $(date +%H:%M:%S) ==="
tail -8 "$P/kbench/bench_kernel.txt"

# refresh derived artifacts (sums + report) now that kernel data exists
cd "$P"
find . -name 'p139_conc_*.json' -o -name 'acceptance_*.txt' -o -name 'markers.txt' \
  -o -name 'posture.txt' -o -name 'bench_kernel*.txt' | sort | xargs sha256sum > SHA256SUMS 2>/dev/null || true
python3 $R/p647/p147_telemetry_report.py "$P" \
  > "$P/REPORT.md" 2>&1 || echo "REPORT_NONZERO (bank intact)"
echo "R174_KBENCH_DONE rc=$KRC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $KRC
