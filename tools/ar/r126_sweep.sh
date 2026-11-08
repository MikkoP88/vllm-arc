#!/bin/bash
# r126_sweep.sh -- P6.33 v2.4 probe knob sweep window.
# Boots the arc6test image + PROBE wheel (ephemeral pip --force-reinstall
# INSIDE the throwaway container; the image is untouched), runs
# r126_sweep.py (10 combos x 6 shapes x 4 lanes, M=1 + M=8 spots,
# interleaved paired A/B + per-cell correctness), banks raw to
# /root/build/p633/, restores W14 (trap).
set -uo pipefail
R=/root/build
P=$R/p633
IMG6=vllm-arc:v0.30.0-b70-arc6test
WHL=/root/build/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc6-cp312-cp312-linux_x86_64.whl

mkdir -p "$P"
[ -f "$P/r126_sweep.py" ] || { echo "FATAL: sweep script missing"; exit 1; }
[ -f "$WHL" ] || { echo "FATAL: probe wheel missing: $WHL"; exit 1; }
docker image inspect $IMG6 >/dev/null 2>&1 || {
  echo "FATAL: $IMG6 not built"; exit 1; }

w14_start() {
  docker rm -f arc-sweep >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p633sweep.log 2>&1 < /dev/null &
}
trap w14_start EXIT

WH=""
for i in $(seq 1 40); do
  WH=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health 2>/dev/null || true)
  [ "$WH" = "200" ] && break
  sleep 60
done
echo "W14 health before sweep: $WH ($(date +%H:%M:%S))"

docker stop radiance-test >/dev/null 2>&1 || true
sleep 10
docker rm -f arc-sweep >/dev/null 2>&1 || true

echo "=== [sweep] boot $(date +%H:%M:%S) ==="
# Env law (serve_arc_b70.sh exec block precedent): python needs
# LD_LIBRARY_PATH=venv/lib:torch/lib. The probe wheel is pip-installed
# INSIDE this throwaway container only (image untouched; the certified
# wheel gets constants baked, never these env reads).
docker run --rm --name arc-sweep \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$P":/work -v "$R/wheels-arc-vxk":/wheels:ro \
  --entrypoint /bin/bash \
  $IMG6 -c "/opt/venv/bin/pip install -q --force-reinstall --no-deps /wheels/$(basename $WHL) && exec /opt/venv/bin/python /work/r126_sweep.py" \
  > "$P/r126_sweep.out" 2>&1
rc=$?
echo "=== [sweep] exit rc=$rc $(date +%H:%M:%S) ==="

grep -m1 "correctness:" "$P/r126_sweep.out" || true
grep -A7 "per-combo M=1 mean" "$P/r126_sweep.out" || true
echo "=== [sweep] window done; W14 restoring ==="
exit $rc
