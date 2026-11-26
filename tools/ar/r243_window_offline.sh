#!/bin/bash
# r243_window_offline.sh -- P6.47 P-C2 (r243): offline XPU validation of
# the MX-NATIVE paged-decode SLIDING-WINDOW surface (the masked kernel
# twin, window pair (W-1,0) threaded by P-C2 through binding/op schema/
# python interface) at the gemma sliding geometry (D=256 page 64 G=8,
# W=1024) + the qwen-geometry masked twin (D=256 page 16 G=6), inside
# cert28 with the build5 (r243a) arc7mx wheel installed over the image
# venv (NO engine boot). Faces: exact (window-masked fp64, W=64
# binding + the d_blind window-blind discriminator), nonbinding
# (lens<=W: windowed vs windowless twin), boundary (W=64 lens
# 1/32/63/64/65/127/128; W=1024 lens 1000/1023/1024/1025), bitwise
# (seq_len==1 dequant bit law on the masked twin), splits (forced 8 +
# auto at ctx 8192, W=64 and W=2048 -- k_block0 walk-bound x
# ReduceSplitK; the W=2048 cell is the genuine 8-way windowed split
# and the pair is the r243a windowed-plan regression: the pre-fix
# full-length plan walked tiles past k_blocks -> mxfp4 DEVICE_LOST),
# widewin (W=1024 at ctx 8192/32768/40960, mxfp4 -- the gemma lane
# shape), telemetry (windowed vs windowless walk cost). Gates: the
# p243 probe's face law (d<=2e-3 or bf16-rounded 1-ulp match;
# cos>=0.99999; bitwise 0 mismatches).
# Prereq: /root/build/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-*.whl
# = build5/r243a (build33: build4 [build30 r242b V-slice fix + P-C2
# window surface + build31 schema-arg interface fix + build32 r242d
# split-plan kv_tile=16 fix] + the r243a windowed work-list fix: the
# interface prices the plan from the windowed tile extent AND the
# kernel clamps the work-list walk to k_blocks; md5 recorded by
# r243a_build.sh in ${WHEEL}.r243a.md5). Work tree staged at
# /root/build/p243/work by the r243 ship step (repo subset: harness +
# vllm_gaudi/xpu/patches/mx_kv, md5-verified). W14 quiesce before, trap
# restore after (rig law; fullres recipe). Artifacts -> /root/build/p243/
# {out,logs}. Exit 0 = r243 offline PASS.
set -uo pipefail
R=/root/build
P=$R/p243
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert28
CNAME=arc-r243

WHEEL=$(ls $R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || { echo "R243_NO_WHEEL (run the build first)"; exit 1; }
WMD5=$(md5sum "$WHEEL" | cut -d" " -f1)
EMD5=$(cat "$WHEEL.r243a.md5" 2>/dev/null | cut -d" " -f1)
[ -n "$EMD5" ] || { echo "R243_NO_BUILD5_MD5 (r243a_build.sh has not completed)"; exit 1; }
[ "$WMD5" = "$EMD5" ] || { echo "R243_WHEEL_MD5_MISMATCH $WMD5 != build5 $EMD5"; exit 1; }
echo "R243_WHEEL build5 md5=$WMD5"
[ -f "$W/tools/ar/p147_mxkv/p243_window_offline.py" ] || { echo "R243_NO_HARNESS -- ship first"; exit 1; }
[ -d "$W/vllm_gaudi/xpu/patches/mx_kv" ] || { echo "R243_NO_MXKV_TREE -- ship first"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R243_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R243_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R243_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R243_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  # canonical full recreate (docker start + serve_only can never
  # restore the actor-recreated radiance-test -- r185b/r186 law)
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p243.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r243] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R243_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r243] offline harness in $IMG + build5 wheel md5=$WMD5 $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
     && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p243_window_offline.py /out" \
  > $P/logs/r243.out 2>&1
rc=$?
echo "=== [r243] harness exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[face|^  ok |^  FAIL|^  note|^  mini|VERDICT|telemetry/" $P/logs/r243.out > $P/out/r243_summary.txt || true
cat $P/out/r243_summary.txt

cd $P/out
sha256sum r243_summary.txt p243_window.json $P/logs/r243.out 2>/dev/null > SHA256SUMS || true
echo "R243_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
