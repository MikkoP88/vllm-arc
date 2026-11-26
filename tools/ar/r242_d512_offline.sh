#!/bin/bash
# r242_d512_offline.sh -- P6.47 P-C1 (r242): offline XPU validation of
# the MX-NATIVE paged-decode kernel at the gemma full_attention
# geometry (D=512 page 64 G=8) + the certified qwen control cell
# (D=256 page 64 G=6), inside cert28 with the build5 (r243a) arc7mx
# wheel installed over the image venv (NO engine boot). Faces: exact
# (vs fp64 on dequantized cache), v2 (vs the certified Triton decoder
# on the same cache, availability-probed at D=512), bitwise
# (seq_len==1 dequant bit-math oracle), splits (forced 8-split
# ReduceSplitK + the V-slice x split interplay at ctx 8192 --
# windowless, so the r243a work-list clamp is a proven no-op here:
# full-length plans land inside [0, k_blocks) by construction),
# boundary (block-64 edge lens), ctrl (same-wheel regression guard),
# telemetry (native vs v2 at 1024/8192/32768/40960 -- the gemma
# maxlen). Gates: the p242 probe's face law (d<=2e-3 or bf16-rounded
# 1-ulp match; cos>=0.99999; bitwise 0 mismatches).
# Prereq: /root/build/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-*.whl
# = build5/r243a (build33: build4 [build30 r242b V-slice fix + P-C2
# window surface + build31 schema-arg interface fix + build32 r242d
# split-plan kv_tile=16 fix] + the r243a windowed work-list fix
# [interface prices the plan from the windowed tile extent; kernel
# clamps the work-list walk to k_blocks -- the pre-fix full-length
# plan + k_block0 walked past the block-table row: r243 mxfp4
# DEVICE_LOST]; md5 recorded by r243a_build.sh in ${WHEEL}.r243a.md5).
# The build1 run (md5 3c0b9b163edc103acf3eb55d80c6bdfd, 2026-11-25
# 21:20Z) FAILED 16/16 D=512 faces — the defect discovery: corruption
# confined to V columns > 256 (every blk_v=1 WG re-staged slice 0's
# payload and scale groups 0..7; slice 1 duplicated slice 0,
# cos~0.49); the ctrl cell passed clean. The build4 run PASSED all
# faces (r242b authority, 2026-11-26); this re-run on build5 is the
# r243a regression (windowless byte-equivalence). Work tree staged at
# /root/build/p242/work by the r242 ship step (repo subset: harness +
# vllm_gaudi/xpu/patches/mx_kv, md5-verified). W14 quiesce before,
# trap restore after (rig law; fullres recipe). Artifacts ->
# /root/build/p242/{out,logs}. Exit 0 = r242 offline PASS.
set -uo pipefail
R=/root/build
P=$R/p242
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert28
CNAME=arc-r242

WHEEL=$(ls $R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || { echo "R242_NO_WHEEL (run the build first)"; exit 1; }
WMD5=$(md5sum "$WHEEL" | cut -d" " -f1)
EMD5=$(cat "$WHEEL.r243a.md5" 2>/dev/null | cut -d" " -f1)
[ -n "$EMD5" ] || { echo "R242_NO_BUILD5_MD5 (r243a_build.sh has not completed)"; exit 1; }
[ "$WMD5" = "$EMD5" ] || { echo "R242_WHEEL_MD5_MISMATCH $WMD5 != build5 $EMD5"; exit 1; }
echo "R242_WHEEL build5 md5=$WMD5"
[ -f "$W/tools/ar/p147_mxkv/p242_d512_offline.py" ] || { echo "R242_NO_HARNESS -- ship first"; exit 1; }
[ -d "$W/vllm_gaudi/xpu/patches/mx_kv" ] || { echo "R242_NO_MXKV_TREE -- ship first"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R242_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R242_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R242_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R242_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  # canonical full recreate (docker start + serve_only can never
  # restore the actor-recreated radiance-test -- r185b/r186 law)
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p242.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r242] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R242_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r242] offline harness in $IMG + build5 wheel md5=$WMD5 $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
     && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p242_d512_offline.py /out" \
  > $P/logs/r242.out 2>&1
rc=$?
echo "=== [r242] harness exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[face|^  ok |^  FAIL|^  note|^  mini|VERDICT|telemetry/" $P/logs/r242.out > $P/out/r242_summary.txt || true
cat $P/out/r242_summary.txt

cd $P/out
sha256sum r242_summary.txt p242_d512.json $P/logs/r242.out 2>/dev/null > SHA256SUMS || true
echo "R242_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
