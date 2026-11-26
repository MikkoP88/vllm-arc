#!/bin/bash
# r245a_p128_offline.sh -- P6.47 P-C1 (r245a): offline XPU validation of
# the MX-NATIVE paged-decode PAGE-128 admission at the gemma full_
# attention geometry (D=512 page 128 G=8), inside cert28 with the
# build6 (r245a) arc7mx wheel installed over the image venv (NO
# engine boot). The gemma hybrid KV manager banks the D=512 group a
# PAGE-128 pool (the sliding D=256 group rides page 64); the r245 e2e
# died at the first real request with "block_size 128 not yet
# compiled" while the sliding layers' capture-time windowed calls had
# already succeeded. r245a = the TORCH_CHECK admission + a dispatch
# branch (head_size==512 && block_size==128) riding the same _512/_16
# sub-tile frame (mainloop page indirection runtime-generic). Faces:
# exact (fp64, all 3 fmts + v2 triangle when the Triton decoder serves
# page 128 -- availability-probed), bitwise (seq_len==1 dequant bit
# law), boundary (page-128 edge lens 1/63/64/127/128/129/255/256 --
# the page_local_idx page transitions), splits (forced-8 + default at
# ctx 8192 -- the r242d 16-tile plan x the 8-sub-tile page walk),
# regression guards on the SAME WHEEL (d512p64 = the r242 authority;
# ctrl256p64 = the certified qwen production lane -- kernel-binary-
# layout-dice law), telemetry (native timings at page 128 vs the r242
# page-64 numbers; the e2e expectation-setter). Gates: the p242 face
# law (d<=2e-3 or bf16-rounded 1-ulp match; cos>=0.99999; bitwise 0
# mismatches). RUN 1 (2026-11-26 11:20-11:30Z, harness rev 1,
# preserved in logs/r245a_run1.out): everything green except the
# boundary face -- d=2.27e-3, 13 pct over the absolute band, on a
# payload whose worst element is a 7e-7 near-cancellation sum plus a
# single bf16 output ulp at a 0.58-magnitude element; the mini ledger
# proved it intrinsic (d(F32,ref) == the face's d; n(steps>1) == 1)
# while splits walked 512 page-local transitions/seq at d=2.44e-4.
# Rev 2 threads the v2 triangle into the boundary face (the r198
# composite, as face_exact always had) -- the gap was the missing
# calibration, not the gate.
# Prereq: /root/build/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-*.whl
# = build6/r245a (build34: build5/r243a + the page-128 admission; md5
# recorded by r245a_build.sh in ${WHEEL}.r245a.md5 = 95451e404b463d9d
# 6efa4bd4a4939b26). Failure evidence of the r245 first run preserved
# at /root/build/p647/r245_pre_r245a_failrun (page-128 traceback +
# 37 n_ok=0 face JSONs behind green CONC_OK lines -- the
# existence-only-green-gate lesson). Work tree staged at
# /root/build/p242/work by the r242 ship step (repo subset: harness +
# vllm_gaudi/xpu/patches/mx_kv, md5-verified). W14 quiesce before,
# trap restore after (rig law; fullres recipe). Artifacts ->
# /root/build/p245a/{out,logs}. Exit 0 = r245a offline PASS.
set -uo pipefail
R=/root/build
P=$R/p245a
IMG=vllm-arc:v0.30.0-b70-cert28
CNAME=arc-r245a
# work tree: the r242 staging (repo subset with the mx_kv package)
W=$R/p242/work

WHEEL=$(ls $R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || { echo "R245A_NO_WHEEL (run the build first)"; exit 1; }
WMD5=$(md5sum "$WHEEL" | cut -d" " -f1)
EMD5=$(cat "$WHEEL.r245a.md5" 2>/dev/null | cut -d" " -f1)
[ -n "$EMD5" ] || { echo "R245A_NO_BUILD6_MD5 (r245a_build.sh has not completed)"; exit 1; }
[ "$WMD5" = "$EMD5" ] || { echo "R245A_WHEEL_MD5_MISMATCH $WMD5 != build6 $EMD5"; exit 1; }
echo "R245A_WHEEL build6 md5=$WMD5"
[ -f "$W/tools/ar/p147_mxkv/p245a_p128_offline.py" ] || { echo "R245A_NO_HARNESS -- ship first"; exit 1; }
[ -d "$W/vllm_gaudi/xpu/patches/mx_kv" ] || { echo "R245A_NO_MXKV_TREE -- ship first"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R245A_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R245A_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R245A_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R245A_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  # canonical full recreate (docker start + serve_only can never
  # restore the actor-recreated radiance-test -- r185b/r186 law)
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p245a.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r245a] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R245A_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r245a] offline harness in $IMG + build6 wheel md5=$WMD5 $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
     && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p245a_p128_offline.py /out" \
  > $P/logs/r245a.out 2>&1
rc=$?
echo "=== [r245a] harness exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[face|^  ok |^  FAIL|^  note|^  mini|VERDICT|telemetry/" $P/logs/r245a.out > $P/out/r245a_summary.txt || true
cat $P/out/r245a_summary.txt

cd $P/out
sha256sum r245a_summary.txt p245a_p128.json $P/logs/r245a.out 2>/dev/null > SHA256SUMS || true
echo "R245A_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
