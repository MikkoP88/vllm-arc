#!/bin/bash
# r271_cert30_gate.sh = r264_cert29_gate.sh + the cert30 deltas (P6.47
# r270/r271, 2026-11-28). The section-10 gate on the CANDIDATE
# vllm-arc:v0.30.0-b70-cert30 (r270: the r263-rebuilt cert29 + the
# ONE-file delta of the 906e8326 content commit, PR #188 = the r269
# MX-KV decode page fix in mx_kv/triton_mx_decode.py: all three decode
# entry points resolve block_size None -> the cache's OWN page; root
# cause r268c -- gemma's glob pool pages at 128 while the engine call
# sites omitted block_size, so the default-64 walk read past the valid
# table prefix at tokens 64+; fix A/B r269: matrix 323/324 RED ->
# 0/329 GREEN under ARC_GRAPH=1). Deltas vs r264:
#   - image = cert30; manifest fork_sha refuse -> cert30_fork_sha ==
#     the r270 content commit (passed as $1); cert29 (4dd3a1b9) and the
#     rc1 crash image (16effb49) are refused.
#   - image-content markers: + the r269 fix lines (three
#     'block_size: int | None = None' signatures, three
#     'block_size = cache.shape[1]' resolutions) and the INSTALLED
#     triton_mx_decode.py md5 == the content commit's blob
#     (e33aa7b749e34ea350441da2e339ff30).
#   - GEMMA LAWFUL SKIN (owner directive 2026-11-28): every gemma leg
#     boots with ARC_TEMPLATE=chat_template_gemma4.jinja + EMPTY
#     parsers (SKIN_G3 form; no qwen skin can be constructed), and a
#     serve-log SKIN assertion gates each gemma leg (the template line
#     present; r264's gemma legs ran templateless = the posture that
#     made the r264 m8g/m4g/mkg verdicts invalid-by-posture).
#   - LOGPROBS IN THE BATTERY (r268c law): the text battery rides
#     logprobs=1 and gates LP_NAN == 0 per prompt -- the 400-nan
#     corruption class is invisible to shape-only checks.
#   - KERNEL HEALTH per leg (the crash class has a kernel signature the
#     serve log only shows after the fact): dmesg lines since the leg's
#     kernel mark matching Engine reset / devcoredump / GPU hang must be
#     ZERO for every leg (incl. the whole battery at the end).
#   - STRIDE_FIX_ENGAGED on the gemma MX legs: >= 1 "MXKV native bt
#     materialized" line (the fixed lane actually rode a multi-row synth
#     continuation in that leg -- a green leg that never entered the
#     fixed path would be vacuous).
#   - W14: the entry never stops radiance-test inside a restore boot's
#     first minutes (stale-baton law, r262), and the EXIT trap restores
#     W14 through the clean recipe tools/ar/r262_w14_restore.sh (restart,
#     delete the lock-only JIT build dirs, GPU-free wait, serve, health +
#     ONE REAL generation).
#   - battery order: gemma first (gref m8g m4g mkg g0 compare -- the fix's
#     validation), then the qwen legs (ctrl m8q m4q mkq lmc -- the bridge
#     bytes changed, so the qwen side is re-certified on this image).
#     Explicit legs still run in the order given (one bank dir).
#   r264b (pre-launch audit, same day) -- the verdict must be trustworthy
#   on exactly those lanes:
#   - GATE_FAIL is a COUNT; every leg runs inside its LEG_VERDICT line
#     (GREEN = zero failed checks in that leg) and the battery ends with
#     R264_CERT_PRECONDITION: owner directive 2026-11-27 -- the gemma MX
#     lanes (m8g/m4g/mkg) and the LMCache lane (lmc) MUST be all green
#     before a certificate is cut -> CERT_CUT_ALLOWED only when every leg
#     of the battery is GREEN in this bank and no check failed.
#   - the suite-time markers are RE-COUNTED after the suite
#     (*_lines_post): the boot-time snapshot predates every ride (the r262
#     line fires at the first multi-row synth continuation, DEVICE_LOST
#     only at a crash) -- the snapshot alone was a vacuous fail for the
#     stride fix and a vacuous pass for device-lost.
#   - ride completion + zero probe errors gate EVERY leg (bank-only gemma
#     legs included): the first cert29 image's crashed legs print
#     "[multi] ERROR <HTTPError 500 ...>" and still count 4 class lines.
#   - a GREEN gemma leg prints its absolute longprobe classes anyway
#     (GEMMA_ABS_CLASSES, bank-only under the comparative law).
# Artifacts -> ${P647_BANK:-/root/build/p647/cert30_r271}/<leg>/.
#
# r255_cert29_gate.sh (r233 + cert29 deltas) -- cert29 (P6.47: cert28 +
# the build34 arc7mx wheel -- page-128 admission at the gemma
# full-attention geometry, r231d split-clamp lineage; .so pins libattn
# c20ca755..., fa2 046e915e..., libmqa 190e261e... unrelinked -- + the
# six-file vllm_gaudi delta of the 16effb49 content commit) section-10
# gate. vs r233 (cert28):
#   - image-content markers carry the build34 .so pin SET + the P-C2
#     window plumbing (engine/native_decode), the P-D factory hook +
#     knob (platform/envs), the P-H2 packed-row fix + the P-H4
#     posture-aware gate (lmc_mx_boot_gate + the lmcint adapter
#     marker); the adapter is READ via read_text, never imported bare
#     (the r253 probe law: the vendored lmcache_integration/__init__
#     dies on lmcache 0.5.5 outside the p41b src-override order).
#   - manifest fork_sha refuse -> cert29_fork_sha (the P-H4 content
#     commit, passed as $1).
#   - serve-script md5 refuse: c2db777506e776a86426852b6bb80de4 (the
#     P-H4-era edit; supersedes the r252-era pin).
#   - gemma MX legs ride the r245-validated NATIVE lane (ARC_QUANT=fp8
#     + ARC_KVDTYPE + ARC_MXKV_NATIVE=1 + the P-C2 window); the
#     cert28 triton gemma posture is closed by the P-C2 boot-loud
#     refusal.
#   - NEW leg gref: the P-C2 refusal face -- gemma + mxfp8 KV with NO
#     native lane MUST die boot-loud ("requires the native decode
#     lane"); a healthy boot there is a FAIL.
#   - NEW leg lmc: the P-H interop lane knob-free -- ARC_LMC=1 over
#     mxfp4 KV (native), NO VLLM_ARC_LMC_MX_GATE bypass: the P-H4
#     flip must admit it with the gate=interop-live posture line.
#     Faces = the r249 G-set: packed-metadata geometry, clean face,
#     re-hit A/A byte-equal with hit proof, the r240b death sequence
#     clean, zero transfer errors.
# Leg semantics otherwise unchanged from r233 (qwen absolute, gemma
# comparative vs same-image g0 native control; r170b churn amendment
# carried). trap restores W14.
# Artifacts -> ${P647_BANK:-/root/build/p647/cert29}/<leg>/.
set -uo pipefail
R=/root/build
P=${P647_BANK:-$R/p647/cert30_r271}
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert30}
CNAME=arc-c30
FORK_SHA=${1:?usage: r271_cert30_gate.sh <FORK_SHA> [legs...]}
shift || true
GATE_FAIL=0
mkdir -p "$P"
# the content commit the candidate image MUST carry (the r270 build, the
# PR #188 squash = the r269 fix), the base commit it inherits (cert29 --
# refused as THIS gate's sha: the base image predates the fix) and the
# crash commit the lineage must never carry (the first cert29 image, rc1)
WANT_SHA=906e8326661a447d1c0cec49575aad3249912f49
BASE_SHA=4dd3a1b9710f1c638f7f0b5ac1885525cf57c3bc
CRASH_SHA=16effb497cd36cb25df7c19cb6f1938d2d45cdc5
ENGINE_MD5=d1fa3c17cde044211cbf62e28ddcc1b1
NATIVE_MD5=81e427590e7ad3482e46bdc25f4a62b2
TRITON_MD5=e33aa7b749e34ea350441da2e339ff30
[ "$FORK_SHA" = "$WANT_SHA" ] \
  || { echo "R271_FORK_SHA_REFUSED ($FORK_SHA != the r270 content commit $WANT_SHA)"; exit 1; }
echo "R271_GATE_BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) image=$IMG fork_sha=$FORK_SHA bank=$P legs=[${*:-full battery}]"

# ---- kernel health (r264): the crash class of the first cert29 image has a
# kernel signature -- "Engine reset: engine_class=ccs" on both cards + a
# devcoredump, 13-18 s before any python surface. Per leg: every dmesg
# line at/after the leg's kernel mark (uptime seconds) matching the
# fault patterns is banked and MUST number zero. ----
kmark() { cut -d' ' -f1 /proc/uptime | cut -d. -f1; }
khealth() {  # $1 = bank dir $2 = kernel mark $3 = label
  local T0=$2 N=""
  dmesg 2>/dev/null | awk -v t0="$T0" \
    '{ s = $1; gsub(/[][]/, "", s); if ((s + 0) >= (t0 + 0)) print }' \
    | grep -E 'Engine reset|exec queue reset|devcoredump|GPU HANG|[Dd]evice lost' \
    > "$1/kernel_faults.txt" || true
  N=$(wc -l < "$1/kernel_faults.txt")
  echo "kernel_fault_lines=$N" >> "$1/markers.txt" 2>/dev/null || true
  echo "KERNEL_HEALTH[$3]: $([ "$N" = "0" ] && echo OK || echo FAIL) fault_lines=$N (dmesg since uptime ${T0}s)"
  if [ "$N" != "0" ]; then
    GATE_FAIL=$((GATE_FAIL+1))
    head -6 "$1/kernel_faults.txt" | cut -c1-160
  fi
}
GATE_K0=$(kmark)

# ---- pre-refuses: no second arc leg, serve-script byte pin, probes ----
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R255_BUSY (arc leg in flight)"; exit 2
fi
GOTSS=$(md5sum $R/serve_arc_b70.sh 2>/dev/null | cut -d" " -f1)
[ "$GOTSS" = "c2db777506e776a86426852b6bb80de4" ] \
  || { echo "R255_SERVE_SCRIPT_STALE ($GOTSS != c2db777506e776a86426852b6bb80de4)"; exit 1; }
for f in $R/p139_conc.py $R/r113_battery.py $R/p618/p618_longprobe.py \
         $R/p647/arc_garbage_probe.py; do
  [ -f "$f" ] || { echo "R255_NO_PROBE $f (ship it first)"; exit 1; }
done
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R255_NO_IMAGE $IMG"; exit 1; }

# ---- image-content markers: refuse a stale cert30 image ----
IMG_OK=$(docker run --rm -i --entrypoint /opt/venv/bin/python "$IMG" - <<'PYEOF' 2>/dev/null
import inspect, pathlib
from vllm_gaudi.xpu.patches.mx_kv import engine
from vllm_gaudi.xpu.patches.mx_kv import native_decode as n
from vllm_gaudi.xpu.patches.mx_kv import triton_mx_decode as t
from vllm_gaudi.xpu.patches import lmc_mx_boot_gate as g
from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode
tsrc = open(t.__file__).read()
esrc = inspect.getsource(engine)
nsrc = inspect.getsource(n)
gsrc = inspect.getsource(g)
# read_text, never import: the adapter path is resolution-ordered (r253)
psrc = pathlib.Path(engine.__file__).parent.parent.parent.joinpath('platform.py').read_text()
vsrc = pathlib.Path(g.__file__).parent.parent.parent.joinpath('envs.py').read_text()
atxt = pathlib.Path(n.__file__).parent.parent.joinpath('lmcint', 'vllm_v1_adapter.py').read_text()
ok = "tl.maximum(1, n_tiles)" in tsrc          # v3 s_eff 1-tile floor
ok = ok and "query.float()" not in esrc        # native-q decode path (inherited)
ok = ok and "_split_verify_decode" in esrc     # batched MTP-verify decode
ok = ok and "_mx_decode_split_rows" in tsrc    # v4 verify-row kernel
ok = ok and "_mx_decode_split_dot" in tsrc     # v5 XMX tl.dot kernel
ok = ok and "_V5_DEFAULT" in esrc              # v5 default flip
ok = ok and engine._V5_DEFAULT == ("mxfp8", "mxk8v4")
ok = ok and "ARC_MXKV_VERIFY_DECODER" in esrc  # verify-decoder override env
ok = ok and "mx_decode_rows_dot" in esrc       # v5 dispatch
ok = ok and "_f16_scale" in tsrc and "_e2m1_byte" in tsrc  # K2/K3 helpers
ok = ok and 'ARC_MXKV_K2", "1") != "0"' in tsrc           # K2 default-on
ok = ok and 'ARC_MXKV_K3", "1") != "0"' in tsrc           # K3 default-on
ok = ok and 'ARC_MXKV_V5_TFLOOR", "2") or 2' in tsrc      # r195 tf2 default
ok = ok and "_V5_G_MIN" in esrc and engine._V5_G_MIN == 8  # G-adaptive knee
ok = ok and "_mx_decode_split_dot2" in tsrc               # v7 research kernel
ok = ok and 'ARC_MXKV_V7", "0") == "1"' in tsrc           # v7 dormant
# native line (cert26+, inherited)
ok = ok and "from .native_decode import mx_decode_native" in esrc
ok = ok and 'os.environ.get("ARC_MXKV_NATIVE", "") == "1"' in esrc
ok = ok and engine._NATIVE_MINLEN == {"mxfp8": (16384, 16384),
                                      "mxfp4": (12288, 8192),
                                      "mxk8v4": (8192, 8192)}
ok = ok and "ARC_MXKV_NSPLIT" in nsrc         # split-depth instrument retained
ok = ok and "MXKV native call fmt=" in nsrc    # capture telemetry
ok = ok and callable(n.mx_decode_native) and callable(mx_paged_decode)
ok = ok and 'os.environ.get("ARC_MXKV_AB", "") == "1"' in esrc  # r224a AB
ok = ok and "ARC_MXKV_AB_MINSL" in esrc
# P-C2 window plumbing (cert29 delta)
ok = ok and "flash_attn_varlen_func as _fa_varlen" in esrc  # window shim
ok = ok and "_HAS_FA_VARLEN" in esrc                        # shim posture flag
ok = ok and "requires the native decode lane" in esrc       # boot-loud refusal
ok = ok and "sliding_window: int | None = None" in nsrc      # window param
ok = ok and "_tkey = (fmt, n_rows, _max_sk, int(bt_rows.shape[1]), _w)" in nsrc
# P-D gate + knob (cert29 delta)
ok = ok and '("vllm.distributed.kv_transfer.kv_connector.factory", "lmc_mx_boot_gate")' in psrc
ok = ok and "VLLM_ARC_LMC_MX_GATE: bool = True" in vsrc
# P-H2 packed-row fix + P-H4 posture-aware gate (cert29 delta)
ok = ok and "def _interop_live" in gsrc
ok = ok and "gate=interop-live" in gsrc
ok = ok and "gate=bypassed" in gsrc
ok = ok and "ARC_LMC_MX_REFUSED" in gsrc
ok = ok and 'ARC_LMC_MX_INTEROP = "packed-row/r249"' in atxt
ok = ok and "hidden_dim_size = _tq_hidden if _packed else num_kv_head * head_size" in atxt
ok = ok and callable(g._install)
# build34 wheel pins: page-128 admission + fa2 stub + unrelinked mqa
import hashlib, vllm_xpu_kernels
vp = pathlib.Path(vllm_xpu_kernels.__file__).parent
ok = ok and hashlib.md5((vp / "libattn_kernels_xe_2.so").read_bytes()).hexdigest() == "c20ca75550f29513a0f73b70ebb1243a"
ok = ok and hashlib.md5((vp / "_vllm_fa2_C.abi3.so").read_bytes()).hexdigest() == "046e915ebce0eb33a37832377f8c2aed"
sos = sorted(vp.rglob("libmqa_logits*.so"))
ok = ok and bool(sos) and hashlib.md5(sos[0].read_bytes()).hexdigest() == "190e261edc79b3b497d8d8e801e65757"
# r262 stride fix (the r263 rebuild delta): the bridge guard + the
# materialization + its one-line-per-class log + the engine call-site
# contract comment, and the INSTALLED bytes == the content commit's blobs
ok = ok and "if not bt_rows.is_contiguous():" in nsrc          # the guard
ok = ok and "bt_rows = bt_rows.contiguous()" in nsrc           # the materialization
ok = ok and "MXKV native bt materialized: rows=%d bt_width=%d stride0=%d " in nsrc
ok = ok and "r262 stride law (2026-11-27): this expand VIEW has stride" in esrc
emd5 = hashlib.md5(pathlib.Path(engine.__file__).read_bytes()).hexdigest()
nmd5 = hashlib.md5(pathlib.Path(n.__file__).read_bytes()).hexdigest()
ok = ok and emd5 == "d1fa3c17cde044211cbf62e28ddcc1b1"        # engine.py blob @ 4dd3a1b9 (inherited)
ok = ok and nmd5 == "81e427590e7ad3482e46bdc25f4a62b2"        # native_decode.py blob @ 4dd3a1b9 (inherited)
# r269 decode page fix (the cert30 delta): all three entry points take
# block_size None and resolve the cache's OWN page at body start -- the
# walk that read gemma's page-128 glob table at pitch 64; the installed
# bytes == the content commit's blob (what the r270 build asserted)
ok = ok and tsrc.count("block_size: int | None = None") == 3
ok = ok and tsrc.count("block_size = cache.shape[1]") == 3
tmd5 = hashlib.md5(pathlib.Path(t.__file__).read_bytes()).hexdigest()
ok = ok and tmd5 == "e33aa7b749e34ea350441da2e339ff30"        # triton_mx_decode.py blob @ 906e8326
print("IMG_ENGINE_MD5", emd5, "IMG_NATIVE_MD5", nmd5, "IMG_TRITON_MD5", tmd5)
print("IMG_CONTENT_OK" if ok else "IMG_CONTENT_STALE")
PYEOF
) || IMG_OK=""
# vllm's import-time plugin logger writes INFO lines to STDOUT (a stdout
# handler is installed at import), so the capture is multi-line -- match
# the verdict by grep, never by string equality (r161 maiden-run lesson).
echo "$IMG_OK" | grep -q "IMG_CONTENT_OK" \
  || { echo "R271_STALE_IMAGE (missing cert30 content line) IMG=$IMG"; exit 1; }
echo "$IMG_OK" | grep -E "^IMG_(ENGINE|NATIVE|TRITON)_MD5" || echo "IMG_MD5_LINES_MISSING (pre-r270 image?)"
echo "R255_IMAGE_CONTENT_OK $IMG"
# manifest provenance refuse: fork_sha must be the cert30 content commit
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -o "cert30_fork_sha=.*" /etc/vllm-arc/IMAGE_MANIFEST' 2>/dev/null) || MF_SHA=""
echo "$MF_SHA" | grep -q "cert30_fork_sha=$FORK_SHA" \
  || { echo "R271_FORK_SHA_MISMATCH ($MF_SHA vs $FORK_SHA)"; exit 1; }
echo "R271_FORK_SHA_OK $MF_SHA"
# the inherited base must be the r263 REBUILD of cert29 (never the rc1
# crash image): both lineage lines from the base manifest survive the
# r270 append (cat of the full base manifest)
docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -o "cert29_fork_sha=.*" /etc/vllm-arc/IMAGE_MANIFEST' 2>/dev/null \
  | grep -q "cert29_fork_sha=$BASE_SHA" \
  || { echo "R271_BASE_MANIFEST_STALE (no cert29_fork_sha=$BASE_SHA lineage line)"; exit 1; }
docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -o "cert29_rebuilt=.*" /etc/vllm-arc/IMAGE_MANIFEST' 2>/dev/null \
  | grep -q "cert29_rebuilt=r263" \
  || { echo "R271_NOT_THE_REBUILD (no cert29_rebuilt=r263 manifest line)"; exit 1; }
echo "R271_BASE_MANIFEST_OK (cert29 r263-rebuild lineage, never rc1)"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  # r264 (stale-baton law, r262): the clean recipe -- restart, delete the
  # lock-only JIT build dirs, GPU-free wait, serve, health + ONE REAL
  # generation (health alone is existence-only). The script refuses by
  # itself if an arc container is live; it blocks until healthy or FAIL.
  mkdir -p $R/logs
  bash $R/r262_w14_restore.sh > $R/logs/w14_restore_r271.log 2>&1
  local rc=$?
  grep -hE "^W14_(STALE_BATON_DIRS_REMOVED|SERVE_LAUNCHED|HEALTH_OK|GEN_OK|GEN_FAIL|RESTORE_FAIL|FINAL|ALREADY_HEALTHY)" \
    $R/logs/w14_restore_r271.log || true
  echo "W14_RESTORE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) log=$R/logs/w14_restore_r271.log"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$CNAME"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "GATE_FAIL_STALE_CONTAINER $CNAME"; return 1
}

gcount() {
  local c
  c=$(docker exec $CNAME grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect $CNAME >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

garble_check() {  # $1 = longprobe out file $2 = gate(1|0, default 1) --
  # r98 law: grep -F (r84-era BRE escapes never matched; do not
  # reintroduce). Gemma legs pass 0: bank-only (the gemma baseline
  # trips REP-LOOP on native KV -- r164 g0 evidence 2026-11-12; the
  # gemma verdict is comparative, see gemma_compare).
  local lines empty errs GATE=${2:-1}
  lines=$(grep -cE "^\[(essay|code|multi|simple)\]" "$1" || true)
  empty=$(( $(grep -cF "garble={'cold': [], 'rehit': [], 'rehit2': []}" "$1" || true)
          + $(grep -cF "garble={'r1': [], 'r2': []}" "$1" || true) ))
  errs=$(grep -cE "^\[(essay|code|multi|simple)\] ERROR" "$1" || true)
  echo "GARBLE: class_lines=$lines empty_garble=$empty probe_errors=$errs (gated=$GATE) in $(basename "$1")"
  # r264b: TWO gates hold on every leg, the bank-only ones included --
  # ride completion (a class line per class: a ride that raises prints
  # "[<class>] ERROR <...>" and drops the class -- the first cert29
  # image's m8g/m4g/mkg banks show "[multi] ERROR <HTTPError 500 ...>")
  # and zero probe errors (HTTP 500 / timeout / empty body of any class)
  [ "${lines:-0}" -ge 4 ] || { echo "RIDE-COMPLETION: FAIL ($1 class_lines=$lines of 4)"; GATE_FAIL=$((GATE_FAIL+1)); }
  [ "${errs:-0}" = 0 ] || { echo "PROBE-ERROR: FAIL ($1 probe_errors=$errs)"; GATE_FAIL=$((GATE_FAIL+1)); }
  if [ "$GATE" = 1 ]; then
    { [ "$lines" -ge 4 ] && [ "$lines" = "$empty" ]; } || { echo "ZERO-GARBLE: FAIL ($1)"; GATE_FAIL=$((GATE_FAIL+1)); }
  fi
}

longreturn_summary() {  # $1 = tag $2 = leg dir $3 = gate(1|0, default 1)
  # -- r99 law: probe writes its jsons to CWD; summary MUST run from the
  # leg dir. Gemma legs pass 0: bank-only (the native g0 baseline trips
  # REP-LOOP + finish=length -- r164 evidence 2026-11-12; the gemma
  # verdict is comparative, see gemma_compare).
  local GATE=${3:-1}
  ( cd "$2" && python3 - "$1" > longreturn_summary.txt 2>&1 << 'PYEOF' || true
import json, os, sys
tag = sys.argv[1]
def load(fn):
    with open(fn) as f:
        return json.load(f)
full = load(f"longprobe_{tag}_full.json")
rehits = [load(f"longprobe_{tag}r{c}_rehit.json") for c in (1, 2, 3)
          if os.path.exists(f"longprobe_{tag}r{c}_rehit.json")]
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
print("LONGRETURN:", "PASS" if not fail else "FAIL " + "; ".join(fail))
print("T1_SEED:", "PASS" if not t1_miss else "MISS " + ",".join(t1_miss))
PYEOF
  )
  if [ "$GATE" = 1 ]; then
    grep -q "LONGRETURN: PASS" "$2/longreturn_summary.txt" || GATE_FAIL=$((GATE_FAIL+1))
    grep -q "T1_SEED: PASS" "$2/longreturn_summary.txt" || GATE_FAIL=$((GATE_FAIL+1))
  fi
  cat "$2/longreturn_summary.txt"
}

gemma_compare() {  # comparative corruption verdict for the gemma legs vs
  # the same-image native g0 leg (reploop-classifier-baseline law; r164
  # disposition 2026-11-12: gemma4-31b think-spam degeneration -- universal
  # cross-class attractor + P1 newline-spam -- is BASELINE on native fp8
  # KV, so the MX gemma legs gate on NO-NEW-MODES + no window
  # amplification, never vs CLEAN). r170b amendment carried: churn into
  # g0's own class set is KNOWN-CLASS churn, banked for review.
  python3 - "$P" > "$P/gemma_compare.txt" 2>&1 << 'PYEOF' || true
import json, sys
from pathlib import Path
bank = Path(sys.argv[1])
CLASSES = ("essay", "code", "multi", "simple")
def load_full(leg):
    d = bank / leg
    cands = sorted(d.glob("longprobe_*_full.json")) if d.is_dir() else []
    if not cands:
        return None
    with open(cands[0]) as f:
        return json.load(f)
def flags_of(doc):
    out = {}
    for cls in CLASSES:
        v = doc.get(cls, {}) if doc else {}
        fl = set()
        for g in list(v.get("garble", {}).values()) + [v.get("t1_garble", [])]:
            fl |= set(g or [])
        out[cls] = fl
    return out
def topwin_n(doc):
    out = {}
    for cls in CLASSES:
        v = doc.get(cls, {}) if doc else {}
        n = 0
        ws = v.get("win_stats") or {}
        for ph in (ws.values() if isinstance(ws, dict) else []):
            for w in (ph.get("top_windows") or [])[:1]:
                n = max(n, int(w.get("n", 0)))
        out[cls] = n
    return out
g0 = load_full("g0")
if g0 is None:
    print("GEMMA_COMPARE: FAIL (no g0 baseline longprobe banked)")
    sys.exit(0)
g0f, g0n = flags_of(g0), topwin_n(g0)
print(f"[g0] baseline flags={ {c: sorted(g0f[c]) for c in CLASSES} } "
      f"topwin={g0n} (native LONGRETURN fails these same classes)")
fails = []
churn_notes = []
for leg in ("m8g", "m4g", "mkg"):
    doc = load_full(leg)
    if doc is None:
        print(f"[{leg}] no longprobe banked (skipped)")
        continue
    lf, ln = flags_of(doc), topwin_n(doc)
    for cls in CLASSES:
        new = lf[cls] - g0f[cls]
        if new:
            fails.append(f"{leg}/{cls}: NEW flag modes {sorted(new)} (g0 {sorted(g0f[cls])})")
        if ln[cls] > max(2 * max(g0n[cls], 1), 8):
            fails.append(f"{leg}/{cls}: window amplification leg={ln[cls]} vs g0={g0n[cls]}")
        hashes = sorted(set(doc.get(cls, {}).get("hashes", [])))
        if len(hashes) > 1:
            churn_notes.append(f"{leg}/{cls}: KNOWN-CLASS churn {hashes}")
    print(f"[{leg}] flags={ {c: sorted(lf[c]) for c in CLASSES} } "
          f"topwin={ln} (g0 topwin={g0n})")
for note in churn_notes:
    print(f"[churn] {note}")
print("GEMMA_COMPARE:", "PASS (no new modes / no amplification; "
      "known-class churn banked for review)"
      if not fails else "FAIL " + "; ".join(fails))
PYEOF
  cat "$P/gemma_compare.txt"
  grep -q "GEMMA_COMPARE: PASS" "$P/gemma_compare.txt" \
    || { echo "GEMMA_COMPARE: FAIL (MX lane worse than native baseline)"; GATE_FAIL=$((GATE_FAIL+1)); }
}

churn() {  # $1 = seed base $2 = served model -- r105 churn with the model
  # parameterized (gemma legs need their own name). Out banks per-leg
  # (${CD}/churn_$base.out, CD dynamic-scoped from run_leg) -- r159 wrote
  # to the bank root and each leg clobbered the previous leg's evidence.
  local base=$1 mdl=$2
  python3 > "${CD:-$P}/churn_$base.out" 2>&1 << PYEOF || true
import json, threading, time, urllib.request
def one(i, out):
    body = {"model": "$mdl", "stream": False, "temperature": 0.0,
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
  tail -1 "${CD:-$P}/churn_$base.out"
}

text_battery() {  # $1 = leg dir $2 = served model $3 = endpoint
  # (completions|chat) -- the r149 3-prompt temp-0 battery (trace25-28
  # validated set); p1 MUST contain Paris. Gemma is chat-only: the raw
  # /v1/completions endpoint has document-continuation semantics there
  # (m8g run: prompt-echo + further questions, dtype-INDEPENDENT -- an
  # endpoint artifact, not KV corruption), so gemma legs use the chat
  # endpoint with the message content identical. $4 = mode (abs|gemma,
  # default abs): gemma's baseline degenerates into think-spam even on
  # NATIVE fp8 KV (r164 g0 disposition 2026-11-12 -- the native control
  # fails P1_PARIS), so gemma mode gates response SHAPE only (3x
  # PROMPT_OK with non-empty text, no traceback); content quality is
  # the comparative gemma_compare verdict.
  # r271 (r268c law): every request rides logprobs (chat: logprobs=true
  # + top_logprobs=1; completions: logprobs=1 -- the p266 matrix probe's
  # validated forms) and the battery gates LP_NAN == 0 + LP_N >= 1 per
  # prompt. The gemma corruption class read WRONG KV addresses: the
  # engine surfaced nan logprobs -> the API 400s the request (the
  # "400-nan" class, http_err in every r268b RED cell). Without logprobs
  # a silently-wrong-read generation can still be shape-clean, so a
  # shape-only battery is vacuous against exactly this class.
  local CD=$1 mdl=$2 EP=${3:-completions} MODE=${4:-abs}
  : > "$CD/probe.out"
  for q in "What is the capital of France? Answer with just the city name." \
           "Write a Python function that returns the n-th Fibonacci number using iteration." \
           "Name three chemical elements that are gases at room temperature."; do
    if [ "$EP" = chat ]; then
      curl -s -m 180 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
        -d "{\"model\":\"$mdl\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":64,\"temperature\":0,\"logprobs\":true,\"top_logprobs\":1}" \
        | python3 -c 'import json,math,sys
r=json.load(sys.stdin); c=r["choices"][0]; m=c.get("message",{})
t=m.get("content") or m.get("reasoning_content") or m.get("reasoning") or ""
lp=(c.get("logprobs") or {}); vals=[d.get("logprob") for d in (lp.get("content") or [])]
n=sum(1 for v in vals if isinstance(v,float) and math.isnan(v))
print("PROMPT_OK", len(t), repr(t[:160]), "LP_N", len(vals), "LP_NAN", n)' >> "$CD/probe.out" 2>&1
    else
      curl -s -m 180 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
        -d "{\"model\":\"$mdl\",\"prompt\":\"$q\",\"max_tokens\":64,\"temperature\":0,\"logprobs\":1}" \
        | python3 -c 'import json,math,sys
r=json.load(sys.stdin); c=r["choices"][0]; t=c.get("text") or ""
lp=(c.get("logprobs") or {}); vals=[x for x in (lp.get("token_logprobs") or []) if x is not None]
n=sum(1 for v in vals if isinstance(v,float) and math.isnan(v))
print("PROMPT_OK", len(t), repr(t[:160]), "LP_N", len(vals), "LP_NAN", n)' >> "$CD/probe.out" 2>&1
    fi
  done
  cat "$CD/probe.out"
  grep -q "PROMPT_OK" "$CD/probe.out" || { echo "TEXT_BATTERY: FAIL (no PROMPT_OK) $CD"; GATE_FAIL=$((GATE_FAIL+1)); return; }
  [ "$(grep -c PROMPT_OK "$CD/probe.out" || true)" = "3" ] || { echo "TEXT_BATTERY: FAIL (missing answers) $CD"; GATE_FAIL=$((GATE_FAIL+1)); }
  # logprob gates: nan anywhere = the corruption class; LP_N 0 = the
  # engine refused/omitted the logprobs fields (vacuous check)
  local LPNAN LPN0
  LPNAN=$(grep -cE "LP_NAN [1-9]" "$CD/probe.out" || true)
  LPN0=$(grep -cE "LP_N 0 " "$CD/probe.out" || true)
  [ "${LPNAN:-0}" = "0" ] || { echo "LP_NAN: FAIL ($LPNAN prompts with nan logprobs -- the r268c corruption class) $CD"; GATE_FAIL=$((GATE_FAIL+1)); }
  [ "${LPN0:-0}" = "0" ] || { echo "LP_NAN: FAIL ($LPN0 prompts returned no logprobs -- vacuous logprob gate) $CD"; GATE_FAIL=$((GATE_FAIL+1)); }
  if [ "$MODE" = abs ]; then
    grep -q "Paris" "$CD/probe.out" || { echo "P1_PARIS: FAIL $CD"; GATE_FAIL=$((GATE_FAIL+1)); }
  else
    # gemma shape gate: non-empty text on every answer (baseline
    # think-spam still emits reasoning; EMPTY = a lane defect)
    local nonempty
    nonempty=$(grep -cE "^PROMPT_OK [1-9]" "$CD/probe.out" || true)
    [ "$nonempty" = "3" ] || { echo "TEXT_BATTERY: FAIL (empty answers: ${nonempty:-0}/3) $CD"; GATE_FAIL=$((GATE_FAIL+1)); }
  fi
  # shape errors surface as python tracebacks inside probe.out (2>&1)
  grep -q "Traceback" "$CD/probe.out" && { echo "TEXT_BATTERY: FAIL (traceback) $CD"; GATE_FAIL=$((GATE_FAIL+1)); }
  sha256sum "$CD/probe.out" > "$CD/probe.sha256"
}

erraudit() {  # $1 = leg dir
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$1/erraudit.txt" 2>/dev/null || echo "ERROR=?" > "$1/erraudit.txt"
  cat "$1/erraudit.txt"
  grep -q "ERROR=0" "$1/erraudit.txt" || { echo "ERRAUDIT: FAIL ($1)"; GATE_FAIL=$((GATE_FAIL+1)); }
}

run_leg() {
  # $1=legname $2=served $3=fullsuite(1|0) $4..=env assignments
  local LEGN=$1 SERVED=$2 FULL=$3; shift 3
  local CD=$P/$LEGN
  local K0; K0=$(kmark)   # r264: the kernel mark every fault line is dated against
  # gemma is chat-only at the probe layer (completions = document-
  # continuation semantics, dtype-independent -- m8g run1 evidence) and
  # gates its quality program comparatively (r164 g0 disposition)
  case $SERVED in gemma*) local EP=chat GMODE=gemma GQ=0 ;; *) local EP=completions GMODE=abs GQ=1 ;; esac
  mkdir -p "$CD"
  { echo "leg=$LEGN image=$IMG served=$SERVED fullsuite=$FULL"; echo "env=[$*]"; } > "$CD/leg_env.txt"
  echo "=== [$LEGN] boot $(date +%H:%M:%S) env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$LEGN"; then
    echo "GATE_FAIL_BOOT_$LEGN"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -40 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    GATE_FAIL=$((GATE_FAIL+1)); khealth "$CD" "$K0" "$LEGN"; return 1
  fi
  echo "HEALTH_OK_$LEGN $(date +%H:%M:%S)"
  sleep 5
  # MX lane markers + posture evidence (the telemetry decomposition:
  # every lane attribute is banked per leg -- totals never stand alone)
  {
    echo "mx_backend_lines=$(gcount 'Using MX KV attention backend')"
    echo "native_lines=$(gcount 'MXKV native call')"
    echo "materialized_lines=$(gcount 'MXKV native bt materialized')"
    echo "device_lost_lines=$(gcount 'UR_RESULT_ERROR_DEVICE_LOST')"
    echo "pc2_windowed_impl=$(gcount 'MXKV P-C2 windowed impl')"
    echo "lmcache_lines=$(gcount 'LMCacheConnector')"
    echo "interop_live=$(gcount 'gate=interop-live')"
    echo "tracebacks=$(gcount 'Traceback (most recent call last)')"
    echo "skin_template_lines=$(gcount 'chat_template_gemma4.jinja')"
    docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" /root/serve_full.log 2>/dev/null | head -4
  } > "$CD/markers.txt"
  cat "$CD/markers.txt"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  # text battery (all legs -- the corruption gate; gemma shape-mode)
  text_battery "$CD" "$SERVED" "$EP" "$GMODE"
  # decode battery (all legs)
  python3 $R/r113_battery.py decode "$SERVED" > "$CD/battery.txt" 2>&1 || echo "BATTERY_NONZERO_EXIT $LEGN"
  tail -6 "$CD/battery.txt"
  if [ "$FULL" = 1 ]; then
    # suite-13: malformed-input hardening (host python3, urllib only)
    ARC_PROBE_MODEL="$SERVED" python3 $R/p647/arc_garbage_probe.py > "$CD/garbage.out" 2>&1
    tail -5 "$CD/garbage.out"
    grep -q "GARBAGE_SUITE_PASS" "$CD/garbage.out" || { echo "SUITE13: FAIL $LEGN"; GATE_FAIL=$((GATE_FAIL+1)); }
    # suite-14: long-return full + 3 churn rehits + hash determinism
    ( cd "$CD" && ARC_PROBE_MODEL="$SERVED" python3 $R/p647/p647_longprobe.py c30$LEGN full > longprobe_full.out 2>&1 )
    grep -E "^\[|LONGPROBE" "$CD/longprobe_full.out"
    garble_check "$CD/longprobe_full.out" "$GQ"
    for cyc in 1 2 3; do
      churn $(( 900 * cyc )) "$SERVED"
      ( cd "$CD" && ARC_PROBE_MODEL="$SERVED" python3 $R/p647/p647_longprobe.py c30${LEGN}r$cyc rehit > longprobe_rehit$cyc.out 2>&1 )
      grep -E "^\[|LONGPROBE" "$CD/longprobe_rehit$cyc.out"
      garble_check "$CD/longprobe_rehit$cyc.out" "$GQ"
    done
    longreturn_summary c30$LEGN "$CD" "$GQ"
  fi
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics.txt" || true
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || echo "serve.log MISSING $LEGN"
  # r264b: the suite-time markers, RE-COUNTED after the suite. The
  # boot-time snapshot in markers.txt predates every ride: the r262 line
  # fires at the FIRST multi-row synth continuation and DEVICE_LOST only
  # at a crash, so the snapshot reads 0 for both whatever happened (a
  # vacuous fail for the first, a vacuous pass for the second).
  {
    echo "materialized_lines_post=$(gcount 'MXKV native bt materialized')"
    echo "device_lost_lines_post=$(gcount 'UR_RESULT_ERROR_DEVICE_LOST')"
    echo "tracebacks_post=$(gcount 'Traceback (most recent call last)')"
    echo "native_lines_post=$(gcount 'MXKV native call')"
    echo "skin_template_lines_post=$(gcount 'chat_template_gemma4.jinja')"
    echo "serve_log_bytes=$(wc -c < "$CD/serve.log" 2>/dev/null || echo 0)"
  } >> "$CD/markers.txt"
  tail -5 "$CD/markers.txt"
  grep -h "MXKV native call" "$CD/serve.log" 2>/dev/null | tail -30 > "$CD/native_telemetry.txt" || true
  docker exec $CNAME grep -c "xcheck3" /root/serve_full.log > "$CD/xcheck3_count.txt" 2>/dev/null || true
  erraudit "$CD"
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f $CNAME >/dev/null 2>&1
  verify_removed || { GATE_FAIL=$((GATE_FAIL+1)); return 1; }
  # r264: 22 s of quiet before the kernel verdict -- a crash also logs a
  # bcs engine reset ~20 s after its ccs reset (r257-r262 banks)
  sleep 22
  khealth "$CD" "$K0" "$LEGN"
  # the python surface of the same class (after the kernel reset): zero
  # in the container's log AND in the banked copy (host-side grep -- a
  # docker-exec count reads 0 whenever the exec itself fails)
  DL_HOST=$(grep -c "UR_RESULT_ERROR_DEVICE_LOST" "$CD/serve.log" 2>/dev/null || true)
  grep -q "device_lost_lines_post=0" "$CD/markers.txt" 2>/dev/null \
    && [ "${DL_HOST:-x}" = "0" ] \
    || { echo "DEVICE_LOST: FAIL ($CD) -- post-suite device_lost host_count=${DL_HOST:-none}"; GATE_FAIL=$((GATE_FAIL+1)); }
  echo "=== [$LEGN] done $(date +%H:%M:%S) ==="
}

# derived longprobe with served-model env override (p618 hardcodes the
# qwen name; gemma legs need their own -- one-line edit, provenance kept)
# + the reasoning-field capture: gemma-4 is a thinking model that IGNORES
# enable_thinking:false (kwarg accepted, template unaffected) and vLLM
# 0.30 emits the stream as message.reasoning (NOT reasoning_content --
# run2 m8g live evidence: content null, reasoning carries "The capital
# of France is Paris."). Without this read every gemma longprobe hash
# is EMPTY -> spurious EMPTY/longreturn fails.
mkdir -p $R/p647
python3 - << 'PYEOF'
import re
src = open("/root/build/p618/p618_longprobe.py", encoding="utf-8").read()
if "ARC_PROBE_MODEL" not in src:
    assert src.count('MODEL = "qwen3.8-27b-fp8"') == 1, "p618 model line drifted"
    src = src.replace('MODEL = "qwen3.8-27b-fp8"',
                      'MODEL = os.environ.get("ARC_PROBE_MODEL", "qwen3.8-27b-fp8")')
    src = src.replace("import hashlib\n", "import hashlib\nimport os\n", 1)
if "msg.get(\"reasoning\")" not in src:
    assert src.count('rsn = msg.get("reasoning_content") or ""') == 1, \
        "p618 reasoning line drifted"
    src = src.replace('rsn = msg.get("reasoning_content") or ""',
                      'rsn = msg.get("reasoning_content") or msg.get("reasoning") or ""')
open("/root/build/p647/p647_longprobe.py", "w", encoding="utf-8", newline="\n").write(src)
print("p647_longprobe.py derived OK")
PYEOF

# r264 (stale-baton law, r262): never stop W14 inside a restore boot's
# first minutes -- a boot killed inside its ~27 s JIT build window
# leaves the lock every later boot waits on. Wait for health first (a
# boot takes ~5 min; allow ~17), and if it never comes up, run the
# clean restore BEFORE taking the rig.
if ! curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "R271_W14_NOT_HEALTHY at entry -- waiting for its boot to settle"
  i=0
  while [ $i -lt 100 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && break
    i=$((i + 1)); sleep 10
  done
  if ! curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "R271_W14_STILL_DOWN -- running the clean restore before the battery"
    bash $R/r262_w14_restore.sh > $R/logs/w14_restore_r271_entry.log 2>&1 || true
  fi
fi
docker stop radiance-test >/dev/null 2>&1
sleep 5

# Postures (cert30):
# - qwen MX legs: ARC_KVDTYPE + ARC_MXKV_NATIVE=1 as certified; ARC_LMC
#   UNSET -> the serve script's Phase-6 auto-off (perf posture r176) --
#   the NOTE + zero LMCacheConnector lines are gated per leg.
# - qwen lmc leg: ARC_LMC=1 over mxfp4 -- LAWFUL since P-H4 (the gate
#   admits it with gate=interop-live); no bypass knob anywhere.
# - gemma legs: the r245-validated native lane (ARC_QUANT=fp8 weights +
#   ARC_KVDTYPE + ARC_MXKV_NATIVE=1 + the P-C2 window riding the native
#   masked kernel); ARC_LMC=0 explicit. PLUS the LAWFUL g3 SKIN (owner
#   directive 2026-11-28): ARC_TEMPLATE=<the g3 template> + EMPTY
#   parsers -- gemma is NEVER tested under the qwen skin (no qwen
#   template/parser may touch a gemma boot; r264's gemma legs ran
#   templateless = invalid-by-posture verdicts, superseded here). The
#   skin engages per leg via skin_check (the serve-log SKIN assertion).
# - gref: gemma + MX KV with NO native -> the P-C2 boot-loud refusal
#   (under the same lawful g3 skin).
QWEN_FP8="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1"
GEMMA_NAT="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER="

lmc_autooff_check() {  # $1 = leg dir -- the qwen MX lanes MUST come up with
  # LMCache off: the serve script's auto-off NOTE in boot.txt (a stale
  # rig serve script silently re-enables the double-quant driver) and
  # zero LMCacheConnector lines in the serve log.
  local CD=$1
  grep -q "NOTE: ARC_LMC auto-off" "$CD/boot.txt" \
    || { echo "LMC_POSTURE: FAIL ($CD) -- serve script stale or posture not applied"; GATE_FAIL=$((GATE_FAIL+1)); }
  grep -q "lmcache_lines=0" "$CD/markers.txt" \
    || { echo "LMC_POSTURE: FAIL ($CD) -- LMCache engaged on an MX lane"; GATE_FAIL=$((GATE_FAIL+1)); }
}

native_posture_check() {  # $1 = leg dir -- the qwen native legs MUST show
  # the capture telemetry (>=1 'MXKV native call' per worker) with
  # splits=auto (NSPLIT retired; the C++ chain owns the pick). serve.log
  # is the docker-cp'd copy banked before container removal.
  local CD=$1
  grep -Eq "native_lines=[1-9]" "$CD/markers.txt" \
    || { echo "NATIVE_POSTURE: FAIL ($CD) -- no native capture telemetry"; GATE_FAIL=$((GATE_FAIL+1)); }
  grep -q "splits=auto" "$CD/serve.log" 2>/dev/null \
    || { echo "NATIVE_POSTURE: FAIL ($CD) -- no splits=auto evidence"; GATE_FAIL=$((GATE_FAIL+1)); }
}

pc2_window_check() {  # $1 = leg dir -- the gemma native legs MUST show the
  # windowed-impl telemetry (r245 grid: pc2_windowed_impl=100 per cell).
  local CD=$1
  grep -Eq "pc2_windowed_impl=[1-9]" "$CD/markers.txt" \
    || { echo "PC2_WINDOW: FAIL ($CD) -- no 'MXKV P-C2 windowed impl' telemetry"; GATE_FAIL=$((GATE_FAIL+1)); }
}

leg_ctrl() {
  run_leg ctrl qwen3.8-27b-fp8 0 $QWEN_FP8 || true
  MX_LINES=$(grep -c "mx_backend_lines=0" "$P/ctrl/markers.txt" || true)
  [ "$MX_LINES" = "1" ] || { echo "DEFAULT-OFF: FAIL (MX backend engaged without ARC_KVDTYPE)"; GATE_FAIL=$((GATE_FAIL+1)); }
  # ctrl keeps the LMC-on certified posture (e4m3-V; NOT auto-off)
  grep -Eq "lmcache_lines=[1-9]" "$P/ctrl/markers.txt" \
    || { echo "LMC_POSTURE: FAIL (ctrl lost the certified LMC-on posture)"; GATE_FAIL=$((GATE_FAIL+1)); }
}
leg_m8q() {
  run_leg m8q qwen3.8-27b-fp8 1 $QWEN_FP8 ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m8q/markers.txt" || { echo "MX_MARKER: FAIL m8q (expect >=1 worker)"; GATE_FAIL=$((GATE_FAIL+1)); }
  lmc_autooff_check "$P/m8q"
  native_posture_check "$P/m8q"
}
leg_m4q() {
  run_leg m4q qwen3.8-27b-fp8 1 $QWEN_FP8 ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m4q/markers.txt" || { echo "MX_MARKER: FAIL m4q (expect >=1 worker)"; GATE_FAIL=$((GATE_FAIL+1)); }
  lmc_autooff_check "$P/m4q"
  native_posture_check "$P/m4q"
}
leg_mkq() {
  run_leg mkq qwen3.8-27b-fp8 1 $QWEN_FP8 ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/mkq/markers.txt" || { echo "MX_MARKER: FAIL mkq (expect >=1 worker)"; GATE_FAIL=$((GATE_FAIL+1)); }
  lmc_autooff_check "$P/mkq"
  native_posture_check "$P/mkq"
}

leg_lmc() {  # the P-H interop lane, knob-free (the r249 G-set on the
  # certified image: the P-H4 flip admits ARC_LMC=1 over mxfp4 with the
  # gate=interop-live posture line; VLLM_ARC_LMC_MX_GATE is NOT set).
  local CD=$P/lmc FAILED=""
  local K0; K0=$(kmark)
  mkdir -p "$CD"
  echo "=== [lmc] boot $(date +%H:%M:%S) (ARC_LMC=1 + mxfp4 + native, gate default-on) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME $QWEN_FP8 \
    ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1 ARC_LMC=1 \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait lmc; then
    echo "GATE_FAIL_BOOT_lmc"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -40 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    GATE_FAIL=$((GATE_FAIL+1)); khealth "$CD" "$K0" lmc; return 1
  fi
  echo "HEALTH_OK_lmc $(date +%H:%M:%S)"
  sleep 5
  {
    echo "lmcache_lines=$(gcount 'LMCacheConnector')"
    echo "interop_live=$(gcount 'gate=interop-live')"
    echo "mx_backend_lines=$(gcount 'Using MX KV attention backend')"
    echo "native_lines=$(gcount 'MXKV native call')"
    echo "tracebacks=$(gcount 'Traceback (most recent call last)')"
    docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" /root/serve_full.log 2>/dev/null | head -4
  } > "$CD/markers.txt"
  cat "$CD/markers.txt"
  grep -Eq "interop_live=[1-9]" "$CD/markers.txt" \
    || { echo "LMC_INTEROP: FAIL (no gate=interop-live posture line -- the P-H4 flip did not admit)"; GATE_FAIL=$((GATE_FAIL+1)); FAILED="$FAILED interop_line"; }
  grep -Eq "lmcache_lines=[1-9]" "$CD/markers.txt" \
    || { echo "LMC_INTEROP: FAIL (LMCacheConnector not engaged)"; GATE_FAIL=$((GATE_FAIL+1)); FAILED="$FAILED lmcache"; }
  grep -q "NOTE: ARC_LMC=1 forced" "$CD/boot.txt" \
    || { echo "LMC_INTEROP: FAIL (forced-on NOTE missing -- serve script stale)"; GATE_FAIL=$((GATE_FAIL+1)); FAILED="$FAILED note"; }
  # ---- G1: the packed-metadata oracle readout (adapter init log) ----
  docker exec $CNAME grep -im1 "kv shape" /root/serve_full.log > "$CD/kvshape_line.txt" 2>/dev/null || true
  if grep -q "(17, 2, 1600, 1, 272)" "$CD/kvshape_line.txt" 2>/dev/null; then
    echo "G1_PACKED_METADATA_OK"
  elif grep -q "(17, 2, 1600, 2, 256)" "$CD/kvshape_line.txt" 2>/dev/null; then
    echo "G1_FAIL_OLD_STANDARD_GEOMETRY (the packed-row fix not live)"; GATE_FAIL=$((GATE_FAIL+1)); FAILED="$FAILED g1"
  else
    echo "G1_FAIL_UNEXPECTED (see kvshape_line.txt)"; GATE_FAIL=$((GATE_FAIL+1)); FAILED="$FAILED g1"
  fi
  cat "$CD/kvshape_line.txt" 2>/dev/null
  # warmup + decode battery (uniform with the other qwen legs)
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3.8-27b-fp8\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  text_battery "$CD" qwen3.8-27b-fp8 completions abs
  python3 $R/r113_battery.py decode qwen3.8-27b-fp8 > "$CD/battery.txt" 2>&1 || echo "BATTERY_NONZERO_EXIT lmc"
  tail -6 "$CD/battery.txt"
  # ---- G2: the clean face (p139_conc through the interop lane) ----
  docker cp $R/p139_conc.py $CNAME:/root/p139_conc.py >/dev/null 2>&1
  docker exec $CNAME rm -f /root/p139_conc_c30lmc.json
  docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py 8192 64 1 c30lmc qwen3.8-27b-fp8 0.0
  i=0
  while [ $i -lt 90 ]; do
    docker exec $CNAME test -f /root/p139_conc_c30lmc.json 2>/dev/null && break
    if ! docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then echo "LMC_G2_ENGINE_DEAD"; break; fi
    sleep 5; i=$((i+1))
  done
  docker cp $CNAME:/root/p139_conc_c30lmc.json "$CD/" >/dev/null 2>&1 || true
  NA=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_ok'])" "$CD/p139_conc_c30lmc.json" 2>/dev/null || echo 0)
  if [ "${NA:-0}" -ge 1 ]; then
    echo "G2_CLEAN_FACE_OK n_ok=$NA store_lines=$(gcount 'Stored')"
  else
    echo "G2_FAIL n_ok=${NA:-?}"; GATE_FAIL=$((GATE_FAIL+1)); FAILED="$FAILED g2"
  fi
  # ---- G3: the fixed-prompt re-hit A/A (byte-true restore oracle) ----
  PROMPT_JSON="$CD/aaprompt.json"
  python3 - "$PROMPT_JSON" << 'PYEOF'
import json, sys
# a long deterministic prompt (~30K chars / several 1600-token chunks):
# the store path banks full chunks, the re-hit retrieves them above
# min_retrieve_tokens=1024; same bytes both passes, greedy temp 0.
# (identical bytes to the r249 aaprompt -- the G1 chunk-count geometry
# 17 depends on them.)
words = ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta"]
prompt = " ".join(f"({i}) the {words[i % len(words)]} sequence continues {i} * {i+1};" for i in range(700))
json.dump({"model": "qwen3.8-27b-fp8", "prompt": prompt,
           "max_tokens": 64, "temperature": 0, "ignore_eos": True}, open(sys.argv[1], "w"))
PYEOF
  docker cp "$PROMPT_JSON" $CNAME:/root/aaprompt.json >/dev/null 2>&1
  T1=$(curl -s -m 600 -o "$CD/aa1.json" -w '%{time_total}' http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' -d @"$PROMPT_JSON")
  # RET_BASE: retrieval-line count AFTER pass 1 (stores only) -- the
  # delta vs post-pass-2 is the hit
  RET_BASE=$(gcount "etriev")
  sleep 10
  T2=$(curl -s -m 600 -o "$CD/aa2.json" -w '%{time_total}' http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' -d @"$PROMPT_JSON")
  sleep 2
  RET2=$(gcount "etriev")
  X1=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['choices'][0]['text'])" "$CD/aa1.json" 2>/dev/null || echo AA1_EMPTY)
  X2=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['choices'][0]['text'])" "$CD/aa2.json" 2>/dev/null || echo AA2_EMPTY)
  echo "$X1" > "$CD/aa1.text"; echo "$X2" > "$CD/aa2.text"
  echo "t1=$T1 t2=$T2 retrieve_before=$RET_BASE retrieve_after=$RET2" | tee "$CD/aa_times.txt"
  DELTA=$(( ${RET2:-0} - ${RET_BASE:-0} ))
  SPEEDUP=$(python3 -c "print(1 if float('${T2:-999}') < 0.5 * float('${T1:-1}') else 0)" 2>/dev/null || echo 0)
  if [ "$X1" = "$X2" ] && [ "$X1" != "AA1_EMPTY" ]; then
    if [ "$DELTA" -gt 0 ]; then
      echo "G3_REHIT_AA_OK retrieve_delta=$DELTA texts_byte_equal=yes t1=$T1 t2=$T2"
    elif [ "$SPEEDUP" = "1" ]; then
      echo "G3_REHIT_AA_OK_QUIETLOG (texts equal + wall ${T2}s < 50% of ${T1}s = functional hit)"
    else
      echo "G3_FAIL_NO_HIT_PROOF retrieve_delta=$DELTA t1=$T1 t2=$T2"; GATE_FAIL=$((GATE_FAIL+1)); FAILED="$FAILED g3"
    fi
  elif [ "$DELTA" -gt 0 ]; then
    echo "G3_FAIL_TEXT_DRIFT (retrieved but generations differ -- corrupt restore)"
    GATE_FAIL=$((GATE_FAIL+1)); FAILED="$FAILED g3_drift"
  else
    echo "G3_FAIL texts_differ_no_retrieve (t1=$T1 t2=$T2 delta=$DELTA)"
    GATE_FAIL=$((GATE_FAIL+1)); FAILED="$FAILED g3"
  fi
  # ---- G4: the r240b death sequence (c32768: n1 -> rehit -> n2 -> n4) ----
  G4="G4_OK"
  RUN=0
  for spec in "1 c30lmc_n1" "1 c30lmc_n1" "2 c30lmc_n2" "4 c30lmc_n4"; do
    set -- $spec; NN=$1; TAG=$2; RUN=$((RUN+1))
    docker exec $CNAME rm -f /root/p139_conc_$TAG.json
    docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py 32768 600 $NN $TAG qwen3.8-27b-fp8 0.0
    i=0; DONE=0
    while [ $i -lt 170 ]; do
      if docker exec $CNAME test -f /root/p139_conc_$TAG.json 2>/dev/null; then DONE=1; break; fi
      if ! docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then
        echo "LMC_G4_ENGINE_DEAD tag=$TAG poll=$i"; DONE=0; break
      fi
      sleep 5; i=$((i+1))
    done
    # same tag twice = the re-hit cell; bank to run-indexed names
    docker cp $CNAME:/root/p139_conc_$TAG.json "$CD/p139_conc_${TAG}_run${RUN}.json" >/dev/null 2>&1 || true
    if [ "$DONE" = "1" ]; then
      NOK=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_ok'])" "$CD/p139_conc_${TAG}_run${RUN}.json" 2>/dev/null || echo 0)
      if [ "${NOK:-0}" -ge "$NN" ]; then
        echo "G4 cell $TAG run$RUN n_ok=$NOK OK"
      else
        echo "G4 cell $TAG run$RUN PARTIAL n_ok=${NOK:-?}/$NN"
        G4="G4_FAIL_partial"
      fi
    else
      echo "G4 cell $TAG run$RUN HUNG (no JSON in $((170*5))s)"
      G4="G4_FAIL_hung_${TAG}_run${RUN}"
      break
    fi
  done
  TO=$(gcount "TimeoutError")
  DL=$(gcount "UR_RESULT_ERROR_DEVICE_LOST")
  if [ "$G4" = "G4_OK" ] && [ "${TO:-0}" = "0" ] && [ "${DL:-0}" = "0" ]; then
    echo "G4_DEATH_SEQUENCE_CLEAN (no wedge, no device-lost, no watchdog)"
  else
    echo "G4_FAIL ($G4 timeouts=$TO device_lost=$DL)"; GATE_FAIL=$((GATE_FAIL+1)); FAILED="$FAILED g4"
  fi
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || echo "serve.log MISSING lmc"
  # ---- G5: clean transfers (run-1 law: gates 1-4 can pass while the
  # lagged-store path error-storms -- the catch-all swallows them) ----
  TB=$(gcount "Traceback (most recent call last)")
  LE=$(gcount "LMCache ERROR")
  if [ "${TB:-0}" = "0" ] && [ "${LE:-0}" = "0" ]; then
    echo "G5_CLEAN_TRANSFERS_OK (zero tracebacks, zero LMCache ERROR lines)"
  else
    echo "G5_FAIL tracebacks=$TB lmc_error_lines=$LE (transfer-path exceptions -- see serve.log)"
    GATE_FAIL=$((GATE_FAIL+1)); FAILED="$FAILED g5"
  fi
  {
    echo "store_lines=$(gcount 'Stored')"
    echo "retrieve_lines=$(gcount 'etriev')"
    echo "lmc_error_lines=$(gcount 'LMCache ERROR')"
  } > "$CD/posture.txt"
  cat "$CD/posture.txt"
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics.txt" || true
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f $CNAME >/dev/null 2>&1
  verify_removed || { GATE_FAIL=$((GATE_FAIL+1)); return 1; }
  sleep 22
  khealth "$CD" "$K0" lmc
  echo "=== [lmc] done $(date +%H:%M:%S) failed=[$FAILED] ==="
}

leg_gref() {  # the P-C2 boot-loud refusal face: gemma + mxfp8 MX KV with
  # NO native lane MUST die at init ("requires the native decode lane").
  # A healthy boot here is a FAIL (the refusal is the certified guard).
  # r255a harness fix: the first pass (23:12:53Z) read "EngineCore not
  # spawned YET" as death -- true of every healthy boot's first ~60s --
  # broke at i=0, docker-cp'd before the in-container redirect even
  # existed ("serve.log MISSING"), then pkill'd the 4-second-old boot
  # before the P-C2 refusal could ever fire at model init (~4 min in
  # for gemma fp8). The loop now sleeps FIRST, copies the serve log
  # EVERY poll (evidence survives the container), breaks on the refusal
  # line itself, and treats process-death as death only after EngineCore
  # was seen alive at least once.
  local CD=$P/gref
  local K0; K0=$(kmark)
  mkdir -p "$CD"
  echo "=== [gref] expect-boot-death $(date +%H:%M:%S) (gemma + mxfp8 KV, NO native) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME $GEMMA_NAT ARC_KVDTYPE=mxfp8 \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  local i=0 HEALTH=0 rl=0 SEEN=0
  while [ $i -lt 90 ]; do
    sleep 10
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { HEALTH=1; break; }
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    rl=$(grep -c "requires the native decode lane" "$CD/serve.log" 2>/dev/null || true)
    rl=${rl:-0}
    if [ "$rl" -gt 0 ]; then break; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then
      SEEN=1
    elif [ "$SEEN" = 1 ]; then
      sleep 5
      docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
      break
    fi
    i=$((i + 1))
  done
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  rl=$(grep -c "requires the native decode lane" "$CD/serve.log" 2>/dev/null || true)
  rl=${rl:-0}
  local skinl=0
  skinl=$(grep -c "chat_template_gemma4.jinja" "$CD/serve.log" 2>/dev/null || true)
  skinl=${skinl:-0}
  if [ "$skinl" -ge 1 ]; then
    echo "SKIN_G3[gref]: template_lines=$skinl (the refusal face rode the lawful skin)"
  else
    echo "SKIN: FAIL (gref) -- the refusal face booted without the g3 template"; GATE_FAIL=$((GATE_FAIL+1))
  fi
  if [ "$HEALTH" = "0" ] && [ "$rl" -gt 0 ]; then
    echo "GREF_PC2_REFUSAL_OK refusal_lines=$rl (boot-loud: the engine refused the windowed MX impl off-native)"
  else
    echo "GREF_PC2_REFUSAL_FAIL health=$HEALTH refusal_lines=$rl (see $CD/serve.log)"
    GATE_FAIL=$((GATE_FAIL+1))
  fi
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f $CNAME >/dev/null 2>&1
  verify_removed || { GATE_FAIL=$((GATE_FAIL+1)); return 1; }
  sleep 22
  khealth "$CD" "$K0" gref
  echo "=== [gref] done $(date +%H:%M:%S) ==="
}

stride_fix_check() {  # $1 = leg dir -- the gemma MX legs MUST show the r262
  # fix riding a multi-row synth continuation: >= 1 "MXKV native bt
  # materialized" line (the longprobe multi rehit is a 118-row ride on
  # every MX dtype -- the exact ride that killed the first cert29 image).
  # A leg that never entered the fixed path would be a vacuous pass.
  local CD=$1
  # the POST-suite count (r264b): the boot-time snapshot is always 0 here
  grep -Eq "materialized_lines_post=[1-9]" "$CD/markers.txt" \
    || { echo "STRIDE_FIX_ENGAGED: FAIL ($CD) -- no 'MXKV native bt materialized' line after the suite"; GATE_FAIL=$((GATE_FAIL+1)); }
  echo "STRIDE_FIX_ENGAGED[$(basename "$CD")]: $(grep -h 'materialized_lines_post=' "$CD/markers.txt" 2>/dev/null || echo 'materialized_lines_post=?')"
}

skin_check() {  # $1 = leg dir -- the gemma-skin directive (owner
  # 2026-11-28): every gemma boot rides the LAWFUL g3 skin --
  # ARC_TEMPLATE + EMPTY parsers (the full posture banks in
  # leg_env.txt). The SKIN assertion is the serve-log line: the api
  # server's non-default-args carries the mounted template path,
  # proving the engine APPLIED the template (the serve script mounts
  # ARC_TEMPLATE at /root/<basename> and labels arc_skin=). Templateless
  # gemma verdicts are invalid-by-posture (the r264 mkg RED class).
  local CD=$1
  grep -Eq "skin_template_lines_post=[1-9]" "$CD/markers.txt" \
    || { echo "SKIN: FAIL ($CD) -- no serve-log chat_template line; the g3 skin did not engage"; GATE_FAIL=$((GATE_FAIL+1)); }
  echo "SKIN_G3[$(basename "$CD")]: $(grep -h 'skin_template_lines_post=' "$CD/markers.txt" 2>/dev/null || echo 'skin_template_lines_post=?') $(grep -h -o 'tmpl=[^ ]*' "$CD/boot.txt" 2>/dev/null | head -1 || echo 'tmpl=?')"
}

leg_m8g() {
  run_leg m8g gemma4-31b 1 $GEMMA_NAT ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m8g/markers.txt" || { echo "MX_MARKER: FAIL m8g (expect >=1 worker)"; GATE_FAIL=$((GATE_FAIL+1)); }
  grep -Eq "native_lines=[1-9]" "$P/m8g/markers.txt" || { echo "NATIVE_POSTURE: FAIL m8g (no native capture telemetry)"; GATE_FAIL=$((GATE_FAIL+1)); }
  pc2_window_check "$P/m8g"
  grep -q "lmcache_lines=0" "$P/m8g/markers.txt" || { echo "LMC_POSTURE: FAIL m8g (LMCache engaged)"; GATE_FAIL=$((GATE_FAIL+1)); }
  stride_fix_check "$P/m8g"
  skin_check "$P/m8g"
}
leg_m4g() {
  run_leg m4g gemma4-31b 1 $GEMMA_NAT ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m4g/markers.txt" || { echo "MX_MARKER: FAIL m4g (expect >=1 worker)"; GATE_FAIL=$((GATE_FAIL+1)); }
  grep -Eq "native_lines=[1-9]" "$P/m4g/markers.txt" || { echo "NATIVE_POSTURE: FAIL m4g (no native capture telemetry)"; GATE_FAIL=$((GATE_FAIL+1)); }
  pc2_window_check "$P/m4g"
  grep -q "lmcache_lines=0" "$P/m4g/markers.txt" || { echo "LMC_POSTURE: FAIL m4g (LMCache engaged)"; GATE_FAIL=$((GATE_FAIL+1)); }
  stride_fix_check "$P/m4g"
  skin_check "$P/m4g"
}
leg_mkg() {
  run_leg mkg gemma4-31b 1 $GEMMA_NAT ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/mkg/markers.txt" || { echo "MX_MARKER: FAIL mkg (expect >=1 worker)"; GATE_FAIL=$((GATE_FAIL+1)); }
  grep -Eq "native_lines=[1-9]" "$P/mkg/markers.txt" || { echo "NATIVE_POSTURE: FAIL mkg (no native capture telemetry)"; GATE_FAIL=$((GATE_FAIL+1)); }
  pc2_window_check "$P/mkg"
  grep -q "lmcache_lines=0" "$P/mkg/markers.txt" || { echo "LMC_POSTURE: FAIL mkg (LMCache engaged)"; GATE_FAIL=$((GATE_FAIL+1)); }
  stride_fix_check "$P/mkg"
  skin_check "$P/mkg"
}
leg_g0() {  # gemma NATIVE-KV control (same fp8-weights posture as the MX
  # legs -- the comparative law: g0 differs from m8g/m4g/mkg by KV dtype
  # ONLY). FULL=1 (r170): the full suite including 3 churn rehits,
  # UNGATED -- the churn-classification discriminator. mx_backend_lines
  # MUST be 0 here.
  run_leg g0 gemma4-31b 1 $GEMMA_NAT || true
  grep -q "mx_backend_lines=0" "$P/g0/markers.txt" || { echo "G0_NATIVE: FAIL (MX backend engaged on the control leg)"; GATE_FAIL=$((GATE_FAIL+1)); }
  skin_check "$P/g0"
  echo "G0_FULL_SUITE_BANKED (ungated native baseline + churn classification)"
}

# ---- r264b: per-leg verdicts + the certificate precondition ----
# Owner directive (2026-11-27): the gemma MX lanes and the LMCache lane
# MUST be all green before a certificate is cut from this image. GATE_FAIL
# is a COUNT of failed checks; every leg runs between an entry snapshot
# and a LEG_VERDICT line, and the battery ends with the precondition line:
# CERT_CUT_ALLOWED only when every leg of the battery is GREEN in THIS
# battery's bank (latest verdict line per leg; no line = NOT-RUN =
# blocked) and this invocation added no failed check.
mkdir -p "$P/_battery"
: >> "$P/_battery/leg_verdicts.txt"
leg_verdict() {  # $1 = leg $2 = failed-check count at the leg's entry
  local N=$((GATE_FAIL - $2)) LR="" TS=""
  echo "LEG_VERDICT[$1]: $([ "$N" = 0 ] && echo GREEN || echo "RED failed_checks=$N") $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1 green=$([ "$N" = 0 ] && echo 1 || echo 0) failed_checks=$N" >> "$P/_battery/leg_verdicts.txt"
  case $1 in m8g|m4g|mkg|g0)
    # the absolute longprobe classes are BANK-ONLY on gemma (comparative
    # law, r164: the native g0 baseline fails the same classes). Printed
    # here so a GREEN gemma leg never hides what the absolute classes said.
    LR=$(grep -o "LONGRETURN: [A-Z]*" "$P/$1/longreturn_summary.txt" 2>/dev/null | head -1)
    TS=$(grep -o "T1_SEED: [A-Z]*" "$P/$1/longreturn_summary.txt" 2>/dev/null | head -1)
    echo "GEMMA_ABS_CLASSES[$1]: ${LR:-LONGRETURN: not-run} | ${TS:-T1_SEED: not-run} (bank-only; the gemma verdict is gemma_compare)" ;;
  esac
}
run_one() {  # $1 = leg name -- one leg inside its verdict lines
  local F0=$GATE_FAIL
  case $1 in
    compare) gemma_compare ;;
    ctrl|m8q|m4q|mkq|lmc|gref|m8g|m4g|mkg|g0) "leg_$1" ;;
    *) echo "UNKNOWN_LEG $1 (ctrl|m8q|m4q|mkq|lmc|gref|m8g|m4g|mkg|g0|compare)"; GATE_FAIL=$((GATE_FAIL+1)) ;;
  esac
  leg_verdict "$1" "$F0"
}
state_of() {  # $1 = leg -> GREEN | RED | NOT-RUN (latest verdict line in this battery's bank)
  local line
  line=$(grep -E " $1 green=[01] failed_checks=[0-9]+$" "$P/_battery/leg_verdicts.txt" 2>/dev/null | tail -1)
  case $line in
    *" green=1 "*) echo GREEN ;;
    *" green=0 "*) echo RED ;;
    *) echo NOT-RUN ;;
  esac
}

if [ $# -gt 0 ]; then
  for LEG in "$@"; do run_one "$LEG"; done
else
  # full battery (r264 order: gemma first -- the fix's validation, the
  # legs that killed the first cert29 image -- then the qwen side, re-
  # certified because the bridge bytes changed): gref gates the P-C2
  # refusal; the gemma MX legs are bank-only per leg + the comparative
  # gemma_compare verdict after the g0 native control; the qwen legs gate
  # ABSOLUTE (clean baseline); the lmc leg gates the P-H interop feature
  # (knob-free).
  run_one gref; run_one m8g; run_one m4g; run_one mkg; run_one g0
  run_one compare
  run_one ctrl; run_one m8q; run_one m4q; run_one mkq; run_one lmc
fi

# whole-battery kernel health (r264): nothing between/after legs either
: > "$P/_battery/markers.txt"
sleep 22
khealth "$P/_battery" "$GATE_K0" battery

# the certificate precondition (owner directive 2026-11-27): the gemma MX
# lanes + LMCache all green, and with them the rest of the battery.
# The marker NAME R264_CERT_PRECONDITION is law-pinned (CLAUDE.local.md
# Law 11 greps exactly this form) -- do not rename it per-run.
PRE_OK=1
REQ=""; REST=""
for l in m8g m4g mkg lmc; do REQ="$REQ $l:$(state_of $l)"; done
for l in gref g0 compare ctrl m8q m4q mkq; do REST="$REST $l:$(state_of $l)"; done
for l in m8g m4g mkg lmc gref g0 compare ctrl m8q m4q mkq; do
  [ "$(state_of $l)" = GREEN ] || PRE_OK=0
done
[ "$GATE_FAIL" = 0 ] || PRE_OK=0
echo "R264_CERT_PRECONDITION gemma_mx+lmcache=[$REQ ] rest=[$REST ] gate_fail=$GATE_FAIL -> $([ "$PRE_OK" = 1 ] && echo CERT_CUT_ALLOWED || echo CERT_CUT_BLOCKED)"

cd "$P"
find . -type f \( -name "*.out" -o -name "*.txt" -o -name "*.json" -o -name "*.sha256" -o -name "serve.log" -o -name "metrics.txt" -o -name "native_telemetry.txt" -o -name "longreturn_summary.txt" \) -exec sha256sum {} \; > SHA256SUMS
echo "LEG_ARTIFACTS $P"
echo "R271_GATE_END $(date -u +%Y-%m-%dT%H:%M:%SZ) image=$IMG fork_sha=$FORK_SHA bank=$P gate_fail=$GATE_FAIL cert_cut=$([ "$PRE_OK" = 1 ] && echo ALLOWED || echo BLOCKED)"
if [ "$GATE_FAIL" = 0 ] && [ "$PRE_OK" = 1 ]; then
  echo "P647_CERT30_GATE_COMPLETE -- PASS (analysis/bank next)"
else
  echo "P647_CERT30_GATE_COMPLETE -- GATE_FAIL=$GATE_FAIL (fix before cert)"
fi
if [ "$GATE_FAIL" -gt 0 ]; then exit 1; fi
exit 0
