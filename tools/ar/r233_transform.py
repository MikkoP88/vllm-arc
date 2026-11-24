#!/usr/bin/env python3
"""r233_transform.py -- derive tools/ar/r233_cert28_gate.sh from
r228_cert27_gate.sh (cert28 deltas only; behavior-preserving per the
oracle-diff law: every replacement is provenance-anchored, the
transform refuses on any drift)."""

import pathlib
import sys

SRC = pathlib.Path("tools/ar/r228_cert27_gate.sh")
DST = pathlib.Path("tools/ar/r233_cert28_gate.sh")

HEADER = """#!/bin/bash
# r233_cert28_gate.sh (r228 + cert28 deltas) -- cert28 (P6.47: the
# r231d seqused-tightened-split-clamp arc7mx build28 wheel) section-10
# gate. Image vllm-arc:v0.30.0-b70-cert28 = cert27 content + the
# arc7mx build28 wheel (vllm_xpu_kernels 0.1.14.1+arc7mx, md5
# e6d8d337 byte-pinned, BAKED; libattn_kernels_xe_2 .so md5
# 61d2f5a3 + _vllm_fa2_C 6e6c4850 + libmqa_logits 190e261e all
# asserted) -- the mx_kv overlay is byte-identical to cert27's
# (inherited via FROM). Kernel delta: device-side split clamp
# min(num_kv_splits, max(1, windowed_k_blocks/4)) mirrored in the
# decode kernel + the split reduce; no-op at >=8K walked ctx.
# Evidence: r231d offline gates (G1/G2/G3 bitwise 12/12/3, 12 r229
# band cells, ladder 2K -43/-39/-51 pct with 8K+ neutral) + r231de
# e2e (3 boots, A/A 0.0-0.1 pct: c2048 -7.6 pct, c4096 -5.6,
# c8192 0.984x, c32768 1.000x). vs r228 (cert27):
#   - image-content markers carry the cert28 .so pin SET (libattn
#     + fa2 + libmqa) + manifest fork_sha refuse (passed as $1 --
#     the content commit).
# Leg semantics otherwise unchanged from r228 (qwen absolute, gemma
# comparative vs same-image g0 native control; r170b churn amendment
# carried). trap restores W14.
# Artifacts -> ${P647_BANK:-/root/build/p647/cert28}/<leg>/.
"""

src = SRC.read_text(encoding="utf-8")

# 1. replace the header (everything before `set -uo pipefail`)
anchor = "set -uo pipefail"
idx = src.index(anchor)
body = src[idx:]

# 2. provenance-anchored replacements -- refuse on drift
REPL = [
    ("IMG=${IMG:-vllm-arc:v0.30.0-b70-cert27}",
     "IMG=${IMG:-vllm-arc:v0.30.0-b70-cert28}"),
    ("CNAME=arc-c27", "CNAME=arc-c28"),
    ("P=${P647_BANK:-$R/p647/cert27}", "P=${P647_BANK:-$R/p647/cert28}"),
    ("usage: r228_cert27_gate.sh <FORK_SHA> [legs...]",
     "usage: r233_cert28_gate.sh <FORK_SHA> [legs...]"),
    # the cert27 wheel pin -> the cert28 three-pin set
    ("""# cert27 wheel pin: the r225-fixed build25 libmqa_logits .so
import hashlib, pathlib, vllm_xpu_kernels
sos = sorted(pathlib.Path(vllm_xpu_kernels.__file__).parent.rglob("libmqa_logits*.so"))
ok = ok and bool(sos) and hashlib.md5(sos[0].read_bytes()).hexdigest() == "190e261edc79b3b497d8d8e801e65757"
print("IMG_CONTENT_OK" if ok else "IMG_CONTENT_STALE")""",
     """# cert28 wheel pins: the build28 clamp carrier + stub + unrelinked mqa
import hashlib, pathlib, vllm_xpu_kernels
vp = pathlib.Path(vllm_xpu_kernels.__file__).parent
ok = ok and hashlib.md5((vp / "libattn_kernels_xe_2.so").read_bytes()).hexdigest() == "61d2f5a365f05084b3dd72c4ae1da3e0"
ok = ok and hashlib.md5((vp / "_vllm_fa2_C.abi3.so").read_bytes()).hexdigest() == "6e6c48508d40cb4c5007f27f79df97cf"
sos = sorted(vp.rglob("libmqa_logits*.so"))
ok = ok and bool(sos) and hashlib.md5(sos[0].read_bytes()).hexdigest() == "190e261edc79b3b497d8d8e801e65757"
print("IMG_CONTENT_OK" if ok else "IMG_CONTENT_STALE")"""),
    ("'grep -o \"cert27_fork_sha=.*\" /etc/vllm-arc/IMAGE_MANIFEST'",
     "'grep -o \"cert28_fork_sha=.*\" /etc/vllm-arc/IMAGE_MANIFEST'"),
    ("grep -q \"cert27_fork_sha=$FORK_SHA\"",
     "grep -q \"cert28_fork_sha=$FORK_SHA\""),
    ("R228_FORK_SHA_MISMATCH", "R233_FORK_SHA_MISMATCH"),
    ("R228_FORK_SHA_OK", "R233_FORK_SHA_OK"),
    ("R228_STALE_IMAGE", "R233_STALE_IMAGE"),
    ("R228_IMAGE_CONTENT_OK", "R233_IMAGE_CONTENT_OK"),
    ("w14_restore_p647c27.log", "w14_restore_p647c28.log"),
    ("p647_longprobe.py c27$LEGN full", "p647_longprobe.py c28$LEGN full"),
    ("p647_longprobe.py c27${LEGN}r$cyc rehit", "p647_longprobe.py c28${LEGN}r$cyc rehit"),
    ("longreturn_summary c27$LEGN", "longreturn_summary c28$LEGN"),
    ("# ---- image-content markers: refuse a stale cert27 image ----",
     "# ---- image-content markers: refuse a stale cert28 image ----"),
]
# the completion token appears on BOTH the PASS and GATE_FAIL lines
PAIR = ("P647_CERT27_GATE_COMPLETE", "P647_CERT28_GATE_COMPLETE")
if src.count(PAIR[0]) != 2:
    sys.exit(f"TRANSFORM_REFUSE: {PAIR[0]!r} count={src.count(PAIR[0])}")
for old, new in REPL:
    if src.count(old) != 1:
        sys.exit(f"TRANSFORM_REFUSE: {old[:60]!r} count={src.count(old)}")
    body = body.replace(old, new, 1)
body = body.replace(*PAIR)

# the STALE_IMAGE message mentions cert27's fixed-wheel line; retarget
if body.count("missing cert27 fixed-wheel line") != 1:
    sys.exit("TRANSFORM_REFUSE: stale-image message anchor drifted")
body = body.replace("missing cert27 fixed-wheel line",
                    "missing cert28 clamped-wheel line", 1)

DST.write_text(HEADER + body, encoding="utf-8", newline="\n")
print(f"r233 written ({len(HEADER + body)} bytes)")
