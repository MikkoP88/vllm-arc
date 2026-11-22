#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p212a_patch_serve.py -- idempotent rig-side patcher for
/root/build/serve_arc_b70.sh: adds three INERT hooks the r212a
native fastcycle needs (all no-ops unless the caller sets the vars):

1. docker-run mounts (after the ARC_MODEL mount line):
   ${ARC_MXKV_OVERLAY:+-v ...:/.../mx_kv:ro}          -- engine.py overlay
   ${ARC_MXKV_WHEEL_DIR:+-v ...:/wheels:ro}           -- arc7mx wheel dir
2. pre-serve wheel install (before the docker exec -d serve line):
   ARC_MXKV_WHEEL=<in-container path> -> pip install --force-reinstall
3. exec env pass-through (next to ARC_MXKV_VERIFY_DECODER):
   ARC_MXKV_NATIVE / ARC_MXKV_NATIVE_MINLEN

Rationale: the cert25zc image predates the native-bridge engine
commit (ac2b1989), so the first native e2e needs the mx_kv tree and
the arc7mx wheel injected at boot. Verified by grep before/after;
re-runs are no-ops. Usage: python3 p212a_patch_serve.py [PATH].
"""

import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/root/build/serve_arc_b70.sh"

with open(PATH, encoding="utf-8", newline="") as f:
    s = f.read()

orig = s

MOUNT_ANCHOR = '    -v "$ARC_MODEL":/models/target:ro \\\n'
MOUNT_ADD = (
    '    ${ARC_MXKV_OVERLAY:+-v "$ARC_MXKV_OVERLAY"'
    ":/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches"
    '/mx_kv:ro} \\\n'
    '    ${ARC_MXKV_WHEEL_DIR:+-v "$ARC_MXKV_WHEEL_DIR":/wheels:ro} \\\n'
)

SERVE_ANCHOR = 'docker exec -d "$ARC_NAME" bash -c "'
WHEEL_BLOCK = (
    "# arc7 native fastcycle (r212a): optional wheel pre-install -- "
    "inert\n"
    "# unless ARC_MXKV_WHEEL is set (path INSIDE the container).\n"
    'if [ -n "${ARC_MXKV_WHEEL:-}" ]; then\n'
    '  docker exec "$ARC_NAME" /opt/venv/bin/pip install --quiet '
    '--force-reinstall --no-deps "$ARC_MXKV_WHEEL" \\\n'
    '    || { echo "ARC_MXKV_WHEEL_INSTALL_FAIL"; exit 1; }\n'
    "fi\n\n"
)

ENV_ANCHOR = (
    "${ARC_MXKV_VERIFY_DECODER:+export "
    "ARC_MXKV_VERIFY_DECODER=${ARC_MXKV_VERIFY_DECODER};}\n"
)
ENV_ADD = (
    "${ARC_MXKV_NATIVE:+export ARC_MXKV_NATIVE=${ARC_MXKV_NATIVE};}\n"
    "${ARC_MXKV_NATIVE_MINLEN:+export "
    "ARC_MXKV_NATIVE_MINLEN=${ARC_MXKV_NATIVE_MINLEN};}\n"
)

if "ARC_MXKV_OVERLAY" not in s:
    assert s.count(MOUNT_ANCHOR) == 1, "mount anchor not unique"
    s = s.replace(MOUNT_ANCHOR, MOUNT_ANCHOR + MOUNT_ADD)
    print("added: overlay/wheel-dir mounts")
if "${ARC_MXKV_WHEEL:-}" not in s:
    assert s.count(SERVE_ANCHOR) == 1, "serve anchor not unique"
    s = s.replace(SERVE_ANCHOR, WHEEL_BLOCK + SERVE_ANCHOR)
    print("added: wheel pre-install block")
if "ARC_MXKV_NATIVE:" not in s:
    assert s.count(ENV_ANCHOR) == 1, "env anchor not unique"
    s = s.replace(ENV_ANCHOR, ENV_ANCHOR + ENV_ADD)
    print("added: ARC_MXKV_NATIVE env hooks")

if s != orig:
    with open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(s)
    print("PATCHED", PATH)
else:
    print("ALREADY_PATCHED", PATH)
for probe in ("ARC_MXKV_OVERLAY", "${ARC_MXKV_WHEEL:-}", "ARC_MXKV_NATIVE:"):
    assert probe in s, f"post-check failed: {probe}"
print("POSTCHECK_OK")
