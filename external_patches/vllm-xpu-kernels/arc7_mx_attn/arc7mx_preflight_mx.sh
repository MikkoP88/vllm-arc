#!/bin/bash
# arc7mx_preflight_mx.sh — r198 fast compile-iteration loop: compile ONLY
# the mx TU with its exact CMake compile command (arc8 lesson: pre-flight
# the changed TU with ninja-level fidelity before any full wheel build).
# Runs INSIDE the builder container (docker exec). Reads the command from
# build/temp/compile_commands.json, redirects -o to /tmp/mx_test.o so
# ninja state stays untouched. Exit code = compiler exit code.
set -uo pipefail
DB=/src/vllm-xpu-kernels/build/temp/compile_commands.json
[ -f "$DB" ] || { echo "PREFLIGHT_NO_COMPILE_DB (configure first)"; exit 9; }

CMD=$(/opt/arcvenv/bin/python - "$DB" <<'PY'
import json, sys
db = json.load(open(sys.argv[1]))
for e in db:
    if e["file"].endswith("mx/mx_paged_decode_xe2.cpp"):
        print(e["command"].replace(
            "-o csrc/xpu/attn/xe_2/CMakeFiles/attn_kernels_xe_2.dir/mx/mx_paged_decode_xe2.cpp.o",
            "-o /tmp/mx_test.o"))
        break
else:
    print("PREFLIGHT_NO_MX_ENTRY", file=sys.stderr)
    sys.exit(9)
PY
)
[ -n "$CMD" ] || exit 9
echo "PREFLIGHT_START $(date +%H:%M:%S)"
cd /src/vllm-xpu-kernels/build/temp
eval "$CMD" 2>&1 | tail -${PREFLIGHT_TAIL:-40}
rc=${PIPESTATUS[0]}
echo "PREFLIGHT_EXIT rc=$rc $(date +%H:%M:%S)"
exit "$rc"
