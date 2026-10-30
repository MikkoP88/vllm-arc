#!/usr/bin/env python3
"""arc r34 fix: disable_eagle_block_drop:true in the speculative-config JSON."""
import py_compile

SERVE = "/root/build/r30/r30_serve.sh"

OLD = """--speculative-config '{"method":"mtp","num_speculative_tokens":4}' \\"""
NEW = """--speculative-config '{"method":"mtp","num_speculative_tokens":4,"disable_eagle_block_drop":true}' \\"""

src = open(SERVE, encoding="utf-8").read()
n = src.count(OLD)
assert n == 1, f"anchor count {n} != 1"
open(SERVE, "w", encoding="utf-8").write(src.replace(OLD, NEW))

s = open(SERVE, encoding="utf-8").read()
assert NEW in s and s.count("disable_eagle_block_drop") == 1
assert s.count("--port 8000") == 1  # tail intact
print("R34_FIX_PATCHED")
print([l for l in s.splitlines() if "speculative-config" in l][0])
