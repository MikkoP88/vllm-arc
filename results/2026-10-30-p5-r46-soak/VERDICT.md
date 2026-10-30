# r46 — 8 h soak + Phase A gates on the merge tree (P5 gate of record)

**VERDICT: GREEN** — the §10 P5 stability gate passes on the merge-tree
lineage (p46 image, tip-lineage 69d8a154).

## Soak (8 h gate)

Run window: 29,156 s across **46 cycles** (gate: ≥ 28,800 s / 8 h), each
cycle = run7n → longctx → prefix/churn → integrity probe, against
`vllm-arc:v0.30.0-b70-p46` booted from the certified line (k5 canonical).

| Check | Result |
|---|---|
| error blocks / tracebacks in serve log | **0** |
| integrity probes (5-prompt sha set) | **46/46 PASS** (zero drift) |
| final golden cells (md5 vs banked) | **5/5 byte-identical** |
| speculative pos-1 acceptance | **0.8919** steady through the last cycle |
| /health end-state | **200** |
| hook posture at audit | 18 installs / 0 failed, FP8 W8A16 ×4 ACTIVE |

## mypy parity (A/B-proven, delta = 0)

- Toolchain: mypy 1.11.1 + pydantic 2.13.5, identical pins in the p45b
  and p46 containers (cache-inherited stack layer, no drift).
- mypy 1.11.1 × pydantic 2.13.5 crashes in cache **write** serialization
  (`unresolved placeholder type None`, pydantic.mypy plugin class) —
  bypass `MYPY_CACHE_DIR=/dev/null` affects only cache write, not
  checking. Upstream's `tools/mypy.sh` CRLF (mirror tag blob) fixed
  in-container (`sed -i "s/\r$//"`).
- **Our tree vs pristine upstream v0.30.0, identical environment:
  identical 21-error sets — delta = 0** (6 upstream kv_offload
  zmq-stub gaps + 15 upstream v1 ndarray/annotation errors, all in
  upstream's own files). Upstream face itself = `tools/mypy.sh 0 local`.

## 64k TTFT face (last never-measured superiority face)

`vllm bench serve` random 65,536-token input, output 512, ignore-eos,
**t1 TTFT = 40,926.31 ms vs 41,565 ms bar — BEATEN**, taken while the
soak battery was cycling (concurrent-load conservative upper bound).
TPOT 24.73 ms; acceptance 87.43 % / 5.37 mean length; per-position
96.79 / 96.79 / 96.79 / 96.33 / 50.46 (pos 4 = k5 draft tail).

## Evidence

Rig leg `arc-p5-r46/` banked beside this VERDICT: `soak.log`,
`summary.txt`, `hook_audit.txt`, `mypy_parity.txt`, `mypy_nocache.txt`,
`mypy_upstream_baseline.txt`, `ttft64k_t1.txt`.
