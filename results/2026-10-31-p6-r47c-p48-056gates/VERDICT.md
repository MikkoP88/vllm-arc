# r47c — p48 image gates (P6.2 lmcache 0.5.6+arc1 + P6.3 serve hygiene)

**VERDICT: GREEN — p48 (fork fdca6f51, image 7b091de80061) passes every
P6 gate: 28/28 image-native checks, certified boot with the 25-var chain,
zero fork-added WARN/ERROR across boot+battery, and performance within
±5 % of the certified 0.5.5 stack on the SAME fresh-boot posture.**

## What ran

1. **Image-native verify** (`p48_verify.py`, throwaway GPU container):
   28/28 PASS, `P48_VERIFY: ALL_GREEN` — the r46u 13-check set (glue
   alias, 5 glue symbols, `_glue` primary+fallback, p41c wrap live,
   post-wrap resolve) + the 0.5.6 set (version pin, `xpu_ops` native
   import, e01 5-op transfer re-pin incl. `single_layer_kv_transfer_sgl`,
   e01 p40c env-gated pinned-pair re-pin — as installed-source markers —
   `tensor_from_ptr` override, `is_usage_tracking_enabled()=False` under
   `LMCACHE_TRACK_USAGE=false`, r43 hooks registered, `lmc_xpu_pinned_env`
   hook gone). Run 1 (26 checks, `p48_verify_run1_26checks.out`) had the
   composition check FAIL — false alarm, settled by `p48_probe.py`:
   `torch_ops.multi_layer_kv_transfer` IS the r43 wrap (the engine's
   `ensure_native` re-pin reads that attribute later, at engine init;
   the verify container never reaches engine init, so instance attrs
   were the `base.device_ops` defaults). Check corrected to test the
   wrap-on-torch_ops precondition.

2. **Certified boot** (serve_arc_b70.sh, k3 + graphs + the 25-var chain
   incl. `LMCACHE_TRACK_USAGE=false`): HEALTH=OK in 285 s.
   - p40c re-pin LIVE both workers: `XpuDeviceOps:
     LMCACHE_DISABLE_PINNED_MEMORY set; alloc/free_pinned_ptr re-pinned
     to the pageable torch baseline` → pools allocate pageable (Mamba
     engine CPU pool 100→50 GB), zero malloc_host failures.
   - glue-unavailable **0**; `[p41c] wrap live` both workers.
   - r43 detector live: `NL_X_NB_BS_NH_TWO_HS_Spec.__init__ is
     deprecated…` (0.5.6 renamed the spec; still accepted — mapping
     works).

3. **Hygiene audit** (`serve_warn_error.txt`, boot + batteries): **zero
   fork-added WARN/ERROR**. Every remaining line is the upstream-accepted
   set (KVConnectorBase_V1 experimental, LMCache no-config-file/builtin-
   hash/controller-sender, TWO_HS-spec deprecation, draft-group note,
   XPU-graph experimental, jit_monitor JIT spikes, generation_config
   override, raw-prompts deprecation). The three P6.3 targets are gone:
   no `different metadata` ERROR (e04 + TRACK_USAGE), no
   `install failed at register()` ×12 (demoted), no `ARC_GRAPH_TRACE`
   ×2 (demoted); Q40 stanza removed from the image entirely.

4. **Performance** — run7n (agg4x256 ×2 + solo200 ×2, xhigh thinking),
   same fresh-boot posture on BOTH stacks:

| face | cert2 (0.5.5) fresh | p48 (0.5.6) fresh | delta |
|---|---|---|---|
| greedy t0 agg #1 (warmup) | 81.55 | 85.04 | +4.3 % |
| greedy t0 agg #2 (steady) | 206.06 | 212.02 | **+2.9 %** |
| greedy t0 solo ×2 | 64.91 / 64.38 | 64.27 / 64.31 | **−0.6 %** |
| t1.0 agg #1 | 128.20 | 124.65 | −2.8 % |
| t1.0 agg #2 | 193.21 | 185.97 | −3.7 % |
| t1.0 solo #1 | 57.44 | 31.94 * | — (JIT spike) |
| t1.0 solo #2 (steady) | 62.40 | 62.33 | **−0.1 %** |

\* p48's t1.0 solo #1 hit the first-temperature-shape sampler JIT
(`_topk_topp_kernel`, visible in the audit at 22:28:02); solo #2 is the
steady face. All steady faces within ±5 %: **gate PASS**.

**Comparator law (recorded for future gates):** the r46t campaign numbers
(greedy 299.9 agg / 84.5 solo) are WARM-FLEET numbers — hours of prior
legs on one boot, all JIT shapes warm, caches hot. A fresh-boot first
battery on the SAME certified cert2 stack measures 206/64.5 — i.e. the
correct gate comparator for a fresh-boot validation leg is a same-posture
A/B, not the campaign table. (First battery on p48 originally read as a
−24 % "regression" against r46t; the cert2 A/B leg disproved it in one
boot. Raw: `ab_k3_t*_run7n.out`.)

## The run-1 failure (E-LAW capture, fixed same session)

Run 1 (`gate_run1_STALEHUNK_FAIL.log`, wheel md5 7bde4e4c…) died at boot:
`sycl::malloc_host failed for 107374182400 bytes (XPU pinned host)` in
both workers — the 100 GiB CPU pool went down the NATIVE pinned path.
Root cause: the first e01-xpu-platform-ops.patch's device_ops hunk was a
STALE REVISION — it carried only the 4-op transfer re-pin, missing BOTH
`single_layer_kv_transfer_sgl` AND the env-gated `alloc_pinned_ptr`/
`free_pinned_ptr` re-pin (the mem_alloc `_use_pinned` gate hunk HAD
landed, but 0.5.6's `memory_management._resolve_pinned_alloc_free`
resolves `device_ops.alloc_pinned_ptr` on the singleton instance, so the
mem-level gate alone is not on the call path). Standalone re-apply of the
kit patch on pristine proved the hunk itself lacked the markers
(`device_ops.py:0`) — patch "success" ≠ patch content. Fixed by
regenerating e01 from trees (113 lines; markers device_ops 4 +
mem_alloc 3; py_compile OK), rebuilding the wheel (md5
352295b4af1e2c58da8a2363b05a4f0c), and HARDENING build_lmcache_arc.sh
with six post-patch semantic-marker assertions (`PATCH_MARKERS_OK`) so a
silently-stale hunk can never ship again. Commit fdca6f51.

## Artifacts

- `gate2.log` — full run-2 gate output (verify 28/28, boot, batteries, audit, keylines)
- `gate_run1_STALEHUNK_FAIL.log` + `p48_verify_run1_26checks.out` — run-1 records
- `p48_verify.py` / `p48_probe.py` / `p48_gate.sh` — the leg tools
- `k3_t*_run7n.out` (p48) + `ab_k3_t*_run7n.out` (cert2 A/B) — raw batteries
- `serve_full.log` / `serve_boot_engine.log` / `serve_warn_error.txt` — engine evidence
- Wheel: `lmcache-0.5.6+arc1-cp312-cp312-manylinux_2_35_x86_64.whl`
  md5 352295b4af1e2c58da8a2363b05a4f0c (pins re-pinned; build log
  rig /root/build/lmc056_build2.log: PATCH_MARKERS_OK 6/6, sycl-spir64
  offload bundle, RETAG_VERIFIED)

Rig end-state: arc containers removed; W14 incumbent restored
(radiance-test + w14_serve_only.sh, log /root/build/w14_restore_r47c.log).
