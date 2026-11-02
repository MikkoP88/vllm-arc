# p614 preflight — CPU-only validation of the patched view fn (2026-11-02)

**What:** `rig/test_tq_view2.py` extracts the patched
`_arc_attn_pool_view` from the fix-v3 adapter by AST, installs the kit
overrides exactly like the platform hook (lmc_kvfmt_detector on the
VLLM_Detector host module, lmc_twohs_transfer on torch_ops), and drives
it at the LIVE geometry measured by the dbg boot: H_kv 2, head 256,
kernel block 128, page 1664, L 17, pool built as
`(NBK, KBLK, row).view(NBK,KBLK,H,slot).permute(0,2,1,3)` (strides
verified == `(KBLK*row, slot, row, 1)` — the exact strides the dbg boot
observed). Slots come from the wheel's
`TurboQuantConfig.from_cache_dtype(preset, 256)`.

**Result: `rig/preflight_v2.out` — PRE_FLIGHT_V2: OK (28/28 PASS):**

- all four presets (4bit_nc slot 262, k8v4 388, k3v4_nc 230, 3bit_nc
  198) produce merged zero-copy page views `(NPAGES, 1664, 1, 2, HS)`
  with `data_ptr` equal to the pool;
- the INSTALLED kit detector classifies them NL_X_NB_BS_NH_TWO_HS at
  BS=1664;
- multi-page D2H (page 0 full + page 5 partial 777 tokens) is
  byte-true against the kernel-block physical storage, and H2D writes
  back byte-true with untouched pages left zeroed — both directions
  through the INSTALLED twohs transfer branch;
- the fp8-shaped pool takes the r43 fp8 face unchanged with the TQ
  flag OFF and ON (first branch declines it);
- negative guards: non-LBNHC contiguous TQ pool raises loudly;
  non-page-tiling kernel count returns the unmerged 128 face (a
  visible detect failure, never silent).

`rig/test_tq_view.py` (v1) is retained as the historical fix-v1
pre-flight; re-running it against fix-v3 code errors on a Shim missing
`_arc_tq_page_tokens` by construction (v1 predates the attribute), so
its rerun output was not banked.

No addresses present in any preflight artifact (scanned clean).
