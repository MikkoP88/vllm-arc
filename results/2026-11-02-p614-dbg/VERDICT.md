# p614 dbg — instrumented boot that pinned the L1v1 root cause (2026-11-02)

**Transient debug evidence (never ships):** `rig/vllm_v1_adapter.py` is
the instrumented adapter (module-level `_ARC_DBG_SEEN` set; one log line
per distinct geometry entering `_arc_attn_pool_view`; spec-loop +
init-line logs). It was bind-mounted read-only over site-packages inside
a p614c1 container by `p614_dbg_boot.sh` (one flip request, then the
`[p614dbg]` lines were grepped out). Boot `rig/boot_dbg.out`, flip
`rig/flip_dbg.out`, extracted lines `rig/dbg_lines.txt`.

## The three-line finding

```
[p614dbg] view-in shape=(6500, 2, 128, 262) stride=(67072, 262, 524, 1) dtype=torch.uint8
[p614dbg] fa-spec FullAttentionSpec kvq=tq=1 scb=262 nhs=256 nkh=2 bs=1664
[p614dbg] init cache_dtype=turboquant_4bit_nc
```

1. **The pool arrives as 128-token KERNEL blocks**, not 1664-token
   manager pages: `(6500, 2, 128, 262)` — `create_kv_cache_views` in the
   wheel subdivides each dense page for the TQ kernel (13×128 = 1664).
   Fix v1's reshape assumed page granularity, so the detector's
   `BS >= 256` bound rejected the BS=128 view.
2. **H_kv == 2 @TP2** (`nkh=2`), so `shape[1] == 2` ALSO satisfies the
   r43 fp8 guard — the fp8 branch swallowed the TQ pool first and
   returned a byte-correct-by-coincidence (permute semantics identical
   at H_kv=2) but still unmerged 5-D view. That is why fix v1's branch
   never fired: the fp8 branch sat above it.
3. Live geometry: head 256 (`nhs=256`), H_kv 2, slot 262 = the row half;
   head-268 static reading from the wheel docs was wrong.

## Consequence (fix v3, commit 6f1a09d9)

- TQ branch moved FIRST in `_arc_attn_pool_view`; fp8 branch untouched
  below it.
- Zero-copy `permute(0,2,1,3)` then merge kernel blocks back to PAGE
  granularity `(NB_pages, 1664, 1, 2, HS)` when they tile the page;
  loud `ValueError` on non-LBNHC pools; unmerged 128 face otherwise
  (visible detect failure, never silent).
- Engine init takes slot bytes from
  `TurboQuantConfig.from_cache_dtype(..., model_config.get_head_size())`
  — self-correcting to the live head size, no static constant.

W14 restored by the boot's EXIT trap (`w14_boot_p614dbg.log`).
No addresses present in any dbg artifact (scanned clean).
