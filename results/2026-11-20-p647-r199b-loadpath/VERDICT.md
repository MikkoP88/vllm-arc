# r199b — MX-native decode load-path redesign (P6.47 Phase 8, run 2026-11-20)

## VERDICT: CORRECTNESS PASS (all r198 faces) / PERF NEUTRAL-PLUS — attribution moved to the V lane

Kernel: arc7_mx_attn r199b — the r199a V1 attribution follow-up. The
dequant's scattered per-element gmem reads are replaced by real
vectorized 2D-block byte copies (e4m3 register path via
`copy(copy_kb/copy_vb, ...)` into `tKrB/tVrB` + `dequant_reg` +
mirror-fragment `reorder`), staged u32 byte-windows for e2m1 K
(`dequant_dtile` staged branch, compile-time form gate on the val-mode
nesting `((2,8),N)`), and the verbatim r198 scalar walk as the e2m1 V
fallback (V's val nesting is `(2,8,2)` — form (ii), no compressible
windows; the staged branch never instantiates for it).

Wheel: `vllm_xpu_kernels-0.1.14.1+arc7mx` md5 `29edb12d45ce6e8999ded78c0ee1a9dd`
(state copy `wheels-arc-vxk/fix/...+arc7mx-r199b.whl`). Source = this
commit's `external_patches/vllm-xpu-kernels/arc7_mx_attn`.

## Faces (r198_native_offline on the r199b wheel) — all PASS

- native-exact / native-v2 / robust (adversarial, signed, len 1/15/16/17,
  shuffled tables) / splits (forced-8 + default): every cell `ok`,
  worst d = 6.96e-03 (len16 mxfp4, same envelope as r198 PASS).
- **bitwise (seq_len==1 dequant bit oracle): 0/1024 mismatches on all
  three formats** — the new load path delivers byte-identical inputs to
  an unchanged decode law.
- Telemetry face: 32K native mxfp8 5.845 ms vs r198 6.111 ms (+5%);
  mxfp4 5.573; mxk8v4 6.423.

## Attribution (p199_ablation, device ms @ 1K/8K/32K)

| lane | K path | V path | 1K | 8K | 32K |
|---|---|---|---|---|---|
| mxfp8 | copy+reg | copy+reg | 1.956 | 1.426 | 5.496 |
| mxfp4 | staged u32 | scalar (r198) | 1.762 | 1.403 | 5.407 |
| mxk8v4 | copy+reg | scalar (r198) | 1.925 | 1.604 | 6.194 |
| V1 const-fill floor (r199a) | — | — | 0.175 | 0.101 | 0.236 |

Laws extracted:

1. **V lane is the bottleneck in BOTH forms.** Triangulating the three
   lanes: V-scalar alone ≈ 5.2-5.3 ms of mxfp4's 32K 5.407; V-copy ≈
   4.4 ms of mxfp8's 5.496. K is effectively solved (staged windows
   cost ≈ 0.1 ms over the floor; K-copy ≈ 0.9).
2. **The V copy is NOT faster than the V scalar walk** (mxfp8 ≈ mxfp4
   within 2%): form (ii)'s 8-rows-x-stride-4 fragment layout defeats
   the 2D-block copy just as it defeats window compression — the
   hardware gather lowers to near-scalar efficiency. A structural fix
   (SLM staging, store-time V repack, or a transposed copy atom) is
   required; tuning the walk is exhausted.
3. **The V1 const-fill floor is a lower bound, not a target**: filling
   constants also severs the copy->dequant->reorder dependency chains
   (and everything they feed), so 0.236 ms was never reachable by
   fixing reads alone. Use const-fill ablations only as structure
   floors; attribute via cross-format triangulation on real wheels.
4. **1K≈8K ≈ 1.4 ms plateau = latency-bound low-occupancy regime**:
   with few CTAs the per-k-tile dependent chain (copy -> dequant ->
   reorder -> gemm) is fully exposed. Software pipelining
   (double-buffered byte fragments across k-tiles) attacks both this
   and the 32K serialization.

## Next (r200)

1. V structural fix: cooperative SLM staging of the V tile (subgroup-
   collective coalesced load + per-WI SLM gather) or engine-side V
   repack at `mx_store` time (we own the quantizer lane).
2. K/VV software pipelining (double-buffer tKrB/tVrB).
3. Re-run this battery; superiority vs certified v2 Triton (0.766 ms
   @32K mxfp8) is the stop condition, not parity.

## Rig state at close

W14 restored (fullres, 03:21 UTC). No arc containers left running.
