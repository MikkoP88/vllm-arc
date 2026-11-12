# P6.47 leg B — MX KV V-stride store bug: root cause, fix, validation (banked 2026-11-12)

Branch `feat/arc-p647-mxkv-engine-vstride-fix`, content commit `83c81c95`
(+ cert24 inputs `40fa6830`). Rig artifacts: `/root/build/p149/trace24spec`
…`trace28fp4eager` (full serve logs + xcheck3 dumps stay rig-side; SHA256SUMS
banked per trace below). Small-file evidence mirrored in `raw/`.

## Root cause (trace24, 2026-11-12)

`triton_mx_store` passed keys' token+head strides for BOTH the k and v loads.
In-engine, key and value arrive at the store call with INDEPENDENT layouts —
the wheel's own store kernels take `stride_val_tok` / `stride_val_head`
separately (wheelref `int4_per_token_head.py`). Sharing keys' strides made the
v load read wrong addresses whenever the layouts differ.

Banked evidence (raw/trace24spec/xcheck3_snr_excerpt.txt, PRE-fix, mxfp8):

```
chunk_snr=h0K0.028V1.255;h1K0.028V4.085   (Worker_TP0)
chunk_snr=h0K0.027V1.038;h1K0.026V3.525   (Worker_TP1)
chunk_snr=h0K0.027V1.802;h1K0.027V6.411   (Worker_TP0, later row)
```

Stored-V rel-L2 **1.0–6.4** (100–640%) with stored-K clean at **0.026–0.028**
(the e4m3 floor) — every V row stored from wrong addresses. Texts corrupted
(trace24spec/probe.out): p1 degenerated into `'\n\nParis\n\nThe capital of
France is Paris...'` repetition, p3 collapsed to 9 chars.

Every earlier check missed it because they were structurally blind:
store-vs-store / decode-vs-bytes consistency checks and head-0-K-only checks
never compare stored V against the RAW source V.

## Fix

`vllm_gaudi/xpu/patches/mx_kv/triton_mx_store.py` — kernel takes
`stride_v_n, stride_v_h` separately; wrapper passes `values.stride(0),
values.stride(1)` + defensive `.contiguous()` when `stride(2) != 1`
(md5 `6c75e5b6153d86453cab58950ea6f289`, local == rig).

Regression guard: `tools/ar/p148_mxkv/store_vstride_probe.py` — v as a
transpose view (strides (256,12800,1) vs k (1024,256,1)); asserts worst
stored-V rel-L2 < 0.08. Bidirectionally proven: old kernel 1.5518 exit 1,
fixed kernel 0.0305 exit 0. Re-run inside the cert24 image: VSTRIDE_OK.

## Validation — all four postures (fixed kernel, live engine, r149 boots)

| trace | fmt / posture | p1 | p2 | p3 | chunk_snr K | chunk_snr V |
|---|---|---|---|---|---|---|
| native fp8 | eager (trace22) | 7 | 209 | 236 | — | — |
| native fp8 | spec (trace21) | 7 | 209 | 236 | — | — |
| trace25fix | mxfp8 spec | 7 | 209 | 239 | 0.026–0.028 | 0.025–0.028 |
| trace26fixeager | mxfp8 eager | 7 | 209 | 236 | (eager: xcheck silent) | — |
| trace27fp4spec | mxfp4 spec | 7 | 203 | 239 | 0.111–0.140 | 0.116–0.136 |
| trace28fp4eager | mxfp4 eager | 7 | 203 | 239 | (eager: xcheck silent) | — |

Verdicts:

1. **mxfp8 eager == native fp8 BYTE-EXACT** on all three prompts
   (7/209/236 both postures) — the 8-bit MX lane reproduces the native lane.
2. **mxfp8 spec**: p1/p2 exact, p3 differs only in a 3-char tail
   (239 vs 236) — family-normal lossy-KV tail variance.
3. **mxfp4 spec == mxfp4 eager BYTE-EXACT** (7/203/239 both) — the 4-bit
   lane is cross-posture STABLE.
4. Every lane answers p1 `'\n\nParis'` exactly (7 chars).
5. Post-fix SNR is symmetric and at the format's inherent floor: mxfp8
   K≈V≈2.7% (e4m3), mxfp4 K≈12% / V≈12% (e2m1) — V no longer divergent;
   the fix covers the mxfp4 branch too.
6. Attention path exonerated along the way (trace20 all-heads oracle,
   trace23 flash_vs_sdpa 0.0011–0.0028): the corruption was store-side
   only.

## Laws extracted

- Separate token+head strides PER INPUT TENSOR in every KV store/quant
  kernel; defensive `.contiguous()` on the D axis.
- Store validation must compare stored bytes vs the RAW SOURCE tensor,
  per kv head, for K AND V — internal-consistency checks are blind to
  wrong-content stores.
- A stable native lane is the discriminator: when a lane's texts deviate
  on every prompt, suspect content corruption, not near-tie luck.

Memory: `kv-store-separate-strides-law`. Plan bank: trace24–28 sections.
Next: cert24 image (fork_sha 40fa6830) + r154 §10 gate.
