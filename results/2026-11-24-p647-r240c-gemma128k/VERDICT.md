# VERDICT — r240c: the gemma 128K annex on cert28 — W4A16 serves what native MX-KV cannot

**What**: the deep-sweep campaign leg 3 — the 8-cell gemma 128K annex
on `vllm-arc:v0.30.0-b70-cert28` (fork_sha `43f84cf9`; identity gate
GREEN at launch). Gate line `R240C_COMPLETE 2026-11-25T12:21:43Z`.
As with r240b, the gate's `CONC_OK` lines are re-audited against the
banked JSONs (the same existence-only gate): **30 of 78 banked faces
are real measurements**; the 48 others are the gemma MX-KV
connection-refused class.

## The headline: gm4x — W4A16 weights + fp8_e4m3 KV at maxlen 131072 — serves the full 128K posture (26/26 real)

The only posture that boots AND serves gemma at 131072:

| rung | n1 decode (tok/s) | TTFT n1 (s) | t1.0 n1 (tok/s) | conc rungs (agg, tok/s) |
|---|---|---|---|---|
| 1K | 40.1 | 0.7 | 38.5 | n1–n8: 38 / 77 / 110 / 147 / 227 |
| 8K | 37.3 | 3.7 | 36.5 | n1–n8: 30 / 50 / 63 / 75 / 91 |
| 32K | 35.0 | 20.9 | 33.6 | n1–n8: 16 / 20 / 22 / 23 / 25 |
| 64K | 32.3 | 57.2 | 30.9 | n1–n4: 8 / 9 / 9 / 10 (n8 = KV_CAP) |
| 128K | **28.3** | **175.6** | 26.3 | n1–n2: 3 / 3 (n3/n4/n8 = KV_CAP) |

KV_CAP rows banked in the cell `conc.log`: 64K n8 (POOL 288,972 <
NEED 529,600), 128K n3/n4/n8 (395,208 / 526,944 / 1,053,888). The
pool 288,972 is 5.15× the certified maxlen-40960 fp8 pool (56,128)
and 3.58× the 64K-closure fp8 pool (80,619) — the W4A16 weight
savings + the maxlen raise buy the 128K posture in one posture.

Consistency: the 1K/8K/32K rungs match the r240b `gm4` cell exactly
(40.1 / 37.3 / 35.0 — same posture at maxlen 40960), so the annex
adds the 64K/128K rungs on a proven ladder.

## The native MX-KV annex cells: booted, zero served (hd512 class)

Same first-decode refusal as their r240b counterparts — gemma's
head_size-512 full-attention layers fall outside the arc7 compiled
`mx_paged_decode` instantiations (D=128/256 only):

| cell | faces served | pool banked (tokens) | class |
|---|---|---|---|
| gkv_m4x (mxfp4-KV @131072) | 0/24 | 238,497 | hd512 first-decode refusal |
| gkv_mkx (mxk8v4-KV @131072) | 0/22 | 162,180 | hd512 first-decode refusal |
| gkv_m8x64 (mxfp8-KV @65536) | 0/2 | 78,175 | hd512 first-decode refusal |

The packed-KV capacity thesis is REAL — the pools prove it (mxfp4-KV
at 131072 banks 238,497 tokens where fp8-KV cannot boot at all) — but
on gemma it is reachable only after an hd512 kernel instantiation
lands (the reopened channel; plan entry). Until then the 128K story
rides the weights lane (gm4x above).

## By-design refusal banks

- `gm8x` (mxfp8 weights @131072): BOOT_FAIL — 5.83 GiB KV needed >
  5.3 available (est-max 103296). 8-bit weights leave too little for
  the 131072 posture.
- `gkv_m8x131` (mxfp8-KV @131072): BOOT_FAIL — 6.01 GiB > 5.64
  (est-max 111744). mxfp8-KV is fp8 bytes — same boundary the weights
  side banks; no KV-side byte advantage on the 8-bit lane.

## The 64K closure (2/2 real each)

One 64K session (65,700 tokens) fits every fp8-class pool at
`ARC_MAXLEN=65536`: gf8x64 24.1 tok/s n1 (TTFT 56.0 s; t1.0 23.8),
gm8x64 23.7 (56.4; t1.0 23.4) — the weights quant does not change
the 64K rung (fp8-class KV pools dominate); n2+ rows bank as KV_CAP.

## Hygiene

W14 restored by the EXIT trap (fresh launch — the recovery path for
the r240b-restore hang class); health-200 verified at close-out. Host
addresses byte-redacted post-pull; SHA256SUMS data-files-only.

**Outcome**: the annex proves the gemma 128K posture SERVES via the
W4A16 weights lane (28.3 tok/s single-stream decode at c131072, TTFT
175.6 s — the record faces for the gemma capacity story), banks the
native packed-KV pools as real capacity facts behind a recorded
hd512 decode refusal, and closes the 8-bit lanes' ladder at 64K.
Reports: `docs/reports/` (the r240 close-out).
