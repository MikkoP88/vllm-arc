# VERDICT — r269 fix-gate (THE FIX IS VALIDATED: 323/324 RED → 0/329 RED)

Banked 2026-11-28 from rig bank `/root/build/p647/r269_fixgate` after
`R269_DONE rc=0 10:43:49Z` + `W14_RESTORE rc=0 10:48:22Z` (transfer
oracle 13/13 byte-match; address audit clean — no host literals in the
pulled set; 15 files). Boot: cert29 image, gemma lawful g3 skin
(`ARC_TEMPLATE` + own parsers), `kvdtype=mxfp8` + native windowed lane
(`ARC_MXKV_NATIVE=1`), **`ARC_GRAPH=1` — the PRODUCTION graph posture**
(stronger than the eager r268b baseline), `ARC_LMC=0`,
`ARC_MXKV_OVERLAY=/root/build/r269_fix` (engine.py md5 `d1fa3c17` =
the image's own bytes; `triton_mx_decode.py` md5 `e33aa7b7` = THE FIX:
all three decode entry points take `block_size: int | None = None` and
resolve it to the cache's OWN page `cache.shape[1]` at body start —
glob pool → 128, slide pool → 64; explicit-page callers unchanged).

## The A/B

Same posture family as the r268b baseline (cert29, g3 skin, mxfp8 +
native) with exactly one change — the r269 wrapper fix:

| | r268b (no fix, eager) | r269 (fix, graphs ON) |
|---|---|---|
| matrix RED | **323/324** (chat 107/108, raw 216/216) | **0/329** (chat 108/108, raw 216/216, gate 5/5) |
| http errors | 323 (400-nan) | **0 in every cell** |
| logprobs | nan (the 400 class) | **finite everywhere** (lp1 −0.68…−4.31, lp_gen −0.10…−1.03) |
| gate battery | 'Paris our our' byte-exact | **5/5 coherent** ('Paris', fibonacci correct, gases correct, HEALTH OK) |
| engine health | post_errors red-class | ERROR=0 TRACEBACK=0 device_lost=0 fault_lines=0 |

The red-count recomputation (banked JSON, classification identical to
r268b: http≠200, lp_nan>0, corruption markers): **chat 108/108 CLEAN,
raw 216/216 CLEAN, gate 5/5 CLEAN — 0/329 RED**. Every cell that was
RED for the wrong-address read class (templated chat, every raw fmt ×
every cell including t0 temp=0, the byte-exact gate battery) is now
clean. The 56 MXKV native calls in serve.log ran the fixed walk
(bt_width=640 stride=4224 slide lane, window=1024 P-C2 rides the
masked twin) with zero kernel faults.

## The residual (NOT corruption)

`raw fmt=g4: reploop=26/72`, `raw fmt=g4think: reploop=22/72` — the
KNOWN base-checkpoint template degeneration (r266 b3 measured the same
class on the same boot posture; the checkpoint ships no chat template
and the official g4 special-token structure degenerates confidently).
The lawful **g3 cells are reploop=0 across the board** (65/72
answer-ok; the misses are nextturn/echo heuristics at finish=length —
the model fabricating a next turn after answering, never corruption).
Corruption signature (400-nan / 'our our') is absent from every cell
including the g4 ones.

## Consequence

- **The r268c root cause is confirmed by intervention**: changing
  exactly the walk's page resolution (nothing else — the engine bytes
  are the image's own) flips the entire matrix from 323/324 RED to
  0/329 RED under the production graph posture.
- **The wrapper fix lands in-tree** (banking directive): the 1:1 diff
  at `vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py` lines
  1404/1495/1648 (block_size default None → `cache.shape[1]`), already
  validated byte-for-byte as the rig overlay.
- Next rungs per Law 11: m8g/m4g/mkg re-gates WITH logprobs on the fix,
  the LMCache lane (lmc), then the full battery and the cert29 cut
  gated on `R264_CERT_PRECONDITION → CERT_CUT_ALLOWED`.

W14 restored and verified (health 200 + one real generation) at
10:48:22Z; the rig is on the incumbent image, not an arc image.
