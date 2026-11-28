# VERDICT — r268b request-bisection leg (the same-boot discriminator)

Banked 2026-11-28 from rig bank `/root/build/p647/r268b_bisect` after
`R268B_DONE rc=0 09:48:39Z` + `W14_RESTORE rc=0 09:53:13Z` (transfer
oracle 8/8 byte-match; address audit clean after the byte-exact perl
redaction of serve.log's 29 host literals). ONE plain-image c3-posture
boot (`vllm-arc:v0.30.0-b70-cert29`, gemma lawful g3 skin,
`kv_cache_dtype=mxfp8` + native windowed lane, `ARC_GRAPH=0`, NO
overlay — the r268 overlay was proven behaviorally inert by diff), then
TWO workloads on the SAME boot: the 33-step request-bisection ladder
(`p268b_ladder.py`, md5 `88345633`) and the FULL p266 matrix probe with
the exact e2 PARGS (`--workers 8 --cells chat,raw,nll,gate --chat-cells
t0,card`, wall 864 s, oracle True, ptok_mismatch 0).

## Result 1 — the ladder: 14/33 RED, `FIRST_RED=A_len_reps12`

| step | shape | result |
|---|---|---|
| A1 control (ptok 6) | raw, temp=0, mt=16, logprobs | OK — " a city of romance, art, and…" |
| A_len reps 1-10 (ptok 12-102) | raw, sequential | ALL OK (finite logprobs 16/16, sensible text) |
| A_len reps12 (ptok 122) | raw, sequential | **RED** (HTTP 400) — the only plain-sequential red |
| B1 gate-battery chat (ptok 40) | chat, temp=0, mt=64 | HTTP 200 **EMPTY text** |
| B2 card chat full params | chat, temp=1.0+card set, mt=96 | **RED** (HTTP 400) |
| C1 two-way raw (r0 ptok≈37, r1≈47) | raw, 2 concurrent | **BOTH RED** |
| C2 eight-way raw r0/r1 | raw, 8 concurrent | **RED, RED** |
| C2 eight-way raw r2-r7 (ptok 57-107) | raw, 8 concurrent | ALL OK |
| C3 eight-way chat c0-c4, c6 | chat, 8 concurrent | **RED** (400-nan) |
| C3 eight-way chat c5, c7 | chat, 8 concurrent | HTTP 200 **EMPTY** |
| D control repeat (after 14 reds) | raw, sequential | OK — no boot poisoning |
| E redrepeat ×2 (A_r12 shape) | raw, sequential | **RED** both — deterministic |

## Result 2 — the same-boot probe: 323/324 RED (`PROBE_RED`)

- **chat**: 107/108 HTTP-400 nan (`Out of range float values are not
  JSON compliant: nan`); t0 (temp=0) 54/54 RED, card 53/54 RED.
- **raw**: **216/216 RED** — every format (g3/g4/g4think) × every cell
  (t0/t03/t07/card/t1only) including temp=0 and the **non-crossing g4
  short prompts** (health ntok=20: 20+96=116 < 128).
- **gate** (the e2 gate battery verbatim: no logprobs, temp=0): HTTP 200
  with **'Paris our our our…' — the e2 signature reproduced EXACTLY**.
- The ONE clean chat: a card cell on the `gases` pid (ptok=37) — 67
  sensible tokens, `finish=stop` (total 104 < 128), among 53 identical
  RED siblings.

## The discriminator analysis

Every simple single-axis law is eliminated:

1. **NOT sampling** — the t0 (temp=0) raw cells are 27/27 RED per
   format; the ladder's temp=0 reds (A_r12, C1, C2-r0/r1) confirm.
2. **NOT prompt length / the 128-page crossing alone** — g4 short
   templated raws (total 116 < 128) are RED while plain raws to ptok
   102 (total 118) are clean; and C2's 57-107-token plain raws are
   clean under the same 8-way burst whose 37/47-token pair is RED.
3. **NOT concurrency alone** — C2 r2-r7 are clean in an 8-way burst.
4. **NOT the chat endpoint alone** — A_r12 and the C1/C2 first-admitted
   pair are plain raw completions.
5. **NOT skin/posture, graph, eager, or kernel math** (r266 batch-1/2,
   e1/e2 pair, r267 canary).

What the RED set shares — the closest characterization the data
supports:

- **Every templated request is in the broken class**: all probe chats
  (server-side g3 render) AND all probe raws (client-side pre-rendered
  template strings with `add_special_tokens=False`) at every
  temperature, format, and length. On this boot the broken class
  manifests as 400-nan with logprobs on, 'Paris our our' without
  (the gate battery), and EMPTY (instant stop-token under garbage
  logits) for a minority — B1/c5/c7.
- **Plain raw requests are clean EXCEPT two narrow conditions**:
  A_r12 — the only ladder request whose decode grows the block table
  into a 3rd page (122+16=138 > 128), deterministic on re-run — and
  the first-admitted pair of a concurrent burst (C1's both; C2's
  r0/r1) while the 6-way later admissions stay clean.
- **The outcome is runtime-state-dependent, not deterministic per
  prompt**: one card chat (identical shape to 53 RED siblings) ran 67
  clean tokens. Under the r267 census (never-written row → non-finite
  → 400-nan; stale-written row → finite-wrong → 'our'; valid row →
  clean), this is exactly the signature of **wrong-ADDRESS reads whose
  content varies with pool state** — G3 metadata VALUES, the surviving
  hypothesis after r267 exonerated the math and r268 exonerated the
  boot.

## Consequence

**r268c (launched 10:07:03Z)** boots the same posture with the
`r268c_overlay` (engine.py `_TRACE` **hardcoded True** — the fix for
the r268 dead letter: `serve_arc_b70.sh`'s curated export list never
passes `ARC_MXKV_TRACE` into the container; overlay md5 `76e3bb12`,
one-line diff vs the r268 overlay `0ce0954c`) and fires a sequential
minimal discriminator set (`p268c_disc.py`, md5 `64726f9d`): T0 the
clean control / T1 the A_r12 page-growth red / T2 the gate chat ('our'
class) / T3 the card chat (400 class) / T4+T5 the probe's own
byte-exact templated raws (non-crossing g4 short + crossing g3) / T6
the C1 2-way burst. The overlay prints the store's slot ids (smap)
and the decode's block-table rows/stride/seq_lens (decx) per call —
T0 is the correct-addressing reference; any divergence on a RED
request names the wrong metadata value directly. Bank:
`/root/build/p647/r268c_trace/{out,logs}`.
