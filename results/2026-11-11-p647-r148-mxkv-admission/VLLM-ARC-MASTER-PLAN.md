SPEC+GRAPH BOOT (r149, 2026-11-11): mxfp8 + MTP k3 + FULL_DECODE_ONLY
XPU-graph boots and serves — READY ~353 s (capture 2x54 graphs), 3/3
PROMPT_OK, block 832, KV pool 290,741 tokens / 8.87x @32K (graph+spec
reserve shrinks the pool vs eager 429,545 — expected). P619 instrument
proves the tq_spec_fix verify-capture path rides the MX impl during
capture (mql=4 msl=5 split-row shapes). QUALITY FLAG: temp-0 battery
text shows degradation fragments (prompt-3 "The2.23 | **Identify**...")
— differential required vs fp8-KV same posture before judging: MTP
probabilistic draft accept can legitimately perturb greedy output, so
the discriminator is the certified dtype under identical spec+graph
(FMT=fp8 SPEC=3 GRAPH=1) — leg-C suite-13/14 metrics decide finally.
CORRUPTION DIFFERENTIAL (r149, 2026-11-11): the quality flag is REAL
and MX-specific. Identical posture on the CERTIFIED dtype
(FMT=fp8 SPEC=3 GRAPH=1): all 3 prompts CLEAN ("Paris"; coherent
gases answer "1. **Identify the criterion** – ... 20 °C / 68").
mxfp8 same posture: prompt-1 repetition + broken clause ("The capital
of France is and the capital of Texas"), prompt-3 garbage ("The2.23 |
**Identify**"). Eager MX boots (SPEC=0 GRAPH=0) were coherent BOTH
formats -> the corruption enters through the SPEC-VERIFY path
(MTP k3 verify = q_len 1<k<=8 -> split-row mx_decode with per-row
causal limit off live seq_lens) or GRAPH REPLAY of MX decode.
ISOLATOR NEXT: mxfp8 SPEC=3 GRAPH=0 — clean => graph replay corrupts;
degraded => spec-verify split-row path corrupts. (fp8+spec+graph boot
also banks the reference KV pool: 299,678 tokens / 9.15x @32K vs MX
290,741 / 8.87x — same-posture capacity deltas now measurable.)
