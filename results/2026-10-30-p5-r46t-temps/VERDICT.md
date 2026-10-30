# r46t — real-world temperature / MTP-depth campaign (P5 close-out)

**VERDICT: GREEN — recommendation: MTP k3 (`num_speculative_tokens: 3`).**
k5 dropped from recommendation and testing per owner directive; all k5
testing slots replaced by k3. Full evidence in `leg/` (this directory).

## Owner directives executed (in order)

1. Real-world temperature sweep 0.2/0.4/0.6/0.8/1.0 (0.8 added mid-run).
2. All testing configurations carry the real-world sampling set
   (top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0,
   repetition_penalty 1.0).
3. Garbage/corruption hardening suite (15 probes) so corrupted inputs
   can't corrupt the engine or its outputs.
4. All performance testing runs **xhigh thinking** (the chat template's
   own `reasoning_effort: xhigh` knob, `enable_thinking: true`).
5. Non-MTP (no `--speculative-config`) baseline at temperature 1.0.
6. Alt-sampling MTP + non-MTP legs, non-thinking, at temperature 0.7 /
   top_p 0.80 / top_k 20 / min_p 0.0 / presence_penalty 1.5 /
   repetition_penalty 1.0.
7. k3 added to all MTP performance tests; then **k5 dropped fully** —
   recommendation and testing — k5 slots replaced by k3 (executed live:
   the rig watcher killed the alt-sampling chain after k4's leg and
   before k5's; k5 never booted on the alt face).

## Faces (run7n battery: agg4x256 ×2 + solo200 ×2; 2-run means)

### A. Real-world sweep — xhigh thinking + real-world set (v3/)

agg tok/s (mean of 2) / solo tok/s (mean of 2):

| temp | k3 agg | k4 agg | k5 agg (DROPPED) | k3 solo | k4 solo | k5 solo (DROPPED) |
|---|---|---|---|---|---|---|
| 0.2 | 174.9 * | 201.7 | 187.1 | 63.9 | 64.6 | 63.9 |
| 0.4 | 208.8 | 184.4 | 200.1 | 63.0 | 60.4 | 61.2 |
| 0.6 | 202.8 | 193.6 | 189.0 | 61.3 | 62.3 | 60.2 |
| 0.8 | 195.8 | 184.9 | 182.6 | 61.4 | 59.1 | 59.9 |
| 1.0 | 186.2 | 182.6 | 170.0 | 61.1 | 54.3 | 53.1 |
| **mean** | **193.7** | 189.4 | 185.8 | **62.1** | 60.2 | 59.7 |

\* k3 t0.2 run#1 (137.2) is first-battery-after-boot warmup cost (same
signature as k3's alt and t0 legs); run#2 = 212.6. Steady-state k3
run#2 mean across temps = 205.4 agg (+5.8 % over k4's 194.2).

Acceptance (post−pre `vllm:spec_decode` counters, per temp):
k3 57.3/57.1/53.1/52.3/49.5 %, k4 50.2/45.9/46.6/43.5/43.2 %,
k5 42.2/43.5/40.6/39.7/33.5 % — acceptance falls with temperature and
falls faster with deeper drafts: the measured mechanism behind the
throughput inversion (k5 won greedy 89.2 % but its depth wastes at
real-world acceptance).

### B. Non-MTP baseline (nospec) — same faces

- t1.0 xhigh thinking: agg 113.3/124.3 (mean 118.8), solo 33.5.
  **MTP value at the flagship real-world face: k3 1.57×, k4 1.54×
  aggregate over no-spec.**
- alt set (t0.7/tp0.80/pp1.5, non-thinking): agg 72.4/124.2 (steady
  124.2), solo 33.4. **MTP value: k3 2.05×, k4 2.03× steady aggregate.**

### C. Alt-sampling legs — non-thinking, t0.7/top_p0.80/pp1.5 (v5/)

| leg | agg #1 / #2 | solo #1 / #2 | acceptance |
|---|---|---|---|
| k3 | 170.6 * / 254.9 | 76.9 / 92.2 | 88.5 % |
| k4 | 236.9 / 252.6 | 85.3 / 104.3 | 82.6 % |
| nospec | 72.4 * / 124.2 | 33.2 / 33.6 | — |

\* first-battery warmup. Near-deterministic continuation (non-thinking
counting) keeps acceptance high — the one regime where k4's deeper
draft pays (steady solo 104.3 vs 92.2). Steady agg is a tie (254.9 vs
252.6, +0.9 % k3).

### D. k3 certified temp-0 face (k5's greedy-battery slot → k3, v3/k3_t0/)

agg 93.7 (warmup) / **299.9** steady — vs k4's certified 297.7/297.9
(tie, +0.7 %); solo 84.5/84.5 vs k4's 85.0–85.2 (−0.7 %). Acceptance
85.6 % greedy (k5 was 89.2 %). k3 costs ~nothing on the certified face.

## Recommendation (data-driven, per owner directive)

**k3** — best overall across the real-world conditions the owner
defined as the target (xhigh-thinking sampled serving): +5.8 % steady
agg and +3.8 % steady solo over k4 on the 5-temp xhigh sweep; ties k4
on the certified greedy face (±0.7 %); only loses the near-deterministic
non-thinking alt solo face (where acceptance ≥82 % rewards depth).
k4 = documented alternative for deterministic/greedy batch workloads.
k5 = dropped (superseded evidence retained in `leg/v3/k5/`). Spec depth
is serve-time CLI (`--speculative-config`), so the flip rebuilds no
image.

## Garbage/corruption hardening — GREEN

- Probe v1 ran on the v4 legs (k4/k5/nospec xhigh boots): recorded 6
  content-shape failures — **post-capture root-caused as a probe
  defect, not an engine defect**: the template defaults thinking ON
  (short-max_tokens probes returned `content=None` with tokens in
  reasoning), and `huge_blob`'s 400 was correct oversized-payload
  rejection. Health stayed 200 after all 15 probes and the temp-0
  sanity pair byte-matched (`a817337c…`) on every boot — no engine
  state corruption anywhere. v1 outputs preserved unmodified in
  `leg/v3/{k4,k5}/garbage.out`, `leg/v3/nospec…` (superseded).
- Probe v2 (thinking-off pinned; content check only at status 200):
  **5/5 suites PASS 15/15** — v4b k3, v5 k3, v5 k4, v5b nospec, v5b
  k3-t0 (`leg/v3/k3/garbage.out`, `leg/v5/*/garbage.out`,
  `leg/v3/k3_t0/garbage.out`).

## Tools (provenance)

`tools/bench_run7n.py` @ 89c78330 (flags: `--temperature`,
`--thinking xhigh`, `--sampling alt`), `tools/arc_garbage_probe.py` v2
@ 62e1511c; drivers `.tmp/r46t_campaign{,2,3,4}.sh` + watcher
(`.tmp/r46t_watcher.sh`, cut the chain pre-k5-alt at 16:09:49 rig).
Rig staging `/root/build/r46t/` (bench_run7n_t.py v4 bytes for legs A;
bench_run7n_t5.py v5 bytes for legs B–D; probe v2 as
arc_garbage_probe2.py — running-campaign files never swapped mid-run).
All boots from the certified line (`r43_serve_k5.sh`, spec depth sed'd;
nospec = line minus `--speculative-config`), image
`vllm-arc:v0.30.0-b70-p46`, W14 trap-restored after every driver.
