# VERDICT — r10: UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 does NOT stop the flip — immediate command lists exonerated; trigger pinned to the first-request COMPLETION window

## The leg

arc-graph-icl0-20261028-053737, image f790fb4adee0, serve knob ARC_ICL=0
(new serve_arc_b70.sh knob; log confirms "ARC_ICL: forcing
UR_L0_USE_IMMEDIATE_COMMANDLISTS=0", passed via docker exec -e into the
vllm process env). Posture otherwise identical to r9.

## Result

- Boot clean, 84 captures, health OK.
- First /v1/completions **200 OK** (a full generation completes).
- 05:40:12 ARC_GRAPH_ZOMBIE_ONSET on the device default — in the
  first-request COMPLETION window — then every request 500s, engine death.
- **ICL is not the mechanism.**

## Accumulated eliminations (defect-b, rounds 1-10)

| round | isolation | verdict |
|---|---|---|
| r3-r6 | storm-wait/dezombie/canary recovery layers | keep integrity, can't cure |
| r7 | fresh-stream swap + permanent-flip law | flip is on-use, permanent |
| prefix-off | prefix-cache write | exonerated |
| champion control | hardware/driver/lane | lane viable; poison is fork-stack |
| r8/r9 | capture stream (side stream, all 84, eager init) | exonerated |
| r8b/r9b probes | bare capture/replay/interleave at toy scale | never reproduces |
| r10 | immediate command lists (=0) | exonerated |

## The sharp fact

Across every round the flip lands right as the FIRST generation
completes — the AsyncOutput copy path / completion cleanup, the one code
region that runs exactly once per request and never during boot warmup.
r9/r10's sample-guard variant only catches the already-poisoned state
sooner (the step after).

## r11 (next): --async-scheduling

The champion (serve_lmc_v133.sh) runs --async-scheduling; our legs never
did. It restructures exactly this region (output copy on the engine-step
thread, scheduler overlap). One host-side serve knob (ARC_ASYNC), no
rebuild. If the flip survives it, r12 = diagnostic build with phase-level
capture-state tracing + capture_begin/capture_end call logging to name
the op that starts the recording.

Raws: serve_full.log, boot.txt, matrix.log (rig leg dir
arc-graph-icl0-20261028-053737).
