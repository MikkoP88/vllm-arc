# VERDICT — r12c: flip pinpointed to the extern-aten-mm window between triton kernels inside the piecewise-compiled prefill callable — defect-b mechanism COMPLETE; fix = compilation mode NONE with graphs ON

## The leg

arc-graph-trace3-20261028-071549, image c966fcf89a3f (fork cbb6937e —
r12b's aten.mm.out wrap dropped, triton-only tracer: capture-state probe
before/after every `CachingAutotuner.run`). Same posture as r9-r12
(GRAPH=1 SPEC=0, 84 captures, boot clean, engine healthy 07:24:39).

## The pinpoint (serve_full.log)

- Log line 293: first `POST /v1/completions` **200 OK** — the flip poison
  did not stop the in-flight generation (integrity law held).
- Log line 294, 07:25:00, both workers: first-ever probe hit is
  **`FLIP at triton-pre`** — the probe AFTER the previous triton kernel
  was clean, the probe BEFORE the next triton kernel reads recording.
  **The stream flipped between two triton kernel launches.**
- Between two triton kernels inside the piecewise-compiled prefill
  callable there is exactly one class of work: the **extern aten mm**
  (`extern_kernels.mm(buf1, reinterpret_tensor(...), out=buf12)`) — the
  frame r12's onset stack dump already named (piecewise_backend.py:380 →
  standalone_compile.py:321 → output_code.py:763 → extern_kernels.mm).
- 838 FLIP lines follow (the flip is permanent — r7 law), all subsequent
  requests 500, engine dead; every matrix cell connection-refused.

## Mechanism (defect-b, closed)

vllm 0.30 on XPU compiles prefill piecewise (inductor backend,
triton-xpu kernels + extern aten mm) while `CUDAGraphMode.
FULL_DECODE_ONLY=(FULL, NONE)` captures decode FULL and leaves prefill
with NO piecewise graphs — so the compiled prefill artifacts launch LOOSE
on the device-default stream with live capture machinery in the process.
Something in that loose extern-mm launch (inductor-XPU queue/queue-policy
selection inside torch) binds the serving stream into a recording state;
`is_current_stream_capturing()` returns True forever after and the next
real capture-shaped allocation/use dies.

Two independent control groups never run this combination (plan
Appendix C): W14 incumbent = FULL_AND_PIECEWISE with capture sizes
[4,8,16,32,64] — pieces captured, never loose — clean, TPOT c8 33 ms;
0.26 D champion = no compile at all (eager prefill) + FULL graphs —
clean, agg 274.41. Cofactor leg: compiled prefill with cudagraph_mode
NONE also clean — compiled prefill ALONE is harmless; the flip needs
compiled-prefill-loose × live-capture-machinery.

## Ranked fix (owner-approved, plan Appendix C)

1. **r13**: `--compilation-config {"mode":"none"}` + graphs ON
   (FULL_DECODE_ONLY, capture sizes unchanged) — CompilationMode.NONE is
   first-class upstream ("fully eager pytorch mode"); piecewise backend
   never constructed; this IS the champion posture on 0.30. Config knob
   authoritative (env TORCH_COMPILE_DISABLE=1 stays a P4 A/B cell only).
2. Productize: `ArcXpuPlatform.check_and_update_config` forces
   `compilation_config.mode = NONE` whenever `cudagraph_mode != NONE`
   (env override for A/B) — lands under p04, cites r12/r12c.
3. W14-style FULL_AND_PIECEWISE stays a P2+ experiment, blocked until
   defect-a (speculator) is fixed; P4 prefill-parity bars decide.

NOT pursued (owner directive): fixing inductor-XPU queue selection inside
torch — banked as falsified/deferred; upstream issue (naming the
extern-mm window under FULL_DECODE_ONLY) only after owner sign-off.

## Gate for r13 (owner: one clean leg is NOT the gate)

Re-run the exact kill sequence several times + soak shape: serialized
bench, burst ×3, zero dmesg engine resets, per plan §10/leg discipline.

Raws: serve_full.log (838 FLIP lines, first at 294), boot.txt, gates.txt,
k1k_sanity.txt, spec_gate.txt, matrix.log (rig leg dir
arc-graph-trace3-20261028-071549).
