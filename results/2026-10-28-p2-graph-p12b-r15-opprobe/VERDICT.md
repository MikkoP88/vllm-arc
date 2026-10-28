# VERDICT — r15 standalone op probe: INCONCLUSIVE (synthetic metadata suspect). Decode-shaped call ran CLEAN; prefill-shaped call DEVICE_LOST/OUT_OF_RESOURCES with AND without graph machinery — the prefill crash is either a real TP2-geometry kernel bug or wrong synthetic kwargs; deciding requires r16 real-kwargs capture

## The probe

flip_probe_gdn.py (banked here; rig /root/build/flip_probe_gdn.py) —
standalone container probe, NO serve, NO monkeypatching (r12b law):
S0 stream-state baseline → S1 side-stream XPUGraph capture+replay of a
dummy gemm (PROBE_NOGRAPH=1 skips) → S2 the two GDN projection gemms →
S5 decode-shaped `torch.ops._xpu_C.gdn_attention` call → S4 prefill-shaped
call (1024 tokens, NK=16/NV=48 FULL counts with tp_size=2, conv_dim
(2048*2/2)+(6144/2), conv_state (3, conv_dim) SD layout, schema-exact args
verified against the op's registered schema).

## Observed (docker exec, arc-test container, 2026-10-28 ~09:5x; stdout not
persisted to a file — verdict reconstructed from session record)

- S0/S1/S1b/S2: clean (capture+replay does not flip the stream by itself).
- **S5 decode-shaped call: CLEAN** (returns, stream clean) — with graph
  machinery live.
- **S4 prefill-shaped call: DEVICE_LOST / OUT_OF_RESOURCES — with AND
  without PROBE_NOGRAPH.** The failure does not need graphs in-process.

## Why parked, not concluded

Real serve legs prove this kernel runs real 1024-token prefills clean at
first request (the flip is detected AFTER the request completes) — so a
standalone unconditional crash would contradict serve behavior unless the
poison is silent there and only crashes under probe memory layout. The
likely alternative: synthetic metadata wrong (num_prefills/qsl/conv_state
layout/A_log-dtype family). Probe stdout was not redirected to a file, so
the exact error strings are not banked — rerun with logging before citing.

## What it still contributes

- Decode-shaped path is not the poison (matches r13 onset = prefill).
- Graph machinery alone (capture+replay of a plain gemm) does not flip the
  stream — the cofactor needs the GDN prefill path specifically.
- IF r16 (capture REAL serve kwargs → standalone replay) reproduces
  DEVICE_LOST standalone, then the probe's crash was real and the defect
  is a kernel bug on TP2 geometry (24 v-heads/rank) — the arcitech
  "cookbook partial-final-group fix" class (their clean B70 run is
  single-GPU; TP2 halves head counts and changes the sub-group arithmetic
  exactly the way #49924's v_head_id bug needed).

Next: r16 — sitecustomize kwargs-capture hook (PYTHONPATH-injected, no
image rebuild; wrap the MODEL forward_xpu class attr, never torch.ops —
r12b law), first request dumps exact tensors, replay standalone ± graphs.

Raws: flip_probe_gdn.py (this dir + rig /root/build/flip_probe_gdn.py).
