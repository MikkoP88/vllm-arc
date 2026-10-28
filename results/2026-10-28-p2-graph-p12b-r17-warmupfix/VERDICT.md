# VERDICT — r17: DEFECT-b FIXED (lane W, first-request warmup interlock). Patched boot: large-as-first traffic 200 with zero onset, follow-ups at full speed. Control boot (VLLM_ARC_WARMUP_DISABLE=1): same request flips (500 + onset). Product patch: vllm_gaudi/xpu/patches/first_request_warmup.py, wired into _XPU_PATCH_MODULES.

## The patch (repo: vllm_gaudi/xpu/patches/first_request_warmup.py)

Interlocks the FIRST ``AsyncLLM.generate`` per engine: before it runs, two
internal warmup generations go through the same engine (full scheduler +
runner + xccl path — the standalone 2-proc probe proved the wedge needs
exactly that context, so the vaccine must too):

1. small stage — 640 chars ≈ 128 tok (inside r16's observed-clean region,
   AR ≈ 1.3 MB), max_tokens=2;
2. max-batch stage — prompt sized for ~2 full-size chunks of
   ``max_num_batched_tokens`` (chunk 1 = the largest possible embedding
   AR, 84 MB at 8192×5120 fp16), max_tokens=2 (also exercises decode
   graph replays).

Fail-loud law: if the warmup itself ever wedges, the failure surfaces at
first-traffic time through the existing integrity guard — never silently
inside later user traffic. Kill-switch for A/B: VLLM_ARC_WARMUP_DISABLE=1
(r17b uses it as the control).

## Evidence (arc-test container, r13/r16 posture verbatim, TP2, graphs ON)

r17a (patch ON, installed via PYTHONPATH sitecustomize — no image rebuild;
product wiring is the platform _XPU_PATCH_MODULES entry):
- interlock fired at first generate (12:19:56): "running defect-b warmup
  interlock first" → "stage small done in 0.5s" → "stage max-batch done
  in 70.4s (442368 prompt chars)" → "warmup complete in 70.9s — engine
  hot". (70 s because the first cut overshot the prompt 12×; the committed
  module now sizes ~2 chunks ≈ 10 s.)
- the triggering LARGE first request (832 tok, the exact payload that
  flipped 3/3 prior boots): **HTTP 200** (72 s total = 71 s warmup +
  ~1.3 s request), ZERO ARC_GRAPH_ZOMBIE_ONSET.
- follow-ups: small 200 in 0.90 s, large 200 in 1.03 s — engine hot.

r17b (control, VLLM_ARC_WARMUP_DISABLE=1, everything else identical):
- interlock correctly absent (install count 0 — kill-switch works);
- the SAME large-first request: **HTTP 500 in 3.8 s + ARC_GRAPH_ZOMBIE_ONSET
  (1)** — the flip, exactly as in r16.

Together with r16: mechanism identified (first eager large TP AR after
decode-graph capture warmup wedges the default stream) → fix validated
with its own counterfactual. Defect-b: CLOSED at the posture level.

## What is NOT yet claimed

- §10 P2 gates (graphs+MTP spec leg, TPOT c8 ≤ 60 ms, acceptance ≥ 0.85,
  warm-hit ≤ 0.75 s) — not run on the patched boot yet; the fix re-validates
  there (MTP posture will exercise the warmup with spec enabled).
- The warmup adds ~10 s (trimmed) to first traffic on every boot; the
  certified-image acceptance bench must absorb that (warm-hit gate measures
  post-warm behavior).
- Upstream issue text (model v3 + warmup workaround) awaits owner sign-off.

## Artifacts

- first_request_warmup.py — the product patch (this dir = banked copy;
  the repo copy under vllm_gaudi/xpu/patches/ is the source of truth)
- r17_sitecustomize.py — PYTHONPATH installer for image-rebuild-free legs
- r17_serve.sh — r13/r16 posture serve verbatim + interlock PYTHONPATH
- r17b serve = same + VLLM_ARC_WARMUP_DISABLE=1 (created on rig)
