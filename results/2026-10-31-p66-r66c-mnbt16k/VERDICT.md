# P6.6 r66c — MNBT 16384 A/B (improvement leg 1b, knob rung)

**Date**: 2026-10-31 · **Phase**: P6.6 (plan §12 item 8, [viii] lane)
**Posture**: `vllm-arc:v0.30.0-b70-cert4` (stock, NO probe image), fresh
boot, `ARC_LMC=0`, graph law (`ARC_GRAPH=1`), MTP k3, greedy, solo faces
U8k/U32k/U64k (unique seeds, same probe as r66a gate1). Client-side TTFT
is the metric — no instrumentation needed.

## What ran

- `p66_gate3.sh` — both boots (mnbt 16384 then a same-session mnbt 8192
  A/B refresher). Pass-1 defect (banked honestly): the 16k boot's face run
  crashed with `KeyError: 'U8k U32k U64k'` — `P66_FACES` was
  space-separated but `p66_solo_ttft.py` splits on `,`. The running script
  picked up the comma-fixed bytes before its second boot (same byte
  count), so the **8k reference faces are valid pass-1 output** and match
  r66a exactly (below).
- `p66_gate3b.sh` — lean re-run of the 16k boot + faces (the part pass 1
  lost). `grep "mnbt=16384"` launch assertion + boot log banked
  (`boot_mnbt16k.out`, `ARC_SERVE_LAUNCHED ... mnbt=16384`).

## The A/B (solo TTFT, thr=none both arms)

| Face | prompt tok | MNBT 8192 | MNBT 16384 | Δ TTFT | rate 8k → 16k |
|---|---|---|---|---|---|
| U8k | 8158 | 3.74 s (2181 t/s) | 4.11 s (1984 t/s) | **+9.9%** | −9% |
| U32k | 32553 | 17.33 s (1878 t/s) | 15.65 s (2080 t/s) | **−9.7%** | +11% |
| U64k | 65303 | 39.19 s (1666 t/s) | 37.60 s (1737 t/s) | **−4.1%** | +4% |

8k arm vs r66a gate1 (3.74 / 17.34 / 39.20): identical within ±0.06 s —
comparator law satisfied (same fresh-boot posture, no drift).

## The cost side (authoritative init lines, both serve logs)

| | MNBT 8192 | MNBT 16384 |
|---|---|---|
| XPU KV cache size (kv_cache_utils.py:2395) | 405,491 tokens | 332,629 tokens (**−18%**) |
| Maximum concurrency (262,144 tok/req) | 1.55x | 1.27x |

Doubling the chunk budget doubles peak prefill activation workspace →
memory profiling reserves more → smaller KV pool. Periodic usage samples
agree in direction (12.9% vs 9.0% at the same face point in the two logs).

## Verdict

**Documented knob, NOT adopted.** MNBT stays 8192 (the certified default).

- The r66b small-M theory is **confirmed at long context**: bigger chunks
  lift the device per-token rate (2080 vs 1878 t/s at U32k) — same
  mechanism r66b measured (queue-drain device time dominating host glue).
- But the trade is three-sided and negative on balance for the fleet
  posture: short-face TTFT +10%, KV pool −18% / concurrency 1.55x → 1.27x
  (the p65 long-context profile's lifeblood — `LMC_DEFER_MIN_TOKENS=40960`
  headroom), long-face gains only −4…−10%.
- **Never combine with `--long-prefill-token-threshold 2048`**: the
  threshold pins chunk size (r66a: thr2048 ⇒ 1664-token chunks regardless
  of MNBT), so MNBT 16384 adds nothing to TTFT there and the −18% KV pool
  is a pure loss. The knob only makes sense stand-alone (thr=none) on a
  long-prompt-only, low-concurrency workload.

## Open mechanism note (untested, flagged honestly)

The U8k +0.37 s regression is face-level real (the 8k arm reproduced
r66a's 3.74 s exactly in the same session), but the mechanism is not
pinned: 8158 tokens run as ONE step under 16k vs 7488+670 under 8k, so
boundary count FELL — candidates are first-touch of the doubled
activation workspace on the first big step, the MTP draft prefill running
on a full-prompt final chunk, or eager-attention workspace effects at
larger single-chunk M. Not chased further: the adoption decision does not
turn on it.

## Artifacts

- `p66_mnbt16k.{txt,json}` + `serve_mnbt16k.log` + `boot_mnbt16k.out` —
  the 16k arm (gate3b)
- `p66_mnbt8k_ab.{txt,json}` + `serve_mnbt8k_ab.log` — the 8k A/B arm
  (gate3 pass 1, second boot)
- `p66_gate3.sh` / `p66_gate3b.sh` + `gate3.log` / `gate3b.log` — legs
  exactly as run, including the pass-1 separator defect
- rig: `/root/build/p66/` (same names)
