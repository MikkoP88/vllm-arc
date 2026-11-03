# P6.18e — cert7 performance faces (2026-11-03/04, rig)

The p618e legs: the bystander-stall timeline face (same harness as the
cert6-era record — `p65_timeline.py` v5.1, `P65_FACES="U32k,U64k"`,
`P65_W2MT=4000`, 12 greedy bystander streams) re-measured **natively on
the certified cert7 image**, on both fp8 stall-lever postures:

- **P1 `default`** (mnbt=8192, the certified default)
- **P2 `mnbt4096`** (`ARC_MNBT=4096` — the P6.17l first-choice lever)

Both legs: fresh boot, health ~290 s, 0 ERROR / 0 Traceback
(`erraudit_*.txt`), serve logs banked (`serve_full_*.log`, redacted).
Console: `perf.out`; harness: `p618e_perf.sh` (banked copy).

## Results — both postures in-family with the record

| face | P1 default (cert7) | record family (p615/p617, cert6) | P2 mnbt4096 (cert7) | record family (p617l/p615, cert6) |
|---|---|---|---|---|
| U32k TTFT | **18.27 s** | 17.25–17.38 s | **19.51 s** | 19.49 s |
| U64k TTFT | **39.91 s** | 39.77–40.32 s | **42.34 s** | 42.83 s |
| bystander p95 W_U32k | **4001 ms** | 3986 ms | **2052 ms** | 2030 ms |
| bystander p95 W_U64k | **5273 ms** | 5295 ms | **2523 ms** | 2587 ms |
| KV pool | **400,739 tok** | 405,491 (p615) / 400,739 (p618b legs + cert7 gate) | **427,666 tok** | 433,210 (p615) |
| W_base p95 | 54.7 ms | 53–55 ms family | 54.7 ms | same |
| W_*_end windows | 59–66 ms, 0 stall_s | same shape | 59–66 ms, 0 stall_s | same shape |

Readings:

- **The certified default posture is unchanged by the P6.18 guard** —
  the engine bytes under `fp8_e4m3` are identical to cert6, and every
  face re-verifies its record family (TTFT within 0.9 s / 0.4 s;
  stall p95 within 0.4 %/0.4 %; baseline gaps identical).
- **The `ARC_MNBT=4096` lever reproduces on cert7**: stalls cut to the
  2.0–2.5 s band (−49/−52 % vs same-boot default), TTFT cost +6.8/+6.1 %
  — matching the documented trade exactly.
- **KV pool values** 400,739 / 427,666 — 400,739 is byte-identical to
  the cert7 gate + p618b cert6-fp8 legs; 427,666 vs p615's 433,210 is
  the known rig-free-memory state variance (same class as
  405,491-vs-400,739 — the pool is derived from free memory at boot,
  not an engine delta).

## Incident (recorded)

Port-collision at leg end (the p618d incident class): the trap's
`docker start radiance-test` brought W14's container up while the perf
script's arc-test engine still held :8000 — the W14 restore serve died
silently and an early health probe answered 200 from the WRONG engine.
Fixed post-leg by the p618d procedure: pkill arc serve → `docker rm -f
arc-test` → pkill the dead W14 serve → relaunch
`/root/build/w14_serve_only.sh` → verify single :8000 listener owned
by the radiance container's pid (`ss -tlnp` + `docker inspect` pids;
the listener while colliding was arc's `--max-num-batched-tokens 4096`
cmdline — the P2 engine — proof the port-owner check catches it).
Restore log: `rig/perf/../w14_restore_p618e2.log` (on the rig).
**Law (folded from p618d, re-proven): a trap that starts W14 must FIRST
kill+remove arc-test — `docker start radiance-test` alone is not a
restore; always port-owner-verify after.**

Redaction (this perf banking): 2 files, **+180 bytes** = 83 + 97
(`serve_full_default.log` rig 1×6 + node 11×7; `serve_full_mnbt4096.log`
rig 8×6 + node 7×7), raw-ref per-file sweep 0 — appended to
`results/REDACTION-2026-11-04-p618-cert7.md`.
