# r212u — verbatim-body rung: BODY INNOCENT; the pool was MIS-ATTRIBUTED — it includes the K walk

**Date**: 2026-11-21 · **Leg**: r212u (arc7 Phase-8 instrument) · **Stack**:
r212u wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`dd6641ccbf0419829dacf7a036b06ece`; r212t tree + `p212u_patch.py` —
rung 16 = the fast loop's EXACT body as a RUNTIME loop inside the
light path, runtime-indexed `kc[r]` / `st[v0][jb>>2]` reads included
(kc 120-initialized in the rung so the read cannot be UB-folded);
fast path byte-identical; TU preflight clean, ninja-resume build
`r212u_build1.log`) + `vllm-arc:v0.30.0-b70-cert25zc` · **Run**:
`r212u_leg1.log` 19:38–19:44 UTC (three same-wheel cells:
m0 / m11 / m16 × 32768)

## Verdict

**m16 379.5 ≡ m11 379.4 (+0.1 µs).** The fast loop's body — verbatim,
as a runtime loop, with runtime-indexed local-array reads and the
variable shifts — is FREE in the light context. Combined with the
chain (branches r212o, loop form r212r, domain r212q −24 µs, converts
r212p, int ops r212s, select+assembly r212t-m14, memcpy store
r212t-m15, verbatim body r212u-m16), the V fast loop's body is
innocent in every component and every form.

| cell | µs | delta vs m11 |
|---|---|---|
| m0 (fast full) | 718.0 | +338.5 = the pool |
| m11 (light, unrolled) | 379.4 | — |
| m16 (verbatim body, runtime) | 379.5 | **+0.1** |

m8 byte-identity oracle 1318.3 (+0.2%, in band); canaries 0; m0 718.0
(same-binary law, fifth consecutive wheel).

### The attribution correction (found reading the mainloop after the CONTEXT fork fired)

The pre-registered CONTEXT fork ("the fast-loop placement/interaction,
disasm decides") is SUPERSEDED by a simpler explanation found by
auditing every `vlane_probe` use in the mainloop: **the m-series cells
sever the K walk too.** `mx_paged_decode_mainloop.hpp`:

- GEMM 1 K loop: `if (params.vlane_probe >= 6)` const-fills `tSrK`
  (severs copy + `dequant_dtile` + reorder) — true for every m-cell
  (9, 11, 12–16);
- GEMM 2 V loop: `if (params.vlane_probe >= 2 && vlane_probe != 4 &&
  != 5 && != 6)` const-fills `tArV` — also true for every m-cell.

So m0 (vprobe 0) = K dequant_dtile REAL + V dequant_vslm REAL, while
every m-cell = K CONST + V-variant. **The "assembly pool" m0−m11
(338 µs) is the K walk's full cost plus the V fast−light delta —
never a V-only quantity.** All V-side deltas (m9≡m11, ladder flat,
m14/m15/m16 free) remain valid: their arms all had K severed. The
r212n lanes (K ≈ 178 µs, Vg ≈ 197 µs) fit this arithmetic
(178 + ~160 ≈ 338). Every m0-moving patch (r212o/q/r) edited BOTH
fast loops, so m0's −24 µs (r212q) may have been K-side.

### The fork

**Next (r212v): K-RESTORE RUNG — m17 = K REAL + V LIGHT.** Three-line
mainloop patch: exclude 17 from the K const-fill (`>= 6 && != 17`),
exclude 17 from the V const-fill, extend only the vslm light range to
17 (dtile keeps `<= 16` so the K walk runs its real fast loop; mode 17
falls to the plain m11-shape light tail). Same wheel reruns m11/m0.
Telescoping split: `k_lane = m17 − m11` (the K walk's real cost with
V light in both arms), `v_pool = m0 − m17` (the V fast−light delta
with K real in both arms), `k_lane + v_pool = pool` (check).
k_lane > 0.7·pool → K DOMINANT → the attack shifts to the K walk
(dtile u32/SWAR/window surgery) with V already at its floor;
v_pool > 0.7·pool → V dominant after all; split → attack the bigger
half first.

## Artifacts

`offline/p212u_{m0,m11,m16}_32768.json`, `r212u_build1.log`,
`r212u_tu.log`, `r212u_leg1.log`. Rig: `/root/build/p199/out/p212u_*`,
`/root/build/r212u_{build1,tu,leg1}.log`. Tools: `p212u_patch.py`,
`r212u_verbatim.sh`.
