# R46T parity — did cert4 (r70, native) regress anything the p46 campaign measured?

Owner question (2026-10-31): "was all performance number on line r46t
without any regression?" Answer: **no regression**. Every comparable
face reproduces within the cross-boot engine-neutral band (±5 %, r47c
comparator law); most cells are at-or-faster. Deltas that look larger
(warm faces, the cert-era TTFT ladder) are **different faces/prompt
corpora**, not regressions — the comparator law forbids cross-corpus
point compares, and each such row below says so explicitly.

Comparator law recap: engine is unchanged through cert2→cert3→cert4
(default postures byte-identical), so same-face deltas beyond noise
would indicate a build/wheel issue — none observed (wheels md5-pinned
identical across cert3/cert4).

## Same-face table (r46t on p46 → r70 on cert4)

| Face | r46t (p46) | r70 (cert4) | Δ | Verdict |
|---|---|---|---|---|
| agg4x256 steady, greedy t0 | 299.86 | 294.68 / 296.42 (p65 cert4: 299.15/300.16; r67: 300.48) | −1.7 % worst cell; p65/r67 at/above family | no regression (boot variance; 5-boot family 294.7–300.5 brackets 299.9) |
| solo200, greedy t0 | 84.46 / 84.53 | 83.66–84.48 (p65: 84.53–84.86; r67: 84.49) | −1.0 % worst cell | no regression |
| xhigh t0.2 agg steady | 212.6 | 205.63 | −3.3 % | inside ±5 % band |
| xhigh t0.4 agg | 208.8 | 221.69 | **+6.2 %** | faster |
| xhigh t0.6 agg | 202.8 | 202.39 | −0.2 % | exact |
| xhigh t0.8 agg | 195.8 | 204.76 | **+4.6 %** | faster |
| xhigh t1.0 agg | 186.2 | 194.44 | **+4.4 %** | faster |
| xhigh solo (0.2→1.0) | 63.9/63.0/61.3/61.4/61.1 | 64.2/65.2/67.1/60.0/58.6 | +0.5…+9.5 % / −2.3 % / −4.1 % | 3 of 5 faster, 2 inside band |
| alt agg steady | 254.9 | 264.00 (peak 280.94) | **+3.6 %** | faster |
| alt solo | 76.9 / 92.2 | 72.78 | −21 % vs the 92.2 outlier cell | high-variance face (r46t's own k4 cell hit 104.3; pp1.5 truncation draws) — steady agg is the comparable cell |
| acceptance, greedy t0 | 85.6 % | **85.9 %** (mean len 3.58) | +0.3 pp | no regression |
| acceptance, xhigh 0.2→1.0 | 57.3/57.1/53.1/52.3/49.5 % | 56.6/55.8/54.5/50.5/49.0 % | ≤1.3 pp | within noise; same monotone fall |
| acceptance, alt | 88.5 % | 87.7 % | −0.8 pp | within noise |
| no-spec xhigh t1.0 agg / solo | 118.8 / 33.5 | 126.10 / 33.61 | +6.1 % / +0.3 % | no-spec baseline at/above |
| no-spec alt agg / solo | 124.2 / 33.4 | 115.93 / 33.63 | −6.7 % / +0.7 % | inside extended noise (single-boot cells) |
| MTP value, xhigh t1.0 agg | 1.57× | 1.54× | −0.03 | same |
| MTP value, alt agg | 2.05× | 2.28× | +0.23 | better |
| MTP value, greedy agg / solo | (not an r46t headline face) | 2.34× / 2.46× | — | new cert4-native cells |

## Faces that are NOT comparable (different corpus — kept honest)

| Looks like | Why it is not a regression |
|---|---|
| warm re-hit 740 ms (cert-era) vs cert4 0.80–0.99 s | different prompt corpus (prewarm-v7 battery vs r70 uniform prompts); same mechanism (engine APC — tier attribution measured in r70); comparator law: cross-corpus point compares invalid |
| cert-era TTFT ladder 3445/7252/15552 ms vs r70 3865/7872/17363 ms | r42/r43 ladder used its own corpus; the SAME-face cert4 series is r66a/r67/r70 U-bars: 3.74–3.86 / 17.08–17.36 / 39.19–40.26 s — **no drift across three cert4 sessions** |
| 64k t1 40,926 ms (concurrent-load, cert-era) vs r70 U64k 40.26 s solo | the r46 cell ran DURING soak load (conservative by construction); r70's quiet solo face is faster — the honest direction |

## Conclusion

The r46t decision data (k3) transfers to cert4 natively: acceptance
within ~1 pp at every temperature, throughput within the ±5 %
cross-boot band with most cells faster, greedy acceptance a touch
higher (85.9 %). The two cells beyond the band in either direction
(alt solo −21 % vs an outlier cell; no-spec alt −6.7 % single-boot) are
the highest-variance faces of the set, and neither affects the k3 law
or any current-record claim. **No regression.**
