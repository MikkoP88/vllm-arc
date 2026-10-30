# r47e — cert3 tag-built image (P6.4 close)

**VERDICT: GREEN — `vllm-arc:v0.30.0-b70-cert3` (id
`sha256:60d79fd1c81e…`) is the certified release image. Manifest
`fork_sha=fdca6f51` == annotated tag `rel/v0.30.0-b70-cert3` commit SHA
== the r47d commit the p48 gates ran on (A/B check); image-native
verify 28/28 `ALL_GREEN` (same check set, same result as the p48 gate
image `7b091de80061`).**

## What ran (r46v-precedent standard for tag-built certification)

1. **Tag**: `rel/v0.30.0-b70-cert3` (annotated, object `b1f33cd4`) at
   commit `fdca6f512316af66f6d6876758f814ee5915688e` — the r47d stale-
   hunk fix commit, i.e. the exact tree the r47c p48 gates validated
   (28/28 verify, certified 25-var boot, zero fork-added WARN/ERROR,
   perf ±5 % same-posture vs cert2). Pushed to both remotes (origin +
   github; github `main` is PR-protected — tags are the certified
   pointers).
2. **Build**: `git archive rel/v0.30.0-b70-cert3` → rig
   `/root/build/arc-cert3.tar` → `build_image.sh <tar> fdca6f51
   vllm-arc:v0.30.0-b70-cert3` (log rig `/root/build/cert3_build.log`).
   - wheel_guard stage 4/9 under `set -euxo pipefail` — all four md5s
     enforced vs `/etc/vllm-arc/pins.md5` (build aborts on mismatch).
   - stage 8/9 in-build gates: version asserts (vllm 0.30.0, torch
     2.13.0, triton 3.7.2) + `PLUGIN_IMPORT_OK` / `ARC_ENTRYPOINT_OK` /
     `PLATFORM_RESOLVE_OK` / `ARC_PLATFORM_IMPORT_OK` /
     `VXK_ARC3_PIN_OK` / `LMCACHE_ARC1_PIN_OK` (`0.5.6+arc1`) /
     `LMCACHE_RUNTIME_DEPS_OK`.
   - 9 cached stack layers (source-only rebuild on the p48 stack —
     layer-order law).
3. **Manifest A/B** (`cert3_manifest.txt`): `fork_sha=fdca6f51`,
   `lmcache=0.5.6+arc1 … 352295b4af1e2c58da8a2363b05a4f0c`,
   built 2026-10-30T22:09:12Z.
4. **Image-native verify** (`cert3_verify.out`): throwaway
   `--privileged --ipc host` container, `p48_verify.py` (the 28-check
   set: r46u 13 + 0.5.6 installed-source markers) →
   `TOTAL 28 checks, 28 pass, 0 fail — P48_VERIFY: ALL_GREEN`.

No live boot was re-run: cert3 is tree-identical to the gate image p48
(same `fork_sha`), whose certified boot + battery + hygiene evidence is
`results/2026-10-31-p6-r47c-p48-056gates/` — the r46v precedent (tag-
built cert2 verified by manifest + image-native repro, no re-boot).

## Docs re-pointed (this commit)

README (lineage, quick start incl. the 25-var paste-block fix —
`LMCACHE_TRACK_USAGE` was missing from the serve script, header said 9),
CONFIGURATION-arc image table (cert3 head, cert2/p48/p47/p46/cert
history), MIGRATION-arc identities → cert3 + P6 addendum, master plan
§12 P6 STATUS.

Rig end-state: W14 incumbent up (`radiance-test` +
`w14_serve_only.sh`); cert3 + p48 images on the rig; no arc containers
left running.

## Artifacts

- `cert3_manifest.txt` — IMAGE_MANIFEST from the cert3 image
- `cert3_verify.out` — 28/28 ALL_GREEN, image-native
- `build_keylines.txt` — image id, tag→commit lineage, gate summary
