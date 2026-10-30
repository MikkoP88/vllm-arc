# r46v — build-path revalidation: is `git checkout <tag>; tools/build_image.sh` real?

**Date:** 2026-10-30 · **Outcome: the README quick-start build step was INVALID as written; the real procedure is now validated end-to-end on the rig and documented**

## The question

Operator: "is building deploying really this simple and are all
dependencies + patches also included?" — aimed at quick-start step 1
(`git checkout rel/v0.30.0-b70-cert` + bare `tools/build_image.sh`).

## Audit findings (code-level: tools/build_image.sh, Dockerfile, wheels.pins.md5)

The 2-line quick start was wrong in three ways:

1. **Invocation** — `build_image.sh` requires
   `<fork-archive.tar> <fork-sha> [tag]` (usage-guards exit otherwise)
   and the bare call in the README would die immediately.
2. **Host** — the script runs ON THE RIG: it stages
   `/root/build/arc_build/{Dockerfile,wheels/,pins.md5,tools/,src/}` and
   `docker build`s there. The repo side produces the archive
   (`git archive`).
3. **Tag predates the r46u fix** — `rel/v0.30.0-b70-cert` points at the
   certified commit; building from it TODAY reproduces the p41c
   glue-alias defect (fixed on main @ 38cf63f8, r46u).

Dependencies truth (what "all included" means precisely):

- **Patches: YES, in-repo.** `COPY src/` + `pip install /build/vllm-arc`
  — the fork dist (incl. `vllm_gaudi/xpu/patches/`) installs from the
  archived tree. Proven below: a build from main passes the r46u glue
  repro image-natively.
- **4 arc wheels: pinned in-repo, FILES rig-resident.** md5s in
  `external_patches/wheels.pins.md5` are enforced at build time by
  `tools/wheel_guard.sh`, but the wheels are copied from rig stashes
  (`/root/build/radiance_wheels/` vllm+triton, `/root/build/wheels-arc-vxk/`
  arc3 kernels, `/root/build/wheels-arc-lmc/` lmcache+arc1). The lmcache
  wheel is rebuildable from `external_patches/lmcache/` (provenance +
  build law documented there); the vxk wheel from
  `external_patches/vllm-xpu-kernels/`.
- **Base image: rig-local.** `vllm-iexperimental:base-v0.26.0` (not in
  any registry) — the validated Intel XPU runtime lineage.
- **Rest of the python stack: version-pinned but FETCHED** — torch
  2.13.0+xpu family from `https://download.pytorch.org/whl/xpu`
  (PIP_EXTRA_INDEX_URL in the Dockerfile), xgrammar/numba/otel/… and the
  lmcache closure (aiofile, redis) from PyPI. After the first build these
  layers are docker-cached (layer-order law: stack layers precede
  `COPY src/`, so source-only rebuilds re-run just the fork dist +
  gates + manifest).

## Empirical validation (the procedure the docs now state)

From the working repo @ main 38cf63f8 (r46u merged):

    git archive --format=tar -o fork.tar main            # repo side
    scp fork.tar root@<rig>:/root/build/
    # rig side:
    tar -xOf fork.tar tools/build_image.sh > build_image.sh
    bash build_image.sh /root/build/fork.tar <sha> vllm-arc:v0.30.0-b70-p47

Observed (evidence/p47_build.log): wheel_guard PASSED (4/4 md5s =
wheels.pins.md5), stack layers CACHED, fork-dist + in-build gates ran
clean, image vllm-arc:v0.30.0-b70-p47 built (11.2 GB),
`fork_sha=38cf63f89450315737dd5ac30ff231cfc52c2994` in
`/etc/vllm-arc/IMAGE_MANIFEST` (evidence/p47_manifest.txt).

**Patches-included proof**: the r46u import repro run against the BUILT
image — no bind-mounted fixes, image-native code — 13/13 PASS,
`R46U_VERIFY: ALL_GREEN` (glue alias + symbols + `_glue()` primary and
fallback + p41c wrap live). This is simultaneously the first image
carrying the r46u fix: `vllm-arc:v0.30.0-b70-p47`.

## Actions

- README quick-start step 1 rewritten to the validated two-phase
  procedure with the prerequisites stated (rig build host, wheel
  stashes/guard, base image, first-build internet) and the r46u note
  (build from main ≥ 38cf63f8 until a re-certified tag is cut).
- No change to build machinery (build_image.sh/Dockerfile) — the audit
  validated them; only the documentation lied.
- p47 left on the rig as the first r46u-fixed image; the operator's
  arc-test container (hot-patched p46) still serves — p47 available for
  the next container recreate (container-freshness law).
