# ARTIFACT — the 210 MB fat library is NOT tracked (by law of the host)

`libccl.so.1.0` (210,192,096 bytes) was banked into this directory at the
r99 gate and then REMOVED from the tree on 2026-11-05: GitHub's pre-receive
hook hard-rejects any file over 100 MB (`GH001`, push declined), so the
tracked tree must carry identity, not bytes.

## Identity (the sha IS the artifact contract)

```
sha256  3a120e8b352fbd6be45ab61a7f4e927d80c9bc6618ff646e6db68b0263eca4a7
size    210,192,096 bytes
```

Same sha banked independently at: VERDICT.md r97 lane header,
`r98/preflight.txt:2` (rig stage dir at gate time), `r99/manifest_cert10.txt`,
and re-verified on the rig `2026-11-05T21:5xZ` (see below).

## Durable homes (verified, not assumed)

1. **Rig stage dir** — `/root/build/p624/stage-arc/libccl.so.1.0`
   (sha256 re-verified identical after the untrack decision, 2026-11-05).
2. **Inside the certified image** — `vllm-arc:v0.30.0-b70-cert10`
   (id `2f86fc2411fb`) at `/opt/venv/lib/libccl.so.1` AND
   `/opt/venv/lib/libccl.so.1.0` (two regular files, both the fat lib;
   proven by the r99 gate build log + in-image sha check, and the r98/r99
   in-situ md5s).
3. **Local bare origin** — `C:/Projects/vllm-arc-origin.git` retains the
   blob unreferenced in its object store from the first (superseded)
   main push; incidentals only, do not rely on this one.

## Reproducing (build is fully tracked even though the product is not)

`tools/build_oneccl_arc.sh arc` on the rig: pinned superproject tag
2022.0.0 @ `1318c3aa…` + submodule pin + hash-verified DLE 2026.0.0
builder + the tracked `external_patches/oneccl/patches/arc-fp8-allreduce.patch`
(v4, sha256 `effe9a27…` VERDICT r97) → stages `stage-arc/libccl.so.1.0`.
A rebuild is NOT promised byte-identical (toolchain nondeterminism is
unmeasured); verify the sha256 above before treating any rebuild as this
artifact. The image recipe (`cert10.Dockerfile` here) already takes the
sha as `LIB_SHA` build-arg and FAILS the build on mismatch.

To re-bank the context dir for a docker build of the image: copy the
artifact from home 1 or 2 into this directory, then
`docker build -f cert10.Dockerfile --build-arg LIB_SHA=3a120e8b… .`
(the Dockerfile header has the full command).
