# P6.18e reproducibility audit — `vllm-arc:v0.30.0-b70-cert7` rebuildable from the fork (2026-11-04)

Owner directive: *every fix / improvement / change carried by the certified
image must exist in the project, in line with the upstream best-practice
patch set — so any user can build a new certified image from the fork.*

Three questions, answered in order:

1. Does every file inside the certified image's fork layer exist in the
   repo, byte-identical? → **YES — proven, 144/144** (§1).
2. Is every *other* image input either a tracked artifact or a rebuildable
   kit? → **YES, with two documented external prerequisites** (§2).
3. Can a user actually run the rebuild? → **YES — the exact commands**
   (§3, same flow that built this image).

## 1. Fork-layer parity: image ⊆ tag tree, byte-exact

Method (rig, 2026-11-04): `git archive` of tag `rel/v0.30.0-b70-cert7`
(fork_sha `5bf4192d`, the same tarball the image was built from —
`cert7b_p618c2.tar`) extracted to `/tmp/parity/src`; the image's installed
`/opt/venv/lib/python3.12/site-packages/vllm_gaudi` walked with
`__pycache__` pruned; both sides `LC_ALL=C sort`-ed, set-compared with
`comm -3`, and md5-compared file-by-file.

| side | files | result |
|---|---|---|
| tag tree (`src/vllm_gaudi`) | 145 | — |
| installed in image | 144 | **every one of the 144 md5-identical to the tag tree** |

The single set difference is **repo-only**: `vllm_gaudi/extension/bucketing/
bucketing_file.txt` — a one-line *sample* file (`# This is a sample
bucketing file`) inherited unchanged from upstream vllm-gaudi (landed
upstream ~v0.16.0rc0, "Buckets from file - alpha version", #375).
It is dropped by `pip install .` because neither our nor upstream's
packaging declares `package_data` — i.e. **upstream's own wheels drop it
identically** (verified against the pristine mirror `C:\Projects\vllm-gaudi`:
same file present, same absence of a package_data declaration). No code
references the filename; the `file` bucketing strategy takes a
user-supplied path from env/config (`vllm_gaudi/extension/bucketing/
file_strategy.py`), so the file's absence is inert for every serving
path — and the direction that matters for this audit (image ⊆ repo) has
**zero exceptions**. Fork packaging is untouched (scope law), so this is
an upstream-inherited packaging nuance, not a fork defect.

Evidence: `rig/parity/{tag,img}_files.txt` + `{tag,img}_md5.txt`
(145/144 lines; md5 lists differ only by the bucketing_file.txt line).

## 2. Every image input → tracked artifact or documented external prerequisite

The build is `tools/build_image.sh <fork.tar> <fork_sha> [tag]` staging
`/root/build/arc_build/{Dockerfile,wheels/,pins.md5,tools/,src/}` and
`docker build --target vllm-arc` (3-stage Dockerfile: base → stack swap →
fork dist + in-build gates + `IMAGE_MANIFEST`).

| image input | where it comes from | status |
|---|---|---|
| fork source (`vllm_gaudi/`, incl. all P6.x patches + the P6.18 guard) | the git tag itself (`git archive`) | **tracked** — parity §1 |
| `Dockerfile` | repo root | **tracked** |
| `tools/build_image.sh`, `tools/wheel_guard.sh` | repo `tools/` | **tracked** |
| wheel pins | `external_patches/wheels.pins.md5` (4 md5s + provenance comments) | **tracked**; enforced in-build by `wheel_guard.sh` (build fails on md5 mismatch) |
| `lmcache-0.5.6+arc1` wheel | **rebuildable**: `external_patches/lmcache/` kit = README (C-base + D-deltas framing) + `build_lmcache_arc.sh` + inner script + 4 patches (`e-adapter-vllm-056-graft`, `e01-xpu-platform-ops`, `e02e03`, `e04-usage-tlm-gate`) applied to upstream 0.5.6 sdist (PyPI, official-wheel md5 in pins comment) + SYCL XPU build (`BUILD_WITH_SYCL=1`, oneAPI 2026.0 generation) | **tracked kit** — wheel provenance byte-documented in pins |
| `vllm_xpu_kernels-0.1.14.1+arc3` wheel | **rebuildable**: `external_patches/vllm-xpu-kernels/` kit = README + `build_vxk_wheel_arc3.sh` + inner + `patch_vxk_int64fix.py` (the §6 int64 wedge fix) + `vxk_gate_repro.sh` | **tracked kit** |
| `vllm-0.30.0+xpu` + `triton-3.7.2+xpu` wheels | donor line-D validated stack (IEXP image venv ground truth, `results/2026-10-27-p0-venv-groundtruth/`); **md5-pinned**, explicit filenames in `build_image.sh` (arc1/arc2 ABI rejects can never slip in) | **external pinned prerequisite** — not on public indices; must be obtained from the rig/donor artifacts (fork-model law: donors are read-only, components enter as pinned artifacts, never as remotes) |
| base image `vllm-iexperimental:base-v0.26.0` | rig-local; Intel XPU runtime base (compute runtime, level-zero, oneCCL, /opt/venv py3.12) — the lineage every validated rig image descends from | **external prerequisite** — `ARG BASE_IMAGE`, overridable at build time; md5s of everything stacked on it are what the pins freeze |
| torch 2.13.0+xpu family, xgrammar etc. | `pip` from `download.pytorch.org/whl/xpu` + explicit pins in the Dockerfile RUN | network pins; the resolved freeze is **banked inside the image** at `/etc/vllm-arc/venv-freeze.txt` (`pip freeze`) so any drift is detectable against the certified freeze |

In-build gates (fail the build, no GPU needed): version asserts
(vllm 0.30.0 / torch 2.13.0 / triton 3.7.2), `vllm_gaudi` plugin import,
`vllm.platform_plugins` entrypoint == `vllm_gaudi.xpu:register`,
platform resolution to `ArcXpuPlatform`, vxk `+arc3` pin, lmcache
`0.5.6+arc1` + `lmcache.xpu_ops` import, lmcache runtime deps. Then
`/etc/vllm-arc/IMAGE_MANIFEST` embeds `fork_sha`, per-wheel md5s, base
image name, and the venv freeze — the provenance anchor that makes any
future image independently checkable against a tag.

## 3. The rebuild commands (exactly what produced cert7)

From a clone of this fork, checked out at a certified tag (`$SHA`,
e.g. `rel/v0.30.0-b70-cert7` → commit `5bf4192d`):

```bash
git archive --format=tar -o fork.tar $SHA
# stage to the rig (wheels/ + base image live there; see §2 externals)
scp fork.tar root@$RIG:/root/build/fork.tar
ssh root@$RIG 'tar -xOf /root/build/fork.tar tools/build_image.sh > /root/build/build_image.sh'
ssh root@$RIG bash /root/build/build_image.sh /root/build/fork.tar $SHA vllm-arc:v0.30.0-b70-cert7
# verify: manifest fork_sha == $SHA, wheel md5s == external_patches/wheels.pins.md5
docker run --rm --entrypoint cat vllm-arc:v0.30.0-b70-cert7 /etc/vllm-arc/IMAGE_MANIFEST
```

(Also mirrored in the README build-from-fork section; the first build
needs internet for the pytorch XPU index, later builds cache.)

## Verdict

**cert7 is reproducible from the fork.** All fork-layer content in the
image exists in the repo byte-identical (144/144, §1); both internally
patched wheels (lmcache, vxk) are rebuildable from tracked kits with
upstream sdists as the only source inputs; the remaining inputs (two
donor +xpu wheels + the base image) are frozen by md5 pins + the
in-image freeze, with no silent substitution possible
(`wheel_guard.sh` + explicit filenames + manifest cross-check). A user
with the two pinned external artifacts and the rig base image can
rebuild the certified image with §3's commands and verify it against
`IMAGE_MANIFEST` — which is how cert7 itself was built and gated
(`rig/cert7b_build2_fixed.out`).
