# r263 — the cert29 image rebuilt from the r262 content commit (P6.47)

**VERDICT: BUILT — `vllm-arc:v0.30.0-b70-cert29` now carries the r262
stride fix (content commit `4dd3a1b9710f1c638f7f0b5ac1885525cf57c3bc`), with
every in-image assert and byte pin green. This is a CANDIDATE, not a
certificate: the section-10 battery on this image is the r264 gate, and the
cert29 cut is held until the gemma MX lanes and the LMCache lane are all
green there (owner directive, 2026-11-27).**

## Why a rebuild

The first cert29 image (content commit `16effb497cd36cb25df7c19cb6f1938d2d45cdc5`)
passed the qwen side of the r255 gate and died on all three gemma MX legs
at the longprobe multi rehit. r262 root-caused the death (the expand-view
block-table stride defect) and proved the bridge guard e2e
(`R262_SUMMARY a_probe=0 c_suite=0 b_ctrl=2`); the fix landed on `main` in
PR #167. An image carries files, so the fixed tree had to be baked.

## What ran

- `tools/ar/r263_ship.sh 4dd3a1b9710f1c638f7f0b5ac1885525cf57c3bc` (local):
  extracts the six `vllm_gaudi` files STRAIGHT FROM THE GIT BLOBS
  (`git show <commit>:<path>` — never the working tree, so a stray edit
  cannot ride into the image), pins each by md5 against the build
  script's table, ships them flat to `/root/build/cert29stage_r263`, and
  ships the runner. Every transfer is md5-verified on the far side.
- `bash /root/build/r263_build_cert29.sh 4dd3a1b9710f1c638f7f0b5ac1885525cf57c3bc`
  (rig; runner md5 `1d28b9a704cd946b740aa3e087ac8576` after the fix below):
  refuses to run without the evidence lines (r245a offline `VERDICT: PASS`,
  `R245_DONE failed=[]`, the P-D probe 11/11, `INTEROP_PASS`, the P-H4
  probe 15/15, the r262 A/B summary), refuses the cert28 base unless its
  manifest carries the certified `cert28_fork_sha`, pins the build34
  wheel by md5 and sidecar, then builds `FROM cert28` + the wheel + the
  six files with a python assert block that runs INSIDE the image.

Two runs. Run 1 produced a correct image and a false-negative final line;
the check was fixed and the build re-run for a coherent log.

## Result

| item | value |
| --- | --- |
| rebuilt image | `vllm-arc:v0.30.0-b70-cert29`, id `77e08563f7c7`, 65.9 GB |
| manifest | `cert29_fork_sha=4dd3a1b9710f1c638f7f0b5ac1885525cf57c3bc` |
| manifest | `cert29_built=2026-11-27T13:33:11Z` |
| manifest | `cert29_rebuilt=r263 (supersedes the 16effb49 crash image, kept as vllm-arc:v0.30.0-b70-cert29-rc1)` |
| crash image, kept | `vllm-arc:v0.30.0-b70-cert29-rc1`, id `4a4e9b0dd244` |
| base | `vllm-arc:v0.30.0-b70-cert28`, id `9852a2ce091a` |
| run 2 | `R263_ALREADY_BUILT` (same content commit), `R263_STAGE_OK`, `R263_CERT29_BUILT 2026-11-27T13:35:47Z`, `R263_IMAGE_OK` |

The in-image assert block (27 source-string asserts + the byte pins) ended
with `CERT29_WHEEL_PLUS_DELTA_LIVE asserts OK engine.py
d1fa3c17cde044211cbf62e28ddcc1b1 native_decode.py
81e427590e7ad3482e46bdc25f4a62b2`. The four r262 asserts are new vs the
r254 build: the bridge guard (`if not bt_rows.is_contiguous():`), the
materialization (`bt_rows = bt_rows.contiguous()`), its one-line-per-class
log string, and the call-site contract comment in `engine.py`.

## The six staged files (md5 of the git blobs at 4dd3a1b9 = md5 in the image)

| file | md5 | vs the 16effb49 image |
| --- | --- | --- |
| `envs.py` | `31ca4ac7cfcd996aed6446d4b24c2bcd` | unchanged |
| `platform.py` | `0733f0b8c7dd2fb89422bafa56ec35f5` | unchanged |
| `lmc_mx_boot_gate.py` | `04408d2e12cbdd8587cac4633624b7bc` | unchanged |
| `lmcint/vllm_v1_adapter.py` | `7ca7cff4e1e04d4e0ea73573b1234664` | unchanged |
| `mx_kv/engine.py` | `d1fa3c17cde044211cbf62e28ddcc1b1` | changed (the r262 call-site comment) |
| `mx_kv/native_decode.py` | `81e427590e7ad3482e46bdc25f4a62b2` | changed (the r262 bridge guard) |

So the whole delta between the crash image and this one is the bridge
guard plus a comment — the A/B of r262, baked.

## Defect in this rung: run 1's final check was a false negative

Run 1 (launched ~13:26Z) aliased the crash image
(`R263_CRASH_IMAGE_ALIASED ... cert29-rc1`), built a correct image
(`R263_CERT29_BUILT 2026-11-27T13:29:02Z`; manifest
`cert29_built=2026-11-27T13:26:20Z`; image id `e091ebd5b3f7`) and then
printed `R263_MANIFEST_FORK_SHA_MISMATCH`. The image was right; the check
was wrong: it compared `tail -1` of `/etc/vllm-arc/IMAGE_MANIFEST` with
the fork sha, and the manifest's last line is the newly appended
`cert29_rebuilt=` entry, not `cert29_fork_sha=`. The manifest was read
directly on the rig (`cert29_fork_sha=4dd3a1b9...`), the check was
changed to `grep -q "^cert29_fork_sha=$FORK_SHA\$"` over the whole file,
the fixed runner was shipped (md5 above) and the build re-run. Run 1's
log is banked as `build_run1.log`; its image lost the tag to run 2 and
sits untagged on the rig.

Law: read a manifest key by name, never by position on an append-only
file.

## What this bank does NOT show

- Any serving evidence. The fix is proven e2e in eager posture only
  (r262); graph posture, the gemma comparative verdict, and the qwen +
  LMCache re-certification on these bytes are the r264 battery
  (`tools/ar/r264_cert29_gate.sh`, launched 2026-11-27T13:43:40Z).
- That a certificate follows. The cut waits for
  `R264_CERT_PRECONDITION ... -> CERT_CUT_ALLOWED`.

## Banked files

`build.log` (run 2), `build_run1.log` (run 1, superseded),
`IMAGE_MANIFEST` (verbatim from the rebuilt image), `manifest_tail.txt`,
`images.txt`, `stage_md5.txt`, `script_md5.txt`, and the data-file
`SHA256SUMS`. No host addresses occur in these artifacts (audit clean).
