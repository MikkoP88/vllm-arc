# r231d — seqused-tightened splits: PASS, offline gates green (P6.47, 2026-11-23)

**Leg**: r231d (arc7 Phase-8, the final priced knob on the MX walk
line). **Patch**: device-side split clamp, mirrored in
`mx_paged_decode_kernel.hpp` (legacy path, ~L333) + `mx_reduce_split.hpp`
(~L223) — `seq_num_kv_splits = min(num_kv_splits, max(1,
windowed_k_blocks/4))` (>=4 k-tiles per split below capacity). Tree
patches via `tools/ar/r231d_apply.py` (pristine-md5 refuses,
`.r231dprebak` siblings); build28 via `r231d_build.sh` (incremental,
VXK_ARC7MX_CLEAN=0, ~43 min).

**Why** (r231c lever 1, the largest measured remaining delta): the
host split count prices from max_seqlen_k CAPACITY (the r212f no-sync
law — seqused_k is device-resident), so shallow-ctx decode runs ~128
splits: +92% over-provisioning at 2K walked ctx. The clamp derives
the env-pin semantics per-seq ON DEVICE. Buffers keep the capacity
layout (offsets still size from the host count) — the reduce reads
only `i < effective_splits` (mirrored derivation, or it would read
unwritten Oaccum slots).

**Pin-path note** (verified in `mx_attn_interface.py` L184-203):
`num_splits_kv` without `host_kv_lens` leaves `splits_per_seq = None`
— the pin reaches the kernel as the legacy scalar and is therefore
clamped too below capacity. At the banked posture
(ARC_MXKV_NSPLIT=128) this only bites below 8K walked ctx, which is
the designed dynamic win; >=8K is byte-identical (G3).

## Identity

- build28 `libattn_kernels_xe_2.so` md5 `61d2f5a365f05084b3dd72c4ae1da3e0`
  (build25/cert27 pin `2d73ff7a3c2aa2caab86179288ca3711`); fa2 stub
  `6e6c48508d40cb4c5007f27f79df97cf` (unchanged — stub source untouched).
- Container: `vllm-arc:v0.30.0-b70-cert27` (fork_sha
  `d39ab5a26bf737d006d47a3f3125a404dca6e928`), bare container +
  .so/stub swap with in-container md5 verify.
- Patched-tree file md5s: kernel `fb599cf75883f96b081a1063d6c85c24`,
  reduce `26e88582edb3a19e80a763f178c2c3c1`.

## Gates (r231d_offline.sh; bank `/root/build/p647/r231d/`, local `bank/`)

- **C derivation** (slinfo): 12 walk-fit picks (idx 18/24/36/48 x
  {r224a mxk8v4, r229a mxfp8, r229b mxfp4}) all sl=5462-5469 ->
  blocks=342 -> **C=85 row-uniform**; 3 capacity picks (idx 1) sl
  8198-8201 -> C=128, capacity=yes.
- **G1 clamp==pin**: build25[ns:85] == build28[auto] bitwise —
  **PASS, 12/12 pairs BITWISE_EQUAL** (`oracle_cmp_G1.txt`).
- **G2 clamp no-op identity**: build25[ns:85] == build28[ns:85]
  bitwise — **PASS, 12/12** (`oracle_cmp_G2.txt`).
- **G3 capacity identity**: build25[auto] == build28[auto] @sl>=8192
  — **PASS, 3/3** (`oracle_cmp_G3.txt`).
- **r229 calibrated band cells** (same 12 walk-fit payloads, build28
  .so): **pass=12 fail=0**; worst NATIVE_VS_WALK dmax=1.47e-2
  mism=1/3072, NATIVE_VS_QMATCHED_ORACLE dmax<=3.125e-2 (inside the
  r229 bands).
- **Ladder** (r231a-identical first-file-per-bank inputs, single
  samples; AUTO = the engine posture):

| sl | mxk8v4 a->d | mxfp8 a->d | mxfp4 a->d | reading |
|---|---|---|---|---|
| 2048 | 0.517 -> 0.295 (−43%) | 0.510 -> 0.309 (−39%) | 0.552 -> 0.271 (−51%) | clamp selects 32 = the measured optimum class |
| 8192 | 0.667 -> 0.712 (+6.7%) | 0.698 -> 0.717 (+2.7%) | 0.643 -> 0.678 (+5.4%) | capacity path, dice band |
| 32768 | 1.213 -> 1.199 (−1.2%) | 1.285 -> 1.287 (+0.2%) | 1.011 -> 1.020 (+0.9%) | record path holds |
| 131072 | 3.494 -> 3.492 | 3.794 -> 3.764 | 2.723 -> 2.729 | flat (±0.8%) |
| 262144 | 6.616 -> 6.630 | 7.347 -> 7.280 | 5.064 -> 5.065 | flat (±0.9%) |

  (a = banked r231a build25 ladder `r231a/r231_splittime.txt`; d =
  build28 `r231d_ladder.txt`, ms.) Below 8K the pin columns >=32 also
  converge to the 32-split time — expected, the scalar pin is clamped
  (pin-path note above); ns1-ns8 cells are within ~1% of build25
  (clamp never bites there).

**Verdict**: the clamp is EXACTLY the pin semantics at an identical
partition (G1), output-neutral where it does not bite (G2) and on the
record path (G3 bitwise; ladder >=8K within the ±6% dice band), and
it recovers the measured shallow-ctx optimum dynamically (2K-class
auto −43/−39/−51%). Offline gates GREEN.

## e2e face

PENDING — next leg: record-parity faces (c8192/c32768, expect parity
— clamp is a no-op at >=8K) + short-ctx faces (c2048/c4096,
demonstrate the dynamic-serving win) on the build28 wheel in a
cert27 container, r230_cert27_itl.sh pattern, A/A per the
single-sample law.

## Hygiene

- W14 restored and HEALTH 200 after the leg (21:47:47 UTC); no arc
  containers left running; builder tree carries the patch
  (markers present, `.r231dprebak` siblings on the rig).
- Wheel: `/root/build/wheels-arc-vxk/…arc7mx….whl` is build28;
  `.r231dprebak` sibling preserves the pre-r231d (build25) wheel.
- Bank: `bank/` (logs, oracle cmps, ladder, slinfo, cells, md5s,
  out-dir manifest + tensor md5s); rig-side `out*` tensor dirs and
  the 1.5 G .so stay on the rig, pinned by md5.
