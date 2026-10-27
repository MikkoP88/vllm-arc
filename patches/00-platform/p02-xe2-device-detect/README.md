# p02 — Xe2 device detect + dual-B70 ZE_AFFINITY law

## What

- `is_xe2_arch()` in `vllm_gaudi/xpu/platform.py`: BMG (Xe2) gate on the PCI
  device-id range 0xE200–0xE2FF (Arc Pro B70 = 0xe223, rig-verified
  2026-10-27). Never raises — used inside the platform-resolution probe.
- `ArcXpuPlatform.check_and_update_config`: warns when TP>1 is requested and
  `ZE_AFFINITY_MASK` is unset (the validated dual-B70 posture sets
  `ZE_AFFINITY_MASK=0,1`; one Arc per socket, level-zero enumeration can
  silently degrade otherwise). Warn-only — operator env always respected.

## Why

The friendly device name is only "Intel(R) Graphics [0xe223]" and
`get_device_properties().architecture` is an opaque integer enum, so the
device-id range is the only stable Xe2 key (Appendix A hardware truth).
The ZE law is a rig law (Appendix F): silent wrong-card enumeration wastes
a leg.

## Validation

results/2026-10-27-p1-platform/ (P1 rig leg boot log carries the Xe2 posture
line; the ZE warning path is exercised by the unset-mask boot probe).
