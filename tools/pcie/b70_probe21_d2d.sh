#!/bin/bash
# b70_probe21_d2d.sh — p71 D2D peer-copy probe (owner directive
# 2026-10-31: "check do this solvable — driver/IOMMU refusing P2P DMA").
# Extends the rig's b70_probe* family with the one decisive measurement:
# effective GPU0<->GPU1 copy bandwidth (direct P2P vs host-bounce vs
# degraded link). IDLE WINDOW ONLY — never while a bench leg or soak is
# in flight (caution law, tools/pcie/README.md).
#
# IOMMU is already RULED OUT as a blocker on this rig (read-only checks
# 2026-10-31: /sys/kernel/iommu_groups empty, /sys/class/iommu empty,
# zero DMAR lines in dmesg, no intel_iommu= on the kernel cmdline, ACS
# fully disabled with 0 control bits set) — so the kernel/IOMMU leg of
# the hypothesis is dead; what remains is driver path + link reality,
# which this probe measures.
set -euo pipefail
IMG=vllm-arc:v0.30.0-b70-cert4
HERE=$(cd "$(dirname "$0")" && pwd)
PROBE="$HERE/d2d_probe.py"

echo "=== xpu-smi discovery (BDF + DRM mapping for the record) ==="
xpu-smi discovery

echo "=== sysfs link state (known xe misreport class — banked for the file) ==="
for d in 0000:b0:01.0 0000:b1:00.0 0000:d9:01.0 0000:da:00.0; do
  printf '%s cur=%s width=%s\n' "$d" \
    "$(cat /sys/bus/pci/devices/$d/current_link_speed)" \
    "$(cat /sys/bus/pci/devices/$d/current_link_width)"
done

echo "=== D2D copies (torch -> level-zero; see verdict table in d2d_probe.py) ==="
docker run --rm --privileged --network none --ipc host \
  -e ZE_AFFINITY_MASK=0,1 \
  -v "$PROBE":/root/d2d_probe.py:ro \
  --entrypoint /opt/venv/bin/python "$IMG" /root/d2d_probe.py

echo B70_PROBE21_D2D_DONE
