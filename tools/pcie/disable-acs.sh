#!/bin/bash
# disable-acs.sh - Disable ACS on all PCIe devices (P2P DMA enablement)
# Source: Dell PowerEdge 16th-Gen setpci practice (ECAP_ACS+0x6.w=0000).
# Deployed on rig REDACTED-HOST (10.20.3.REDACTED) 2026-10-31 for dual Arc Pro B70 P2P.
# Idempotent; the systemd unit disable-acs.service re-applies it at every
# boot (config-space writes do not persist across reboots).
# Repo copy = exact bytes of /usr/local/sbin/disable-acs.sh on the rig.
if [ "$EUID" -ne 0 ]; then
  echo "ERROR: $0 must be run as root"
  exit 1
fi
changed=0
skipped=0
for BDF in $(lspci -d "*:*:*" | awk '{print $1}'); do
  # skip if it doesn't support ACS
  setpci -s ${BDF} ECAP_ACS+0x6.w > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    skipped=$((skipped+1))
    continue
  fi
  cur=$(setpci -s ${BDF} ECAP_ACS+0x6.w 2>/dev/null)
  setpci -s ${BDF} ECAP_ACS+0x6.w=0000
  echo "ACS disabled on ${BDF} (was ${cur})"
  changed=$((changed+1))
done
echo "disable-acs: ${changed} ACS devices zeroed, ${skipped} without ACS skipped"
left=""
for BDF in $(lspci -d "*:*:*" | awk '{print $1}'); do
  v=$(setpci -s ${BDF} ECAP_ACS+0x6.w 2>/dev/null) || continue
  [ "$v" != "0000" ] && left="${left} ${BDF}=${v}"
done
if [ -n "$left" ]; then
  echo "WARNING: non-zero ACSCtl remains:${left}"
else
  echo "verify: all ACSCtl = 0000"
fi
exit 0
