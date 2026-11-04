#!/bin/bash
# p619e — build the cert8 image from the tag archive.
set -euo pipefail
bash /root/build/build_image.sh /root/build/cert8_p619.tar \
  e47739b018411fb72d66c5499701b430c5a302e2 vllm-arc:v0.30.0-b70-cert8
