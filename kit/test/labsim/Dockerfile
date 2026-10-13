# Stand-in for the lab box's OS for the GPU e2e (kit/test/e2e-gpu.sh): Ubuntu 24.04 with Mesa's
# Vulkan drivers (RADV), as a desktop install has them. The GPU comes from the host via --device.
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends mesa-vulkan-drivers pciutils acl procps \
 && rm -rf /var/lib/apt/lists/*
