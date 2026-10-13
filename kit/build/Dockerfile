# Build + test image for the lab kit. Debian's static libc/libstdc++ are compiled for
# baseline x86-64, unlike this CachyOS host's v3 packages, so static binaries built here
# carry no AVX2 from the libraries. qemu-user lets us run them as a Penryn/Nehalem CPU.
FROM debian:bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates qemu-user file binutils \
 && rm -rf /var/lib/apt/lists/*
