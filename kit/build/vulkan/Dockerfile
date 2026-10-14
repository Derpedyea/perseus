# Vulkan build of llama.cpp for the lab box's GPU (an AMD RX 580). Ubuntu 24.04, the lab box's
# OS: the binaries link glibc 2.39 dynamically (Vulkan's loader is dlopen-based, which a static
# glibc can't do) and must not need a newer glibc than the lab has. Packages per llama.cpp's
# docs/build.md ("libvulkan-dev glslc spirv-headers").
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates file binutils \
      libvulkan-dev libvulkan1 glslc spirv-headers patchelf qemu-user \
 && rm -rf /var/lib/apt/lists/*
