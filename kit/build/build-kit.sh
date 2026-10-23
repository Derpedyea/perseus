#!/usr/bin/env bash
# Builds the offline lab kit on a machine WITH internet (BUILD_JOBS=N caps compile jobs). Everything heavy lives in $WORK
# (default ~/.cache/perseus-kit), never in the repo.
#
#   build-kit.sh src                 clone/refresh llama.cpp
#   build-kit.sh binaries            static, old-CPU-safe binaries for every ISA variant (Docker),
#                                    plus a Vulkan build (nehalem) for the lab's GPU
#   build-kit.sh test                run the binaries under QEMU as Penryn/Nehalem CPUs, incl. negative controls
#   build-kit.sh models [TIER]       download + sha256-verify models up to TIER (diag|core|full), resumable
#   build-kit.sh assemble DEST [TIER]  write DEST/PERSEUS-KIT (scripts, binaries, GPU op lists, models, MANIFEST.json)
#   build-kit.sh zip OUT.zip         the diag kit (assemble ... diag) as one zip: PERSEUS-KIT/ at the top
# Tiers: diag = no models (machine facts, memory bandwidth, CPU kernel + GPU checks; FAT32 is fine),
#        core = + the tiny Qwen3.5-0.8B pair (1.0 GB of models), full = + Qwen3.8-Flash-Next IQ3_XXS (75.8 GB)
#        and Flint UD-Q4_K_XL (25.1 GB).
#        Default: full.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNELS="$(cd "$KIT/../kernels" && pwd)"
WORK="${PERSEUS_KIT_WORK:-$HOME/.cache/perseus-kit}"
IMAGE=perseus-kit-build:bookworm
IMAGE_VK=perseus-kit-build-vk:noble
FORKS="stock"
VARIANTS="penryn nehalem haswell"
TARGETS="llama-bench llama-completion llama-speculative-simple llama-cli"
# The Vulkan build ships these (llama-server: phase 6); test-export-graph-ops stays here (it writes
# the GPU op lists).
VK_TARGETS="llama-bench llama-completion llama-server test-backend-ops"
# Models whose graphs the GPU self-test (step 3-02) replays: what the lab runs on the GPU.
GPU_OP_MODELS="Qwen3.5-0.8B-Q4_0.gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf flint-UD-Q4_K_XL.gguf"

die() { echo "build-kit: $*" >&2; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

tier_rank() { case "$1" in diag) echo 0 ;; core) echo 1 ;; full) echo 2 ;; *) die "unknown tier '$1' (diag|core|full)" ;; esac; }

# Emits "tier runner file size sha url" rows at or below TIER.
model_rows() {
  local max; max=$(tier_rank "$1")
  grep -v '^#' "$KIT/models.tsv" | while IFS=$'\t' read -r tier runner file size sha url; do
    [ -n "$tier" ] || continue
    [ "$(tier_rank "$tier")" -le "$max" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tier" "$runner" "$file" "$size" "$sha" "$url"
  done
}

# Memory-capped (no swap) so a runaway test can't push the host into swap.
in_image() { # image script
  docker run --rm --memory=12g --memory-swap=12g -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$WORK:/work" -v "$KIT:/kit:ro" "$1" bash -euo pipefail -c "$2"
}
in_docker() { in_image "$IMAGE" "$1"; }

cmd_src() {
  mkdir -p "$WORK/src"
  local name url branch
  for spec in "stock https://github.com/ggml-org/llama.cpp.git master"; do
    read -r name url branch <<<"$spec"
    local dir="$WORK/src/llama.cpp-$name"
    if [ -d "$dir/.git" ]; then
      # the perseus patches (cmd_binaries re-applies them) would block the checkout
      git -C "$dir" fetch -q --depth 1 origin "$branch" && git -C "$dir" checkout -q -f FETCH_HEAD
    else
      git clone -q --depth 1 --branch "$branch" "$url" "$dir"
    fi
    log "$name @ $(git -C "$dir" rev-parse --short HEAD) ($(git -C "$dir" log -1 --format=%cs))"
  done
}

isa_flags() {
  local off="-DGGML_SSE42=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_BMI2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF"
  case "$1" in
    penryn)  echo "$off" ;;
    nehalem) echo "${off/-DGGML_SSE42=OFF/-DGGML_SSE42=ON}" ;;
    haswell) echo "-DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_BMI2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF" ;;
    *) die "unknown variant $1" ;;
  esac
}

# GCC has no -march=penryn; Penryn = Core 2 + SSE4.1.
march_flags() {
  case "$1" in
    penryn) echo "-march=core2 -msse4.1 -mtune=generic" ;;
    *) echo "-march=$1 -mtune=generic" ;;
  esac
}

cmd_binaries() {
  [ -d "$WORK/src/llama.cpp-stock" ] || die "run '$0 src' first"
  log "building Docker image $IMAGE"
  docker build -q -t "$IMAGE" "$KIT/build" >/dev/null
  mkdir -p "$WORK/bin" "$WORK/build"
  # perseus patches for pre-AVX CPUs (kernels/), applied to a clean copy of the files they touch
  local stock="$WORK/src/llama.cpp-stock"
  git -C "$stock" checkout -- ggml/src/ggml-cpu/arch/x86/quants.c ggml/src/ggml-cpu/arch-fallback.h ggml/src/ggml-cpu/quants.h ggml/src/ggml-cpu/ggml-cpu.c ggml/src/ggml-cpu/repack.cpp ggml/src/ggml-cpu/tiled ggml/src/ggml-vulkan/ggml-vulkan.cpp
  python3 -I "$KERNELS/apply_ssse3.py" "$stock" || die "could not apply the SSSE3 kernels"
  python3 -I "$KERNELS/apply_ssse3_fn.py" "$stock" || die "could not apply the Flash-Next SSSE3 kernels"
  python3 -I "$KERNELS/apply_tiled_sse.py" "$stock" || die "could not enable the tiled matmul on SSE4.1"
  python3 -I "$KIT/build/patch_vulkan.py" "$stock" || die "could not patch ggml-vulkan's perf logger"
  log "membw (static, baseline x86-64)"
  in_docker "mkdir -p /work/bin/tools && gcc -O2 -Wall -Wextra -Werror -static -pthread -march=x86-64 -mtune=generic -o /work/bin/tools/membw /kit/src/membw.c"
  for fork in $FORKS; do
    for v in $VARIANTS; do
      log "llama.cpp $fork / $v"
      in_docker "
        cmake -S /work/src/llama.cpp-$fork -B /work/build/$fork-$v -DCMAKE_BUILD_TYPE=Release \
          -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF -DGGML_OPENMP=OFF \
          -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
          -DCMAKE_EXE_LINKER_FLAGS=-static -DCMAKE_C_FLAGS='$(march_flags "$v")' -DCMAKE_CXX_FLAGS='$(march_flags "$v")' \
          $(isa_flags "$v") > /work/build/$fork-$v.configure.log 2>&1 || { tail -30 /work/build/$fork-$v.configure.log; exit 1; }
        cmake --build /work/build/$fork-$v -j${BUILD_JOBS:-\$(nproc)} --target $TARGETS > /work/build/$fork-$v.build.log 2>&1 || { tail -40 /work/build/$fork-$v.build.log; exit 1; }
        mkdir -p /work/bin/$fork/$v
        for t in $TARGETS; do cp /work/build/$fork-$v/bin/\$t /work/bin/$fork/$v/; strip /work/bin/$fork/$v/\$t; done
        file /work/bin/$fork/$v/llama-bench | grep -q 'statically linked' || { echo 'not static'; exit 1; }
      "
    done
  done
  build_vulkan
  log "kernel self-test (SSSE3 kernels vs llama.cpp's generic reference)"
  "$KERNELS/build-test.sh" $VARIANTS
  for v in $VARIANTS; do
    cp "$WORK/test/test_ssse3_kernels-$v" "$WORK/bin/stock/$v/test-ssse3-kernels"
    cp "$WORK/test/test-fn-kernels-$v" "$WORK/bin/stock/$v/test-fn-kernels"
    cp "$WORK/test/test-tiled-mulmat-$v" "$WORK/bin/stock/$v/test-tiled-mulmat"
    cp "$WORK/test/test-lut-$v" "$WORK/bin/stock/$v/test-lut"
    if [ "$v" != haswell ]; then cp "$WORK/test/test-repack-mulmat-$v" "$WORK/bin/stock/$v/test-repack-mulmat"; fi
  done
  log "binaries in $WORK/bin"
}

# The lab's GPU (AMD RX 580) through llama.cpp's Vulkan backend, with the same perseus CPU patches
# (routed experts run on the CPU). Built on Ubuntu 24.04, the lab's OS: Vulkan's loader is
# dlopen-based, so glibc is linked dynamically (2.39, the lab's version); libstdc++ is static.
# The Khronos loader ships next to the binaries (rpath $ORIGIN/lib), so a box without libvulkan1
# still starts them; the GPU driver itself (Mesa's RADV) must be on the box. glibc-min records the
# newest glibc symbol version they use: run.sh skips the GPU steps on a box with an older glibc.
build_vulkan() {
  log "llama.cpp stock / nehalem + Vulkan"
  docker build -q -t "$IMAGE_VK" "$KIT/build/vulkan" >/dev/null
  in_image "$IMAGE_VK" "
    cmake -S /work/src/llama.cpp-stock -B /work/build/stock-vk-nehalem -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF -DGGML_OPENMP=OFF -DGGML_VULKAN=ON \
      -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
      '-DCMAKE_EXE_LINKER_FLAGS=-static-libstdc++ -static-libgcc' -DCMAKE_C_FLAGS='$(march_flags nehalem)' -DCMAKE_CXX_FLAGS='$(march_flags nehalem)' \
      $(isa_flags nehalem) > /work/build/stock-vk-nehalem.configure.log 2>&1 || { tail -30 /work/build/stock-vk-nehalem.configure.log; exit 1; }
    cmake --build /work/build/stock-vk-nehalem -j${BUILD_JOBS:-\$(nproc)} --target $VK_TARGETS test-export-graph-ops > /work/build/stock-vk-nehalem.build.log 2>&1 || { tail -40 /work/build/stock-vk-nehalem.build.log; exit 1; }
    d=/work/bin/stock-vk/nehalem
    rm -rf \$d && mkdir -p \$d/lib
    for t in $VK_TARGETS; do
      # llama-completion keeps its symbols (+3 MB): run.sh's CPU profile (perf, step 4b-09) names functions
      cp /work/build/stock-vk-nehalem/bin/\$t \$d/ && { [ \$t = llama-completion ] || strip \$d/\$t; } && patchelf --set-rpath '\$ORIGIN/lib' \$d/\$t
      need=\$(objdump -p \$d/\$t | awk '/NEEDED/ {print \$2}' | sort | tr '\n' ' ')
      [ \"\$need\" = 'ld-linux-x86-64.so.2 libc.so.6 libm.so.6 libvulkan.so.1 ' ] || { echo \"\$t needs: \$need\"; exit 1; }
    done
    cp -L /usr/lib/x86_64-linux-gnu/libvulkan.so.1 \$d/lib/
    for f in $VK_TARGETS lib/libvulkan.so.1; do objdump -T \$d/\$f; done | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -1 > \$d/glibc-min
    grep -qE '^2\\.[0-9]+\$' \$d/glibc-min || { echo \"can't read the glibc version the Vulkan build needs\"; exit 1; }
  "
}

# True if FILE and, for a split model (NAME-00001-of-0000N.gguf), every one of its N parts are
# downloaded and verified: llama.cpp opens them all, even for metadata.
model_downloaded() { # file
  local f="$1" n k
  case "$f" in
    *-00001-of-[0-9][0-9][0-9][0-9][0-9].gguf)
      n=${f%.gguf}; n=$((10#${n##*-of-}))
      for k in $(seq 1 "$n"); do
        [ -f "$WORK/models/$(printf '%s-%05d-of-%05d.gguf' "${f%-00001-of-*}" "$k" "$n").sha256ok" ] || return 1
      done ;;
    *) [ -f "$WORK/models/$f.sha256ok" ] ;;
  esac
}

# Op lists for the GPU self-test (step 3-02): every op the model's prompt (32-token batches) and
# decode graphs run, with shapes and types, from llama.cpp's test-export-graph-ops (it reads the
# model's metadata only, no weights). The lab replays them on its GPU against its CPU with
# test-backend-ops --test-file. kit/build/gpu_ops.py leaves out the ops the replay can't judge.
# Writes straight into the kit being assembled; scratch names carry this process's pid, so
# assembles running side by side (the e2e suites) don't share files.
gpu_ops_into() { # dest_dir
  [ -x "$WORK/build/stock-vk-nehalem/bin/test-export-graph-ops" ] || die "no Vulkan build; run '$0 binaries'"
  rm -rf "$1" && mkdir -p "$1"
  local f raw="ops-raw.$$.txt"
  for f in $GPU_OP_MODELS; do
    if ! model_downloaded "$f"; then log "gpu-ops: $f (or a file split from it) not downloaded, the GPU self-test won't cover it"; continue; fi
    in_image "$IMAGE_VK" "/work/build/stock-vk-nehalem/bin/test-export-graph-ops -m /work/models/$f -c 4096 -ub 32 -o /work/build/$raw > /work/build/$raw.log 2>&1 || { tail -20 /work/build/$raw.log; exit 1; }"
    python3 -I "$KIT/build/gpu_ops.py" "$WORK/src/llama.cpp-stock/ggml/include/ggml.h" "$WORK/build/$raw" "$1/${f%.gguf}.txt" || die "gpu_ops.py failed on $f"
    rm -f "$WORK/build/$raw" "$WORK/build/$raw.log"
  done
}

# Runs one binary under an emulated CPU; prints the exit code. 132 = SIGILL.
qemu_exit() { # cpu binary args...
  local cpu="$1"; shift
  in_docker "qemu-x86_64 -cpu $cpu $* >/dev/null 2>&1 && echo 0 || echo \$?" | tail -1
}
# Same, in the Vulkan build's image (its binaries need Ubuntu 24.04's glibc).
qemu_exit_vk() { # cpu binary args...
  local cpu="$1"; shift
  in_image "$IMAGE_VK" "qemu-x86_64 -cpu $cpu $* >/dev/null 2>&1 && echo 0 || echo \$?" | tail -1
}

cmd_test() {
  local f
  for f in Qwen3.5-0.8B-Q4_0 Qwen3.5-0.8B-Q4_K_M; do
    [ -f "$WORK/models/$f.gguf" ] || die "need $f.gguf (run '$0 models core')"
  done
  local fails=0
  check() { # label expected actual
    if [ "$2" = "$3" ]; then log "ok    $1 (exit $3)"; else log "FAIL  $1: expected exit $2, got $3"; fails=$((fails + 1)); fi
  }
  bench() { echo "-m /work/models/$1.gguf -p 4 -n 2 -r 1 -t 4 -o jsonl"; }

  # 1) Is QEMU's CPU model strict? One instruction per extension, inside and outside each model.
  in_docker "mkdir -p /work/test && gcc -O0 -static -o /work/test/isa_probe /kit/build/isa_probe.c"
  check "QEMU Penryn runs SSSE3"     0   "$(qemu_exit Penryn /work/test/isa_probe ssse3)"
  check "QEMU Penryn runs SSE4.1"    0   "$(qemu_exit Penryn /work/test/isa_probe sse41)"
  check "QEMU Penryn traps SSE4.2"   132 "$(qemu_exit Penryn /work/test/isa_probe sse42)"
  check "QEMU Penryn traps POPCNT"   132 "$(qemu_exit Penryn /work/test/isa_probe popcnt)"
  check "QEMU Penryn traps AVX"      132 "$(qemu_exit Penryn /work/test/isa_probe avx)"
  check "QEMU Nehalem runs SSE4.2"   0   "$(qemu_exit Nehalem /work/test/isa_probe sse42)"
  check "QEMU Nehalem runs POPCNT"   0   "$(qemu_exit Nehalem /work/test/isa_probe popcnt)"
  check "QEMU Nehalem traps AVX"     132 "$(qemu_exit Nehalem /work/test/isa_probe avx)"
  check "QEMU Nehalem traps AVX2"    132 "$(qemu_exit Nehalem /work/test/isa_probe avx2)"
  # 2) A real binary built for a newer CPU must trap (proves the variants differ).
  check "haswell llama-bench traps on Nehalem" 132 "$(qemu_exit Nehalem /work/bin/stock/haswell/llama-bench $(bench Qwen3.5-0.8B-Q4_0))"
  # 3) The shipped builds run every kernel the kit exercises on the oldest CPU they target.
  check "membw topo on Penryn" 0 "$(qemu_exit Penryn /work/bin/tools/membw topo)"
  check "membw read on Penryn" 0 "$(qemu_exit Penryn /work/bin/tools/membw read --threads 2 --mb 64 --secs 0.2 --reps 1)"
  check "SSSE3 kernels match reference on Penryn"  0 "$(qemu_exit Penryn /work/bin/stock/penryn/test-ssse3-kernels 20)"
  check "SSSE3 kernels match reference on Nehalem" 0 "$(qemu_exit Nehalem /work/bin/stock/nehalem/test-ssse3-kernels 20)"
  check "Flash-Next kernels match reference on Penryn"  0 "$(qemu_exit Penryn /work/bin/stock/penryn/test-fn-kernels 20)"
  check "Flash-Next kernels match reference on Nehalem" 0 "$(qemu_exit Nehalem /work/bin/stock/nehalem/test-fn-kernels 20)"
  check "tiled matmul matches reference on Penryn"  0 "$(qemu_exit Penryn /work/bin/stock/penryn/test-tiled-mulmat)"
  check "tiled matmul matches reference on Nehalem" 0 "$(qemu_exit Nehalem /work/bin/stock/nehalem/test-tiled-mulmat)"
  check "IQ3_S repack matches reference on Penryn"  0 "$(qemu_exit Penryn /work/bin/stock/penryn/test-repack-mulmat)"
  check "IQ3_S repack matches reference on Nehalem" 0 "$(qemu_exit Nehalem /work/bin/stock/nehalem/test-repack-mulmat)"
  check "lookup-table kernels exact on Penryn"      0 "$(qemu_exit Penryn /work/bin/stock/penryn/test-lut)"
  local v
  for v in penryn:Penryn nehalem:Nehalem; do
    local build="${v%%:*}" cpu="${v##*:}"
    check "stock $build Q4_0 bench on $cpu"   0 "$(qemu_exit "$cpu" /work/bin/stock/$build/llama-bench $(bench Qwen3.5-0.8B-Q4_0))"
    check "stock $build Q4_K_M bench on $cpu" 0 "$(qemu_exit "$cpu" /work/bin/stock/$build/llama-bench $(bench Qwen3.5-0.8B-Q4_K_M))"
    check "stock $build completion on $cpu"   0 "$(qemu_exit "$cpu" /work/bin/stock/$build/llama-completion -m /work/models/Qwen3.5-0.8B-Q4_0.gguf -p hello -n 4 --temp 0 -no-cnv -t 4)"
  done
  # 4) The Vulkan build's CPU code (its own toolchain) on the lab's CPU generation; no GPU here,
  # so -dev none: every op runs on the CPU, perseus kernels included.
  check "Vulkan build, CPU only, on Nehalem" 0 "$(qemu_exit_vk Nehalem /work/bin/stock-vk/nehalem/llama-completion -m /work/models/Qwen3.5-0.8B-Q4_0.gguf -dev none -p hello -n 4 --temp 0 -no-cnv -t 4)"
  [ "$fails" -eq 0 ] || die "$fails QEMU check(s) failed"
  log "all QEMU checks passed"
}

cmd_models() {
  local tier="${1:-full}"
  [ "$(tier_rank "$tier")" -gt 0 ] || { log "tier $tier carries no models: nothing to download"; return 0; }
  mkdir -p "$WORK/models"
  local failed=0
  while IFS=$'\t' read -r _tier _runner file size sha url; do
    local dst="$WORK/models/$file"
    if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$size" ] && [ -f "$dst.sha256ok" ]; then
      log "have  $file"; continue
    fi
    log "fetch $file ($((size / 1000000)) MB)"
    curl -fsSL --retry 5 --retry-delay 5 -C - -o "$dst.part" "$url" || { log "FAIL download $file"; failed=$((failed + 1)); continue; }
    if [ "$(stat -c %s "$dst.part")" != "$size" ]; then log "FAIL size $file"; failed=$((failed + 1)); continue; fi
    if [ "$(sha256sum "$dst.part" | cut -d' ' -f1)" != "$sha" ]; then
      log "FAIL sha256 $file (deleted, re-run to retry)"; rm -f "$dst.part"; failed=$((failed + 1)); continue
    fi
    mv "$dst.part" "$dst" && touch "$dst.sha256ok"
    log "ok    $file"
  done < <(model_rows "$tier")
  [ "$failed" -eq 0 ] || die "$failed model(s) failed"
}

cmd_assemble() {
  local dest="${1:-}" tier="${2:-full}"
  [ -n "$dest" ] && [ -d "$dest" ] || die "usage: $0 assemble DEST_DIR [diag|core|full] (DEST must exist, e.g. the mounted USB)"
  tier_rank "$tier" >/dev/null
  [ -d "$WORK/bin/stock" ] || die "no binaries in $WORK/bin; run '$0 binaries'"
  local out="$dest/PERSEUS-KIT"
  local need=0 big=0
  while IFS=$'\t' read -r _t _r file size _s _u; do
    [ -f "$WORK/models/$file.sha256ok" ] || die "model $file missing or unverified; run '$0 models $tier'"
    need=$((need + size))
    [ "$size" -lt 4294967296 ] || big=1
  done < <(model_rows "$tier")
  # Only what the kit runs: $WORK/bin may still hold builds of forks the kit no longer uses.
  local bins="$FORKS stock-vk tools" b
  # + binaries + results: run.sh refuses to start with under 200 MB free; a full run's results are a few MB
  need=$((need + $(cd "$WORK/bin" && du -scb $bins 2>/dev/null | tail -1 | cut -f1) + 300000000))
  local fs; fs=$(stat -f -c %T "$dest")
  # FAT32 can't hold files over 4 GB (Flash-Next's). Nothing executes from the USB (run.sh copies
  # bin/ to the local disk and chmods it there), so udisks' vfat default 'showexec' (no +x on
  # extensionless files) doesn't matter: kit/test/e2e-diag.sh.
  case "$fs" in msdos|vfat|fat) [ "$big" = 0 ] || die "$dest is FAT ($fs): files over 4 GB won't fit. Format the USB as exFAT or ext4 (or assemble the diag or core tier)." ;; esac
  local free; free=$(( $(stat -f -c '%a * %S' "$dest") ))
  [ "$free" -ge "$need" ] || die "$dest has $((free / 1000000)) MB free, kit needs $((need / 1000000)) MB"

  log "writing $out ($((need / 1000000)) MB, tier $tier, fs $fs)"
  mkdir -p "$out/models" "$out/bin" "$out/lib" "$out/src"
  cp "$KIT/run.sh" "$KIT/README.txt" "$KIT/models.tsv" "$out/"
  cp "$KIT/lib/"*.sh "$KIT/lib/"*.py "$out/lib/"
  cp "$KIT/src/membw.c" "$out/src/"
  for b in $bins; do if [ -d "$WORK/bin/$b" ]; then cp -r "$WORK/bin/$b" "$out/bin/"; fi; done
  gpu_ops_into "$out/gpu-ops"
  while IFS=$'\t' read -r _t _r file size _s _u; do
    if [ -f "$out/models/$file" ] && [ "$(stat -c %s "$out/models/$file")" = "$size" ]; then log "have  $file"; continue; fi
    log "copy  $file"
    cp "$WORK/models/$file" "$out/models/$file.part" && mv "$out/models/$file.part" "$out/models/$file"
  done < <(model_rows "$tier")
  local extra; extra=$(comm -23 <(ls "$out/models" | grep -v '\.part$' | sort) <(model_rows "$tier" | cut -f3 | sort) | tr '\n' ' ')
  [ -z "$extra" ] || log "note: $out/models also holds $extra(left in place from an earlier, bigger kit; run.sh uses them)"
  {
    printf '{\n  "built_at": "%s",\n  "tier": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$tier"
    printf '  "stock_commit": "%s",\n' "$(git -C "$WORK/src/llama.cpp-stock" rev-parse HEAD)"
    printf '  "stock_patch": "kernels/apply_ssse3.py sha256 %s, kernels/apply_ssse3_fn.py sha256 %s, kernels/apply_tiled_sse.py sha256 %s, kit/build/patch_vulkan.py sha256 %s",\n' \
      "$(sha256sum "$KERNELS/apply_ssse3.py" | cut -d' ' -f1)" "$(sha256sum "$KERNELS/apply_ssse3_fn.py" | cut -d' ' -f1)" \
      "$(sha256sum "$KERNELS/apply_tiled_sse.py" | cut -d' ' -f1)" "$(sha256sum "$KIT/build/patch_vulkan.py" | cut -d' ' -f1)"
    local dirty=""; [ -n "$(git -C "$KIT" status --porcelain -- . 2>/dev/null)" ] && dirty="-dirty"
    printf '  "kit_commit": "%s%s",\n' "$(git -C "$KIT" rev-parse --short HEAD 2>/dev/null || echo none)" "$dirty"
    printf '  "variants": "%s"\n}\n' "$VARIANTS"
  } > "$out/MANIFEST.json"
  sync
  log "done. Safely eject the USB before unplugging."
}

# Nothing in the kit relies on exec bits or symlinks (run.sh runs under 'bash' and copies bin/ to
# the local disk before chmod-ing it), so any unzip tool onto any filesystem works.
cmd_zip() {
  local zip="${1:-}"
  [ -n "$zip" ] || die "usage: $0 zip OUT.zip"
  [ ! -e "$zip" ] || die "$zip exists; remove it first"
  zip="$(cd "$(dirname "$zip")" && pwd)/$(basename "$zip")"
  ZIP_TMP=$(mktemp -d "$WORK/zip.XXXX")  # global: the EXIT trap runs after this function returns
  trap 'rm -rf "$ZIP_TMP"' EXIT
  cmd_assemble "$ZIP_TMP" diag
  (cd "$ZIP_TMP" && zip -q -r -X "$zip" PERSEUS-KIT) || die "zip failed"
  unzip -tq "$zip" || die "unzip -t failed on $zip"
  log "$zip: $(( $(stat -c %s "$zip") / 1000000 )) MB, sha256 $(sha256sum "$zip" | cut -d' ' -f1)"
}

case "${1:-}" in
  src) cmd_src ;;
  binaries) cmd_binaries ;;
  test) cmd_test ;;
  models) cmd_models "${2:-full}" ;;
  assemble) shift; cmd_assemble "$@" ;;
  zip) cmd_zip "${2:-}" ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
