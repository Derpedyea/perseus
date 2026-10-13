#!/usr/bin/env bash
# Builds the offline lab kit on a machine WITH internet. Everything heavy lives in $WORK
# (default ~/.cache/perseus-kit), never in the repo.
#
#   build-kit.sh src                 clone/refresh llama.cpp (stock) + PrismML fork (prism branch)
#   build-kit.sh binaries            static, old-CPU-safe binaries for every fork x ISA variant (Docker)
#   build-kit.sh test                run the binaries under QEMU as Penryn/Nehalem CPUs, incl. negative controls
#   build-kit.sh models [TIER]       download + sha256-verify models up to TIER (core|standard), resumable
#   build-kit.sh assemble DEST [TIER]  write DEST/PERSEUS-KIT (scripts, binaries, models, MANIFEST.json)
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNELS="$(cd "$KIT/../kernels" && pwd)"
WORK="${PERSEUS_KIT_WORK:-$HOME/.cache/perseus-kit}"
IMAGE=perseus-kit-build:bookworm
FORKS="stock prism"
VARIANTS="penryn nehalem haswell"
TARGETS="llama-bench llama-completion llama-speculative-simple llama-cli"

die() { echo "build-kit: $*" >&2; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

tier_rank() { case "$1" in core) echo 1 ;; standard) echo 2 ;; *) die "unknown tier '$1' (core|standard)" ;; esac; }

# Emits "tier runner file size sha url" rows at or below TIER.
model_rows() {
  local max; max=$(tier_rank "$1")
  grep -v '^#' "$KIT/models.tsv" | while IFS=$'\t' read -r tier runner file size sha url; do
    [ -n "$tier" ] || continue
    [ "$(tier_rank "$tier")" -le "$max" ] && printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tier" "$runner" "$file" "$size" "$sha" "$url"
  done
}

# Memory-capped (no swap) so a runaway test can't push the host into swap.
in_docker() {
  docker run --rm --memory=10g --memory-swap=10g -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$WORK:/work" -v "$KIT:/kit:ro" "$IMAGE" bash -euo pipefail -c "$1"
}

cmd_src() {
  mkdir -p "$WORK/src"
  local name url branch
  for spec in "stock https://github.com/ggml-org/llama.cpp.git master" "prism https://github.com/PrismML-Eng/llama.cpp.git prism"; do
    read -r name url branch <<<"$spec"
    local dir="$WORK/src/llama.cpp-$name"
    if [ -d "$dir/.git" ]; then
      git -C "$dir" fetch -q --depth 1 origin "$branch" && git -C "$dir" checkout -q FETCH_HEAD
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
  [ -d "$WORK/src/llama.cpp-stock" ] && [ -d "$WORK/src/llama.cpp-prism" ] || die "run '$0 src' first"
  log "building Docker image $IMAGE"
  docker build -q -t "$IMAGE" "$KIT/build" >/dev/null
  mkdir -p "$WORK/bin" "$WORK/build"
  # perseus patches for pre-AVX CPUs (kernels/), applied to a clean copy of the files they touch
  local stock="$WORK/src/llama.cpp-stock"
  git -C "$stock" checkout -- ggml/src/ggml-cpu/arch/x86/quants.c ggml/src/ggml-cpu/quants.h ggml/src/ggml-cpu/ggml-cpu.c ggml/src/ggml-cpu/repack.cpp ggml/src/ggml-cpu/tiled
  python3 -I "$KERNELS/apply_ssse3.py" "$stock" || die "could not apply the SSSE3 kernels"
  python3 -I "$KERNELS/apply_tiled_sse.py" "$stock" || die "could not enable the tiled matmul on SSE4.1"
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
        cmake --build /work/build/$fork-$v -j\$(nproc) --target $TARGETS > /work/build/$fork-$v.build.log 2>&1 || { tail -40 /work/build/$fork-$v.build.log; exit 1; }
        mkdir -p /work/bin/$fork/$v
        for t in $TARGETS; do cp /work/build/$fork-$v/bin/\$t /work/bin/$fork/$v/; strip /work/bin/$fork/$v/\$t; done
        file /work/bin/$fork/$v/llama-bench | grep -q 'statically linked' || { echo 'not static'; exit 1; }
      "
    done
  done
  log "kernel self-test (SSSE3 kernels vs llama.cpp's generic reference)"
  "$KERNELS/build-test.sh" $VARIANTS
  for v in $VARIANTS; do
    cp "$WORK/test/test_ssse3_kernels-$v" "$WORK/bin/stock/$v/test-ssse3-kernels"
    cp "$WORK/test/test-tiled-mulmat-$v" "$WORK/bin/stock/$v/test-tiled-mulmat"
  done
  log "binaries in $WORK/bin"
}

# Runs one binary under an emulated CPU; prints the exit code. 132 = SIGILL.
qemu_exit() { # cpu binary args...
  local cpu="$1"; shift
  in_docker "qemu-x86_64 -cpu $cpu $* >/dev/null 2>&1 && echo 0 || echo \$?" | tail -1
}

cmd_test() {
  local f
  for f in Qwen3.5-0.8B-Q4_0 Qwen3.5-0.8B-Q4_K_M Ternary-Bonsai-2-27B-PQ2_0 Ternary-Bonsai-2-27B-PTQ1_0 Underdog-Saluki-27B-1.0-IQ2-mix; do
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
  check "tiled matmul matches reference on Penryn"  0 "$(qemu_exit Penryn /work/bin/stock/penryn/test-tiled-mulmat)"
  check "tiled matmul matches reference on Nehalem" 0 "$(qemu_exit Nehalem /work/bin/stock/nehalem/test-tiled-mulmat)"
  local v
  for v in penryn:Penryn nehalem:Nehalem; do
    local build="${v%%:*}" cpu="${v##*:}"
    check "stock $build Q4_0 bench on $cpu"   0 "$(qemu_exit "$cpu" /work/bin/stock/$build/llama-bench $(bench Qwen3.5-0.8B-Q4_0))"
    check "stock $build Q4_K_M bench on $cpu" 0 "$(qemu_exit "$cpu" /work/bin/stock/$build/llama-bench $(bench Qwen3.5-0.8B-Q4_K_M))"
    check "stock $build completion on $cpu"   0 "$(qemu_exit "$cpu" /work/bin/stock/$build/llama-completion -m /work/models/Qwen3.5-0.8B-Q4_0.gguf -p hello -n 4 --temp 0 -no-cnv -t 4)"
    check "prism $build PQ2_0 bench on $cpu"  0 "$(qemu_exit "$cpu" /work/bin/prism/$build/llama-bench $(bench Ternary-Bonsai-2-27B-PQ2_0))"
    check "prism $build PTQ1_0 bench on $cpu" 0 "$(qemu_exit "$cpu" /work/bin/prism/$build/llama-bench $(bench Ternary-Bonsai-2-27B-PTQ1_0))"
    check "stock $build Saluki IQ-mix bench on $cpu" 0 "$(qemu_exit "$cpu" /work/bin/stock/$build/llama-bench $(bench Underdog-Saluki-27B-1.0-IQ2-mix))"
  done
  [ "$fails" -eq 0 ] || die "$fails QEMU check(s) failed"
  log "all QEMU checks passed"
}

cmd_models() {
  local tier="${1:-standard}"
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
  local dest="${1:-}" tier="${2:-standard}"
  [ -n "$dest" ] && [ -d "$dest" ] || die "usage: $0 assemble DEST_DIR [core|standard] (DEST must exist, e.g. the mounted USB)"
  local out="$dest/PERSEUS-KIT"
  local need=0
  while IFS=$'\t' read -r _t _r file size _s _u; do
    [ -f "$WORK/models/$file.sha256ok" ] || die "model $file missing or unverified; run '$0 models $tier'"
    need=$((need + size))
  done < <(model_rows "$tier")
  need=$((need + 1500000000))  # binaries + room for results
  local fs; fs=$(stat -f -c %T "$dest")
  case "$fs" in msdos|vfat|fat) die "$dest is FAT ($fs): files over 4 GB won't fit. Format the USB as exFAT or ext4." ;; esac
  local free; free=$(( $(stat -f -c '%a * %S' "$dest") ))
  [ "$free" -ge "$need" ] || die "$dest has $((free / 1000000000)) GB free, kit needs $((need / 1000000000)) GB"

  log "writing $out ($((need / 1000000000)) GB, tier $tier, fs $fs)"
  mkdir -p "$out/models" "$out/bin" "$out/lib" "$out/src"
  cp "$KIT/run.sh" "$KIT/README.txt" "$KIT/models.tsv" "$out/"
  cp "$KIT/lib/"*.sh "$out/lib/"
  cp "$KIT/src/membw.c" "$out/src/"
  cp -r "$WORK/bin/." "$out/bin/"
  while IFS=$'\t' read -r _t _r file size _s _u; do
    if [ -f "$out/models/$file" ] && [ "$(stat -c %s "$out/models/$file")" = "$size" ]; then log "have  $file"; continue; fi
    log "copy  $file"
    cp "$WORK/models/$file" "$out/models/$file.part" && mv "$out/models/$file.part" "$out/models/$file"
  done < <(model_rows "$tier")
  {
    printf '{\n  "built_at": "%s",\n  "tier": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$tier"
    printf '  "stock_commit": "%s",\n' "$(git -C "$WORK/src/llama.cpp-stock" rev-parse HEAD)"
    printf '  "prism_commit": "%s",\n' "$(git -C "$WORK/src/llama.cpp-prism" rev-parse HEAD)"
    printf '  "stock_patch": "kernels/apply_ssse3.py sha256 %s, kernels/apply_tiled_sse.py sha256 %s",\n' \
      "$(sha256sum "$KERNELS/apply_ssse3.py" | cut -d' ' -f1)" "$(sha256sum "$KERNELS/apply_tiled_sse.py" | cut -d' ' -f1)"
    local dirty=""; [ -n "$(git -C "$KIT" status --porcelain -- . 2>/dev/null)" ] && dirty="-dirty"
    printf '  "kit_commit": "%s%s",\n' "$(git -C "$KIT" rev-parse --short HEAD 2>/dev/null || echo none)" "$dirty"
    printf '  "variants": "%s"\n}\n' "$VARIANTS"
  } > "$out/MANIFEST.json"
  sync
  log "done. Safely eject the USB before unplugging."
}

case "${1:-}" in
  src) cmd_src ;;
  binaries) cmd_binaries ;;
  test) cmd_test ;;
  models) cmd_models "${2:-standard}" ;;
  assemble) shift; cmd_assemble "$@" ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
