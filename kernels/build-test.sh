#!/usr/bin/env bash
# Builds the self-tests against the stock llama.cpp build of each CPU variant (rebuilding that
# variant's ggml libraries first), inside the kit's build image:
#   test_ssse3_kernels.c, test_fn_kernels.c, llama.cpp's tests/test-tiled-mulmat.cpp,
#   lut_bench.c (standalone) and, for the pre-AVX variants (the only ones with the IQ3_S repack),
#   test_repack_mulmat.c.
#   kernels/build-test.sh [variant...]     default: penryn nehalem haswell
# Output: $WORK/test/<test>-<variant>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${PERSEUS_KIT_WORK:-$HOME/.cache/perseus-kit}"
VARIANTS="${*:-penryn nehalem haswell}"
for v in $VARIANTS; do
  [ -d "$WORK/build/stock-$v" ] || { echo "no $WORK/build/stock-$v; run kit/build/build-kit.sh binaries first" >&2; exit 1; }
  case "$v" in penryn) march="-march=core2 -msse4.1" ;; *) march="-march=$v" ;; esac
  docker run --rm --memory=10g --memory-swap=10g -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$WORK:/work" -v "$HERE:/kernels:ro" perseus-kit-build:bookworm bash -euo pipefail -c "
      cmake --build /work/build/stock-$v -j\$(nproc) --target ggml-base ggml-cpu ggml > /work/build/stock-$v.libs.log 2>&1 \
        || { tail -30 /work/build/stock-$v.libs.log; exit 1; }
      S=/work/src/llama.cpp-stock B=/work/build/stock-$v/ggml/src
      mkdir -p /work/test
      gcc -O2 $march -mtune=generic -Wall -Wextra -static -pthread \
        -I\$S/ggml/include -I\$S/ggml/src -I\$S/ggml/src/ggml-cpu \
        /kernels/test_ssse3_kernels.c \$B/libggml.a \$B/libggml-cpu.a \$B/libggml-base.a -lstdc++ -lm \
        -o /work/test/test_ssse3_kernels-$v
      gcc -O2 $march -mtune=generic -Wall -Wextra -static -pthread \
        -I\$S/ggml/include -I\$S/ggml/src -I\$S/ggml/src/ggml-cpu \
        /kernels/test_fn_kernels.c \$B/libggml.a \$B/libggml-cpu.a \$B/libggml-base.a -lstdc++ -lm \
        -o /work/test/test-fn-kernels-$v
      g++ -O2 $march -mtune=generic -static -pthread -I\$S/ggml/include \$S/tests/test-tiled-mulmat.cpp \
        \$B/libggml.a \$B/libggml-cpu.a \$B/libggml-base.a -lm -o /work/test/test-tiled-mulmat-$v
      gcc -O2 $march -mtune=generic -Wall -Wextra -static -o /work/test/test-lut-$v /kernels/lut_bench.c
      if [ $v != haswell ]; then
        gcc -O2 $march -mtune=generic -Wall -Wextra -static -pthread -I\$S/ggml/include /kernels/test_repack_mulmat.c \
          \$B/libggml.a \$B/libggml-cpu.a \$B/libggml-base.a -lstdc++ -lm -o /work/test/test-repack-mulmat-$v
      fi
    "
  echo "built the $v self-tests in $WORK/test"
done
