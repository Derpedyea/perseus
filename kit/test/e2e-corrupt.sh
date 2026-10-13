#!/usr/bin/env bash
# One flipped byte in a model on the "USB" must fail that model's stage step (checksum, caught
# by the background copier) and skip its benchmark, while the healthy model still runs.
# usage: e2e-corrupt.sh ASSEMBLED_KIT_DIR
set -uo pipefail
SRC="$1"; T=$(mktemp -d "$HOME/.cache/perseus-kit/corrupt-test.XXXX"); K="$T/PERSEUS-KIT"; LD="$T/local"
cp -a --reflink=auto "$SRC" "$K" && rm -rf "$K/results" && mkdir -p "$LD"
printf '\xff' | dd of="$K/models/Qwen3.5-0.8B-Q4_K_M.gguf" bs=1 seek=104857600 conv=notrunc status=none
P="/media/lab/MY USB/PERSEUS-KIT"; C=perseus-e2e-corrupt
docker rm -f "$C" >/dev/null 2>&1
docker run -d --init --memory=10g --memory-swap=10g -e PERSEUS_VARIANT=nehalem --name "$C" --network none -u "$(id -u):$(id -g)" \
  -v "$K:$P" -v "$LD:/var/tmp/perseus-kit-local" perseus-kit-build:bookworm \
  bash -c "cd '$P' && exec setsid -w bash run.sh > /tmp/run.out 2>&1" >/dev/null
i=0; until [ -f "$(ls -d "$K"/results/*/ 2>/dev/null | tail -1)steps/2-05-tiny-q4km-bench/status" ] || [ "$i" -ge 900 ]; do sleep 2; i=$((i + 2)); done
docker exec "$C" bash -c "cd '$P' && bash run.sh stop" >/dev/null
sleep 5; docker rm -f "$C" >/dev/null
R=$(ls -d "$K"/results/*/ | tail -1)
pass=0 fail=0
expect() { [ "$2" = "$3" ] && { echo "PASS  $1 ($3)"; pass=$((pass + 1)); } || { echo "FAIL  $1: expected '$2', got '$3'"; fail=$((fail + 1)); }; }
s() { cat "$R/steps/$1/status" 2>/dev/null || echo none; }
expect "healthy model still runs"           ok      "$(s 2-03-tiny-q4_0-bench)"
expect "corrupt copy fails its stage step"  failed  "$(s 2-04-tiny-q4km-stage)"
expect "failure names the cause"            yes     "$(grep -q 'checksum mismatch' "$R/steps/2-04-tiny-q4km-stage/stderr.txt" && echo yes || echo no)"
expect "bench skipped for corrupt model"    skipped "$(s 2-05-tiny-q4km-bench)"
expect "summary lists the failure"          yes     "$(grep -q '2-04-tiny-q4km-stage *failed' "$R/summary.txt" && echo yes || echo no)"
rm -rf "$T"
echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
