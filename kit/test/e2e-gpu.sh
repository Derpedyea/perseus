#!/usr/bin/env bash
# GPU phase (3) and Flash-Next phase (4) on an Ubuntu 24.04 container with Mesa's Vulkan driver,
# with the host's GPUs passed in or not. Needs a host with a discrete AMD GPU (/dev/dri/renderD128
# here) and an AMD iGPU (renderD129); E2E_DGPU / E2E_IGPU override the render nodes.
#  (a) discrete GPU visible, Vulkan with the features Polaris lacks turned off: 3-01 picks it,
#      3-02 self-test passes, 3-03 times Flash-Next's GPU matmuls, tiny model runs on the GPU;
#      Flash-Next steps skip on a box with less free RAM than the model needs
#  (b) no GPU visible: 3-01 skipped with the device list, every later GPU step skipped
#  (c) only the iGPU visible: it doesn't count as the GPU; 3-01 skipped
#  (d) plain Ubuntu 24.04 without libvulkan1: the Vulkan binaries start on the loader the kit ships
#  (e) the GPU disappears after 3-01 picked it (resume without /dev/dri): the GPU steps fail, they
#      don't pass on the CPU or on zero test cases
#  (f) a kit missing an op list's .builtin: 3-02 fails instead of skipping the ops it lists
# Models are hardlinked into the kit (no 76 GB copy) and run with --no-copy.
# usage: e2e-gpu.sh
set -uo pipefail
BK="$(cd "$(dirname "${BASH_SOURCE[0]}")/../build" && pwd)/build-kit.sh"
LABSIM="$(cd "$(dirname "${BASH_SOURCE[0]}")/labsim" && pwd)"
W="${PERSEUS_KIT_WORK:-$HOME/.cache/perseus-kit}"
DGPU="${E2E_DGPU:-/dev/dri/renderD128}" IGPU="${E2E_IGPU:-/dev/dri/renderD129}"
IMAGE=perseus-kit-labsim:noble C=perseus-e2e-gpu P="/media/lab/MY USB/PERSEUS-KIT"
FN1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf FN2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
TINY=Qwen3.5-0.8B-Q4_0.gguf
pass=0; fail=0
ok() { echo "PASS  $1"; pass=$((pass + 1)); }
bad() { echo "FAIL  $1"; fail=$((fail + 1)); }
expect() { [ "$2" = "$3" ] && ok "$1 ($3)" || bad "$1: expected '$2', got '$3'"; }
has() { case "$3" in *"$2"*) ok "$1" ;; *) bad "$1: '$2' not in: $(echo "$3" | tail -5 | tr '\n' '|')" ;; esac; }
for f in $TINY $FN1 $FN2; do [ -f "$W/models/$f.sha256ok" ] || { echo "need $f (build-kit.sh models full)"; exit 2; }; done
T=$(mktemp -d "$W/gpu-test.XXXX")
cleanup() { docker rm -f "$C" >/dev/null 2>&1; rm -rf "$T"; }
trap cleanup EXIT
docker build -q -t "$IMAGE" "$LABSIM" >/dev/null || { echo "labsim image build failed"; exit 2; }

# A diag kit plus hardlinks to the models it needs: same files, no copy.
"$BK" assemble "$T" diag > "$T/assemble.out" 2>&1 || { cat "$T/assemble.out"; exit 2; }
K="$T/PERSEUS-KIT"
for f in $TINY $FN1 $FN2; do ln "$W/models/$f" "$K/models/$f" || { echo "hardlink failed (same filesystem needed)"; exit 2; }; done
expect "GPU op lists in the kit, each with its .builtin" "Qwen3.5-0.8B-Q4_0.txt Qwen3.5-0.8B-Q4_0.txt.builtin ${FN1%.gguf}.txt ${FN1%.gguf}.txt.builtin" \
  "$(ls "$K/gpu-ops" | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')"

ct_start() { # extra docker args...
  docker rm -f "$C" >/dev/null 2>&1
  docker run -d --init --memory=12g --memory-swap=12g --name "$C" --hostname labmac --network none -u "$(id -u):$(id -g)" -e HOME=/tmp/home \
    --group-add "$(stat -c %g "$DGPU")" "$@" -v "$K:$P" -v "$T/local:/var/tmp/perseus-kit-local" "$IMAGE" sleep infinity >/dev/null
}
inside() { docker exec "$C" bash -c "$1"; }
kit_run() { inside "mkdir -p /tmp/home && cd '$P' && $* bash run.sh --new --no-copy > /tmp/run.out 2>&1; echo \$? > /tmp/run.rc"; }
kit_resume() { inside "cd '$P' && $* bash run.sh --no-copy > /tmp/run.out 2>&1; echo \$? > /tmp/run.rc"; }
RD() { ls -d "$K"/results/*/ | tail -1; }
st() { cat "$(RD)steps/$1/status" 2>/dev/null || echo none; }
why() { cat "$(RD)steps/$1/reason" 2>/dev/null; }
out() { cat "$(RD)steps/$1/stdout.txt" 2>/dev/null; }
mkdir -p "$T/local"

# ---------- (a) discrete GPU, Polaris-like feature set ----------
POLARIS="-e GGML_VK_DISABLE_F16=1 -e GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 -e GGML_VK_DISABLE_COOPMAT=1 -e GGML_VK_DISABLE_COOPMAT2=1 -e GGML_VK_DISABLE_BFLOAT16=1 -e GGML_VK_DISABLE_DOT2=1"
ct_start --device "$DGPU" --device "$IGPU" $POLARIS
kit_run "PERSEUS_PAUSE_AFTER=4-04-flashnext-cpu-only"
has "run paused after phase 4" "Paused. Results so far are saved" "$(inside 'cat /tmp/run.out')"
expect "3-01 picks a GPU" ok "$(st 3-01-gpu-probe)"
has "3-01 picks the discrete GPU, not the iGPU" "picked:   Vulkan0: AMD Radeon RX 9070" "$(out 3-01-gpu-probe)"
expect "gpu.env names it by llama.cpp device name" "GPU_DEV=Vulkan0" "$(head -1 "$(RD)gpu.env" 2>/dev/null)"
has "3-05 bench ran on the picked GPU" "AMD Radeon RX 9070" "$(out 3-05-tiny-gpu-bench)"
expect "3-02 self-test passes" ok "$(st 3-02-gpu-selftest)"
# one run per op list, plus llama.cpp's own cases for each op the lists leave out (gpu-ops/*.builtin)
n_sets=$(( $(ls "$K"/gpu-ops/*.txt | wc -l) + $(cat "$K"/gpu-ops/*.builtin | sort -u | wc -l) ))
expect "3-02 runs every op list and built-in set, all passing" "$n_sets" \
  "$(sed 's/\x1b\[[0-9;]*m//g' <<<"$(out 3-02-gpu-selftest)" | grep -cE '^ +([1-9][0-9]*)/\1 tests passed')"
has "3-02 covers the ops left out of the replay" "== llama.cpp's ARGSORT cases" "$(out 3-02-gpu-selftest)"
expect "3-03 op bench" ok "$(st 3-03-gpu-op-bench)"
expect "3-03 times every Flash-Next GPU matmul" yes "$([ "$(out 3-03-gpu-op-bench | grep -c 'GB/s')" -ge 70 ] && echo yes || echo no)"
expect "3-04 tiny model on the GPU" ok "$(st 3-04-tiny-gpu-sanity)"
has "3-04 says it offloaded" "offloaded" "$(cat "$(RD)steps/3-04-tiny-gpu-sanity/stderr.txt")"
expect "3-05 tiny GPU bench" ok "$(st 3-05-tiny-gpu-bench)"
has "3-05 bench ran on Vulkan with all layers" '"n_gpu_layers": 99' "$(out 3-05-tiny-gpu-bench)"
expect "4-01/4-02 stage skipped (--no-copy)" "skipped skipped" "$(st 4-01-flashnext-stage) $(st 4-02-flashnext-ngram-stage)"
# ram_check's rule (run.sh): 115% of both files + 1.5 GB. Within 2 GiB of it, free RAM moving
# during the run can tip either way, so both outcomes count there.
fn_bytes=$(awk -F'\t' -v a="$FN1" -v b="$FN2" '$3 == a || $3 == b {s += $4} END {print s}' "$K/models.tsv")
need_kb=$(( fn_bytes / 1024 * 115 / 100 + 1572864 ))
have_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
for s in 4-03-flashnext-cpu-gpu 4-04-flashnext-cpu-only; do
  if [ "$have_kb" -lt $(( need_kb - 2097152 )) ]; then
    expect "$s skipped for RAM" skipped "$(st $s)"; has "$s says how much RAM" "free RAM" "$(why $s)"
  elif [ "$have_kb" -gt $(( need_kb + 2097152 )) ]; then
    expect "$s ran" ok "$(st $s)"
  else
    has "$s ran or skipped for RAM (free RAM at the threshold)" "$(st $s)" "ok skipped"
  fi
done
sum=$(cat "$(RD)summary.txt")
has "summary: picked GPU" "picked:   Vulkan0: AMD Radeon RX 9070" "$sum"
has "summary: GPU matmul table" "Flash-Next decode matmuls on the GPU" "$sum"
expect "summary: table has 15 rows (us, GB/s, format, tensor)" 15 "$(grep -cE '^ +[0-9.]+ us +[0-9.]+ GB/s  [A-Za-z0-9_]+ +[A-Za-z0-9_.-]+$' <<<"$sum")"
# K-quant formats are spelled with a capital K (q5_K): Flash-Next's output matmul (Q5_K) is the
# biggest single read per token, so it belongs in the slowest-first table
has "summary: table includes the Q5_K output matmul" "q5_K     result_output" "$sum"
has "env.txt records the Polaris-like overrides" "GGML_VK_DISABLE_F16=1" "$(cat "$(RD)steps/3-03-gpu-op-bench/env.txt")"
# the kit changes nothing outside its local dir: the GPU driver's shader cache lands there too
expect "nothing written under \$HOME" "" "$(inside 'find /tmp/home -mindepth 1 | head -3')"
expect "Mesa's shader cache is in the kit's local dir" yes "$([ -n "$(find "$T/local/cache/mesa_shader_cache" -type f 2>/dev/null | head -1)" ] && echo yes || echo no)"

# ---------- (b) no GPU ----------
ct_start
kit_run "PERSEUS_PAUSE_AFTER=3-05-tiny-gpu-bench"
expect "(b) 3-01 skipped" skipped "$(st 3-01-gpu-probe)"
has "(b) 3-01 says no devices and where to look" "no Vulkan devices: no GPU driver, or no access to /dev/dri" "$(why 3-01-gpu-probe)"
expect "(b) later GPU steps skipped" "skipped skipped skipped skipped" \
  "$(st 3-02-gpu-selftest) $(st 3-03-gpu-op-bench) $(st 3-04-tiny-gpu-sanity) $(st 3-05-tiny-gpu-bench)"
has "(b) reason points at the probe" "see steps/3-01-gpu-probe" "$(why 3-02-gpu-selftest)"
has "(b) sysinfo records the Vulkan drivers" "radeon_icd" "$(cat "$(RD)steps/1-02-sysinfo/vulkan.txt")"

# ---------- (c) only the iGPU ----------
ct_start --device "$IGPU"
kit_run "PERSEUS_PAUSE_AFTER=3-01-gpu-probe"
expect "(c) iGPU alone: 3-01 skipped" skipped "$(st 3-01-gpu-probe)"
has "(c) names the iGPU it saw" "no discrete Vulkan GPU, only: ggml_vulkan: 0 = AMD Ryzen" "$(why 3-01-gpu-probe)"
has "(c) the iGPU is integrated" "uma: 1" "$(why 3-01-gpu-probe)"

# ---------- (e) GPU gone after the probe ----------
ct_start --device "$DGPU"
kit_run "PERSEUS_PAUSE_AFTER=3-01-gpu-probe"
expect "(e) 3-01 picked the GPU" ok "$(st 3-01-gpu-probe)"
ct_start
kit_resume "PERSEUS_PAUSE_AFTER=3-05-tiny-gpu-bench"
expect "(e) GPU steps fail once the GPU is gone" "failed failed failed failed" \
  "$(st 3-02-gpu-selftest) $(st 3-03-gpu-op-bench) $(st 3-04-tiny-gpu-sanity) $(st 3-05-tiny-gpu-bench)"
has "(e) and say so" "now Vulkan0 is 'not there'" "$(cat "$(RD)steps/3-02-gpu-selftest/stderr.txt")"

# ---------- (d) no Vulkan loader on the box ----------
d=$(docker run --rm --network none -u "$(id -u):$(id -g)" -v "$K/bin/stock-vk:/vk:ro" ubuntu:24.04 bash -c \
  'ls /usr/lib/x86_64-linux-gnu/libvulkan.so.1 >/dev/null 2>&1 && echo "has libvulkan1"; /vk/nehalem/llama-bench --list-devices > /tmp/o 2>&1; echo "exit $?"; cat /tmp/o')
has "(d) the image has no libvulkan1, the binary starts" "exit 0" "$(head -1 <<<"$d")"
has "(d) and lists devices (none here)" "Available devices" "$d"

# ---------- (f) damaged kit: an op list without its .builtin ----------
rm "$K/gpu-ops/${FN1%.gguf}.txt.builtin"
ct_start --device "$DGPU"
kit_run "PERSEUS_PAUSE_AFTER=3-02-gpu-selftest"
expect "(f) 3-02 fails without the .builtin" failed "$(st 3-02-gpu-selftest)"
has "(f) and names it" "${FN1%.gguf}.txt.builtin is missing" "$(cat "$(RD)steps/3-02-gpu-selftest/stderr.txt")"

echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
