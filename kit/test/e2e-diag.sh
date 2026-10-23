#!/usr/bin/env bash
# Diagnostics-only kit (tier diag, no models):
#  (a) 'build-kit.sh models diag' downloads nothing; 'assemble DEST diag' writes scripts, binaries
#      and a diag MANIFEST with zero models; a run reaches DONE with every model step 'absent',
#      no background copier and nothing but bin/ in the local dir. DEST may be a FAT32 stick
#      mounted by udisks (showexec: nothing on it is executable).
#  (b) a diag run paused right after the tiny-model steps, the same USB re-assembled with the
#      core tier, then resumed: the steps that were 'absent' run now (fails if 'absent' is final).
# usage: e2e-diag.sh [DEST]   DEST for (a) (default: a temp dir). (b) always assembles in a temp
#   dir under the build cache, since core can't go on FAT32.
# E2E_TAG keeps container names apart so several copies can run at once.
set -uo pipefail
BK="$(cd "$(dirname "${BASH_SOURCE[0]}")/../build" && pwd)/build-kit.sh"
W="${PERSEUS_KIT_WORK:-$HOME/.cache/perseus-kit}"
TAG="${E2E_TAG:-0}"
T=$(mktemp -d "$W/diag-test.XXXX")
P="/media/lab/MY USB/PERSEUS-KIT"
pass=0; fail=0
ok() { echo "PASS  $1"; pass=$((pass + 1)); }
bad() { echo "FAIL  $1"; fail=$((fail + 1)); }
expect() { [ "$2" = "$3" ] && ok "$1 ($3)" || bad "$1: expected '$2', got '$3'"; }
has() { case "$3" in *"$2"*) ok "$1" ;; *) bad "$1: '$2' not in: $(echo "$3" | tail -5 | tr '\n' '|')" ;; esac; }

# container NAME on kit dir KIT with local dir LD; runs as the calling user, offline, memory-capped
ct_start() { # name kit local
  docker rm -f "$1" >/dev/null 2>&1
  docker run -d --init --memory=10g --memory-swap=10g -e PERSEUS_VARIANT=nehalem --name "$1" --network none \
    -u "$(id -u):$(id -g)" -v "$2:$P" -v "$3:/var/tmp/perseus-kit-local" perseus-kit-build:bookworm sleep infinity >/dev/null
}
inside() { docker exec "$C" bash -c "$1"; }
# runs the kit to its own exit (DONE, pause, time limit); env assignments may precede the args
kit_run() { inside "cd '$P' && $* bash run.sh >> /tmp/run.out 2>&1; echo \$? >> /tmp/run.rc"; }
RD() { ls -d "$1"/results/*/ | tail -1; }
st() { cat "$(RD "$1")steps/$2/status" 2>/dev/null || echo none; }
# step ids of the plan, and the ones whose function needs a model file
plan_ids() { sed -n 's/^  S \([^ ]*\) .*/\1/p' "$1/run.sh"; }
# a step whose function (behind step_cfg's config, if any) needs a model
model_ids() { sed -n 's/^  S \([^ ]*\) \(step_cfg [a-z-]* \)\{0,1\}step_\(stage\|sanity\|bench\|gpu_sanity\|gpu_bench\|gen\|moe_bench\|fn_pick\|fn_profile\) .*/\1/p' "$1/run.sh"; }
cleanup() { docker rm -f "perseus-e2e-diag-a-$TAG" "perseus-e2e-diag-b-$TAG" >/dev/null 2>&1; rm -rf "$T"; }
trap cleanup EXIT

# ---------- (a) ----------
mkdir -p "$T/shim" "$T/emptywork"
printf '#!/bin/sh\necho "$@" >> "%s/curl-calls"\nexit 1\n' "$T" > "$T/shim/curl"; chmod +x "$T/shim/curl"
PATH="$T/shim:$PATH" PERSEUS_KIT_WORK="$T/emptywork" "$BK" models diag > "$T/models.out" 2>&1
expect "models diag exits 0" 0 "$?"
expect "models diag downloads nothing" "no calls, no files" \
  "$([ -f "$T/curl-calls" ] && echo "curl called" || echo "no calls"), $([ -n "$(ls -A "$T/emptywork" 2>/dev/null)" ] && echo files || echo "no files")"

DEST="${1:-$T/stick-a}"; mkdir -p "$DEST"
if [ -d "$DEST/PERSEUS-KIT" ]; then
  [ -f "$DEST/PERSEUS-KIT/MANIFEST.json" ] || { echo "$DEST/PERSEUS-KIT exists and isn't a kit; refusing to touch it"; exit 2; }
  rm -rf "$DEST/PERSEUS-KIT"
fi
"$BK" assemble "$DEST" diag > "$T/assemble-a.out" 2>&1
expect "assemble diag exits 0" 0 "$?"
K="$DEST/PERSEUS-KIT"
expect "MANIFEST tier" diag "$(sed -n 's/.*"tier": "\([a-z]*\)".*/\1/p' "$K/MANIFEST.json")"
expect "zero models on the USB" 0 "$(find "$K/models" -type f | wc -l)"
expect "no file over 4 GiB (FAT32 limit)" 0 "$(find "$K" -type f -size +4194303k | wc -l)"
echo "      diag kit: $(du -sb "$K" | cut -f1) bytes on $(stat -f -c %T "$DEST")"

C="perseus-e2e-diag-a-$TAG"; LD="$T/local-a"; mkdir -p "$LD"
ct_start "$C" "$K" "$LD"
kit_run ""
out=$(inside "cat /tmp/run.out")
expect "run exits 0" 0 "$(inside 'cat /tmp/run.rc')"
has "startup line says diagnostics-only" "Diagnostics kit: no models on this USB" "$out"
has "DONE counts the absent tests" "DONE (" "$out"
expect "FINISHED written" yes "$([ -f "$(RD "$K")FINISHED" ] && echo yes || echo no)"
n_plan=$(plan_ids "$K" | wc -l)
expect "every planned step has an outcome" "$n_plan" "$(ls "$(RD "$K")"steps/*/status | wc -l)"
expect "every model step absent" "" "$(for s in $(model_ids "$K"); do [ "$(st "$K" "$s")" = absent ] || echo "$s=$(st "$K" "$s")"; done | tr '\n' ' ')"
expect "other steps ok or skipped" "" "$(for s in $(plan_ids "$K" | grep -vxF -f <(model_ids "$K")); do case "$(st "$K" "$s")" in ok|skipped) ;; *) echo "$s=$(st "$K" "$s")" ;; esac; done | tr '\n' ' ')"
expect "Flash-Next kernel checks ran" "ok ok" "$(st "$K" 2-00f-flashnext-kernel-selftest) $(st "$K" 2-00g-flashnext-kernel-bench)"
# Debian 12's glibc (2.36) is older than the Vulkan build's (Ubuntu 24.04): GPU steps skip, saying why
expect "GPU probe skipped on an older glibc" skipped "$(st "$K" 3-01-gpu-probe)"
has "GPU probe says which glibc" "needs glibc 2.38 or newer; this box has 2.36" "$(cat "$(RD "$K")steps/3-01-gpu-probe/reason" 2>/dev/null)"
expect "GPU op list steps skipped too" "skipped skipped" "$(st "$K" 3-02-gpu-selftest) $(st "$K" 3-03-gpu-op-bench)"
expect "local dir holds only bin/" ".perseus-kit bin" "$(ls -A "$LD" | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//')"
expect "no background copier started" absent "$([ -f "$(RD "$K")prefetch.log" ] && echo present || echo absent)"
s=$(inside "cd '$P' && bash run.sh status")
has "status: finished" "state: finished" "$s"
has "status counts the absent steps" "need models not on this USB" "$s"
sum=$(cat "$(RD "$K")summary.txt")
has "summary says how many need models" "tests need models this USB doesn't carry" "$sum"
expect "summary lists every step" "$n_plan" "$(for s in $(plan_ids "$K"); do grep -q "^$s " <<<"$sum" && echo "$s"; done | wc -l)"
docker rm -f "$C" >/dev/null

# ---------- (b) ----------
mkdir -p "$T/stick-b"; K="$T/stick-b/PERSEUS-KIT"; LD="$T/local-b"; mkdir -p "$LD"
"$BK" assemble "$T/stick-b" diag > "$T/assemble-b1.out" 2>&1 || bad "assemble diag for (b)"
C="perseus-e2e-diag-b-$TAG"
ct_start "$C" "$K" "$LD"
LAST_TINY=2-05-tiny-q4km-bench
TINY="2-01-tiny-q4_0-stage 2-02-tiny-q4_0-sanity 2-03-tiny-q4_0-bench 2-04-tiny-q4km-stage $LAST_TINY"
kit_run "PERSEUS_PAUSE_AFTER=$LAST_TINY"
has "diag run paused" "Paused. Results so far are saved" "$(inside 'cat /tmp/run.out')"
expect "not finished" absent "$([ -f "$(RD "$K")FINISHED" ] && echo present || echo absent)"
expect "tiny-model steps absent before" "absent absent absent absent absent" "$(for s in $TINY; do st "$K" "$s"; done | tr '\n' ' ' | sed 's/ $//')"

"$BK" assemble "$T/stick-b" core > "$T/assemble-b2.out" 2>&1
expect "re-assemble with core exits 0" 0 "$?"
expect "MANIFEST tier after re-assemble" core "$(sed -n 's/.*"tier": "\([a-z]*\)".*/\1/p' "$K/MANIFEST.json")"

# The resume's first step is a stage step, right after the copier is launched. A shim delays the
# copier's start by 3 s (as idle priority under load can): the step must wait for it, not copy
# the same file alongside it.
inside "mkdir -p /tmp/shim && printf '#!/bin/sh\nsleep 3\nexec /usr/bin/ionice \"\$@\"\n' > /tmp/shim/ionice && chmod +x /tmp/shim/ionice"
kit_run 'PATH=/tmp/shim:$PATH' "PERSEUS_PAUSE_AFTER=$LAST_TINY"
expect "resumed run paused again" 2 "$(inside "grep -c 'Paused. Results so far are saved' /tmp/run.out")"
res=$(for s in $TINY; do st "$K" "$s"; done | tr '\n' ' ' | sed 's/ $//')
expect "tiny-model steps ran after resume" "ok ok ok ok ok" "$res"
[ "$res" = "ok ok ok ok ok" ] || for s in $TINY; do echo "      $s: $(tail -3 "$(RD "$K")steps/$s/stderr.txt" 2>/dev/null | tr '\n' '|')"; done
has "first stage step waited for the slow-starting copier" "for the background copy" "$(cat "$(RD "$K")steps/2-01-tiny-q4_0-stage/stdout.txt" 2>/dev/null)"
expect "same results folder" 1 "$(ls -d "$K"/results/*/ | wc -l)"
expect "finished steps not rerun" 1 "$(grep -c '>> 1-01-env' "$(RD "$K")run.log")"
expect "status no longer counts phase 2 as absent" "" "$(inside "cd '$P' && bash run.sh status" | grep '^  2\. ' | grep -o 'need models not on this USB')"

# ---------- (c) a split model only partly downloaded ----------
# Flash-Next is two files; with only the first verified, assemble must leave its GPU op list out
# (llama.cpp can't open half a split model), not fail: a diag or core kit doesn't need it.
FN1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf TINY=Qwen3.5-0.8B-Q4_0.gguf
if [ -f "$W/models/$FN1.sha256ok" ] && [ -f "$W/models/$TINY.sha256ok" ]; then
  PW="$T/partwork"; mkdir -p "$PW/models" "$PW/build/stock-vk-nehalem/bin" "$PW/src/llama.cpp-stock/ggml/include" "$T/stick-c"
  cp -a --reflink=auto "$W/bin" "$PW/bin"
  cp --reflink=auto "$W/build/stock-vk-nehalem/bin/test-export-graph-ops" "$PW/build/stock-vk-nehalem/bin/"
  cp "$W/src/llama.cpp-stock/ggml/include/ggml.h" "$PW/src/llama.cpp-stock/ggml/include/"
  for f in "$FN1" "$TINY"; do ln "$W/models/$f" "$PW/models/$f" && touch "$PW/models/$f.sha256ok"; done
  PERSEUS_KIT_WORK="$PW" "$BK" assemble "$T/stick-c" diag > "$T/assemble-c.out" 2>&1
  expect "(c) assemble with half of Flash-Next exits 0" 0 "$?"
  expect "(c) only the tiny model's op list" "Qwen3.5-0.8B-Q4_0.txt" "$(ls "$T/stick-c/PERSEUS-KIT/gpu-ops" 2>/dev/null | grep -v builtin | tr '\n' ' ' | sed 's/ $//')"
  has "(c) says why Flash-Next's is missing" "or a file split from it) not downloaded" "$(cat "$T/assemble-c.out")"
else
  echo "SKIP  (c): needs $FN1 and $TINY in $W/models (build-kit.sh models full)"
fi

echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
