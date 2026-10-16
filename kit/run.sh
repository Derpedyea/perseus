#!/usr/bin/env bash
# perseus lab kit: collects hardware facts and inference benchmarks on the lab box and
# logs everything to results/ on the USB. Usage: bash run.sh --help. Plain-language
# instructions: README.txt.
# Tests only: PERSEUS_PAUSE_AFTER=STEP-ID pauses after that step, as 'run.sh pause' would;
# PERSEUS_SYSFS_DRM=DIR stands in for /sys/class/drm (the GPU's clock switch, lib/gpu.sh).
set -u

KIT="$(cd "$(dirname "$0")" && pwd)"
. "$KIT/lib/common.sh"
. "$KIT/lib/ui.sh"
. "$KIT/lib/gpu.sh"

# Fixed path (not $HOME) so 'sudo bash run.sh' and 'bash run.sh clean' agree on it.
LOCAL="${PERSEUS_LOCAL:-/var/tmp/perseus-kit-local}"
# The GPU's performance level before phase 4b pinned it (lib/gpu.sh); here so 'clean' sees it too.
DPM_RECORD="$LOCAL/gpu-dpm.restore"

usage() {
  cat <<EOF
perseus lab kit: measures this machine and the models on this USB.
Results go to results/ on the USB. Plain-language instructions: README.txt

Usage
  $UI_CMD [--for TIME] [--retry-failed] [--new] [--no-copy]
  $UI_CMD status | pause | stop | clean | help

Examples
  $UI_CMD --for 95m    a 2-hour visit: no new tests after 95 min
  $UI_CMD              overnight: run everything
  $UI_CMD status       how far it is (from a second terminal)

Commands
  (none)    run the tests, most important first; continues an unfinished run
  status    progress of the current run
  pause     finish the current test, then stop
  stop      stop now; the current test reruns next time (same as Ctrl-C)
  clean     delete the model copies on this disk ($LOCAL)
            (and put the GPU's clocks back if a cut-off run left them pinned)
  help      this text (also -h, --help)

Options
  --for TIME      don't start new tests after TIME: 95 (min), 90m, 8h, 1h30m
  --retry-failed  also rerun tests that failed, timed out or crashed
  --new           start a fresh results folder instead of continuing
  --no-copy       run models straight off the USB (slow loads, no disk use)

Environment
  PERSEUS_VARIANT=penryn|nehalem|haswell   force a CPU build (troubleshooting)
  NO_COLOR=1                               no colours on screen
EOF
}
# Errors go to stderr, exit 2.
usage_error() { printf 'run.sh: %s\nRun "%s --help" for usage.\n' "$1" "$UI_CMD" >&2; exit 2; }

CMD=run BUDGET_SET=0 BUDGET_ARG="" BUDGET_MIN="" RETRY_FAILED=0 NEW_RUN=0 NO_COPY=0
while [ $# -gt 0 ]; do
  case "$1" in
    status|pause|stop|clean) CMD="$1" ;;
    help|-h|--help) usage; exit 0 ;;
    --for) [ $# -ge 2 ] || usage_error "--for needs a time, e.g. --for 95m or --for 8h"
           BUDGET_SET=1 BUDGET_ARG="$2"; shift ;;
    --for=*) BUDGET_SET=1 BUDGET_ARG="${1#--for=}" ;;
    --retry-failed) RETRY_FAILED=1 ;;
    --new) NEW_RUN=1 ;;
    --no-copy) NO_COPY=1 ;;
    *) usage_error "unknown argument '$1'" ;;
  esac
  shift
done
# An empty --for= is an error too, not "no limit".
if [ "$BUDGET_SET" = 1 ]; then
  BUDGET_MIN=$(parse_minutes "$BUDGET_ARG") \
    || usage_error "--for '$BUDGET_ARG' isn't a time: use minutes (95) or 90m, 8h, 1h30m (at most a week)"
fi
ui_init

OS=$(uname -s)
HOST=$(hostname 2>/dev/null | cut -d. -f1)
[ -n "$HOST" ] || HOST=unknown
LOCK="$KIT/results/.lock"
PAUSE_FLAG="$KIT/results/.pause"
PROMPT=$'def is_prime(n):\n    """Return True if n is a prime number."""\n'
PROMPT_LONG=$'Here is a quick sort implementation in C++. Just code, no comments:\n\n#include'

# Model files the plan uses (catalog with sizes/checksums: models.tsv).
TINY_Q40=Qwen3.5-0.8B-Q4_0.gguf
TINY_Q4KM=Qwen3.5-0.8B-Q4_K_M.gguf
BONSAI=Ternary-Bonsai-2-27B-PQ2_0.gguf
BONSAI_T=Ternary-Bonsai-2-27B-PTQ1_0.gguf
SALUKI=Underdog-Saluki-27B-1.0-IQ2-mix.gguf
Q27=Qwen3.8-27B-Q4_0.gguf
MTP27=mtp-Qwen3.8-27B-Q4_0.gguf
FLASHNEXT=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
FLASHNEXT_NGRAM=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf

# ---------- the plan: most decision-relevant data first ----------
# (Helpers the plan's arguments call are defined here: 'status' evaluates the plan early.)
# 6, 8, one per core, one per hardware thread: the distinct ones this box has.
fn_threads() {
  local t out="" seen=","
  for t in 6 8 "$NPHYS" "$NCPUS"; do
    [ "$t" -le "$NCPUS" ] || continue
    case "$seen" in *",$t,"*) continue ;; esac
    seen="$seen$t,"; out="${out:+$out,}$t"
  done
  echo "$out"
}

# phase NUM TITLE MINUTES (rough estimate for the Mac Pro) / S STEP-ID FUNCTION ARGS...
plan() {
  phase 1 "machine facts + memory bandwidth" 5
  S 1-01-env step_env
  S 1-02-sysinfo step_sysinfo
  S 1-03-membw-1-thread step_membw one-thread --threads 1 --cpus "$FIRST_PHYS"
  S 1-04-membw-all-cores-local step_membw all-cores-local --phys
  S 1-05-membw-all-threads-local step_membw all-threads-local
  S 1-06-membw-node0-local step_membw_multi node0-local --node-cpus 0 --phys --policy bind:0
  S 1-07-membw-node0-cpus-remote-mem step_membw_multi node0-cpus-node1-mem --node-cpus 0 --phys --policy bind:1
  S 1-08-membw-all-cores-interleave step_membw_multi all-cores-interleave --phys --policy interleave
  S 1-09-membw-all-cores-mem-on-node0 step_membw_multi all-cores-mem-on-node0 --phys --policy bind:0

  phase 2 "kernel checks" 5
  S 2-00-kernel-selftest step_kernel_selftest test-ssse3-kernels
  S 2-00b-tiled-selftest step_tiled_selftest
  S 2-00c-kernel-bench step_kernel_bench test-ssse3-kernels
  S 2-00d-repack-selftest step_repack_selftest
  S 2-00e-lut-bench step_lut_bench
  # Flash-Next formats (q6_K, q5_K, iq4_nl, q2_0) + q4_1; q6_K/q5_K/q4_1 run in phase 6's Qwen3.8-27B
  S 2-00f-flashnext-kernel-selftest step_kernel_selftest test-fn-kernels
  S 2-00g-flashnext-kernel-bench step_kernel_bench test-fn-kernels
  S 2-01-tiny-q4_0-stage step_stage "$TINY_Q40"
  S 2-02-tiny-q4_0-sanity step_sanity "$TINY_Q40"
  S 2-03-tiny-q4_0-bench step_bench "$TINY_Q40" all-sockets cores 128 32 3
  S 2-04-tiny-q4km-stage step_stage "$TINY_Q4KM"
  S 2-05-tiny-q4km-bench step_bench "$TINY_Q4KM" all-sockets cores 128 32 3

  # The lab box's AMD RX 580 through llama.cpp's Vulkan backend. 3-01 picks the GPU; 3-02 checks
  # it computes what the CPU does for every op the GPU will run (no model needed); 3-03 times
  # Flash-Next's GPU-side matmuls on it (GB/s per weight format).
  phase 3 "GPU checks" 10
  S 3-01-gpu-probe step_gpu_probe
  S 3-02-gpu-selftest step_gpu_selftest
  S 3-03-gpu-op-bench step_gpu_op_bench
  S 3-04-tiny-gpu-sanity step_gpu_sanity "$TINY_Q40"
  S 3-05-tiny-gpu-bench step_gpu_bench "$TINY_Q40" 128 32 3

  # The target: Qwen3.8-Flash-Next (180B, 6B active), ISTA's GSQ-RCO IQ3_XXS (within 1% of BF16 on
  # AIME/GPQA/LiveCodeBench). CPU+GPU: attention, shared experts and router on the GPU, routed
  # experts in RAM on the perseus SSSE3 kernels. Same prompt and settings with and without the GPU.
  phase 4 "Flash-Next: CPU+GPU vs CPU only" 35
  S 4-01-flashnext-stage step_stage "$FLASHNEXT"
  S 4-02-flashnext-ngram-stage step_stage "$FLASHNEXT_NGRAM"
  S 4-03-flashnext-cpu-gpu step_flashnext gpu
  S 4-04-flashnext-cpu-only step_flashnext cpu

  # Where Flash-Next's CPU+GPU time goes (run 2: 278 ms a token against a ~85 ms bandwidth
  # ceiling), and three free knobs. novis: GPU buffers in plain VRAM; ggml-vulkan asks for
  # CPU-visible VRAM first, only 256 MB without Resizable BAR, and spills to system RAM over PCIe.
  # dpm-high: the GPU's clocks pinned (lib/gpu.sh); it idles while the CPU runs the experts. dio:
  # weights read into RAM instead of mmap. 4b-05 picks the fastest of 4b-01..04 for the rest.
  # Every GPU step places the same way: all but the routed experts on the GPU. Staging (4-01/02) is
  # done by now, so no background copy runs beside these.
  phase 4b "Flash-Next: where the time goes" 45
  S 4b-01-fn-base step_cfg base step_fn_bench gpu -p 512 -n 64 -r 3 -t "$NPHYS"
  S 4b-02-fn-novis step_cfg novis step_fn_bench gpu -p 512 -n 64 -r 3 -t "$NPHYS"
  S 4b-03-fn-dpm-high step_cfg dpm-high step_fn_bench gpu -p 512 -n 64 -r 3 -t "$NPHYS"
  S 4b-04-fn-novis-dpm-high step_cfg novis-dpm-high step_fn_bench gpu -p 512 -n 64 -r 3 -t "$NPHYS"
  S 4b-05-fn-pick step_fn_pick 4b-01-fn-base:base 4b-02-fn-novis:novis 4b-03-fn-dpm-high:dpm-high 4b-04-fn-novis-dpm-high:novis-dpm-high
  S 4b-06-fn-threads step_cfg best step_fn_bench gpu -p 0 -n 64 -r 3 -t "$(fn_threads)"
  # prompt batches: llama.cpp copies the experts to the GPU per batch (op offload) unless told not to
  S 4b-07-fn-no-op-offload step_cfg best step_fn_bench gpu -p 512 -n 0 -r 3 -t "$NPHYS" -nopo 1
  S 4b-08-fn-gpu-op-times step_cfg best step_fn_profile gpu-ops
  S 4b-09-fn-cpu-profile step_cfg best step_fn_profile perf
  # the spill mechanism on its own: the tiny model fully on the GPU, and 3-03's matmuls
  S 4b-10-tiny-gpu-novis step_cfg novis step_gpu_bench "$TINY_Q40" 128 32 3
  S 4b-11-gpu-op-bench-novis step_cfg novis step_gpu_op_bench
  S 4b-12-fn-cpu-only step_fn_bench cpu -p 512 -n 64 -r 2 -t "$NPHYS"
  # last: it reads 71 GB into RAM, which pushes the mmap steps' page cache out
  S 4b-13-fn-load-dio step_cfg best step_fn_bench gpu -p 0 -n 64 -r 3 -t "$NPHYS" -lm dio

  phase 5 "compact 27Bs: Bonsai 2 (PQ2_0) vs Saluki (IQ2-mix)" 25
  S 5-01-bonsai-pq2-stage step_stage "$BONSAI"
  S 5-02-bonsai-pq2-sanity step_sanity "$BONSAI"
  S 5-03-bonsai-pq2-all-sockets step_bench "$BONSAI" all-sockets cores 32 16 2
  S 5-04-bonsai-pq2-one-socket step_bench "$BONSAI" one-socket cores 32 16 2
  # Saluki runs on the perseus SSSE3 kernels (kernels/), checked by 2-00 on this CPU.
  S 5-05-saluki-stage step_stage "$SALUKI"
  S 5-06-saluki-sanity step_sanity "$SALUKI"
  S 5-07-saluki-all-sockets step_bench "$SALUKI" all-sockets cores 32 16 2
  S 5-08-saluki-one-socket step_bench "$SALUKI" one-socket cores 32 16 2

  phase 6 "Qwen3.8-27B Q4_0 reference" 20
  S 6-01-qwen27-stage step_stage "$Q27"
  S 6-02-qwen27-sanity step_sanity "$Q27"
  S 6-03-qwen27-all-sockets step_bench "$Q27" all-sockets cores 32 16 2
  S 6-04-qwen27-one-socket step_bench "$Q27" one-socket cores 32 16 2

  phase 7 "MTP speculative decoding" 15
  S 7-01-mtp-head-stage step_stage "$MTP27"
  S 7-02-bonsai-pq2-mtp-off step_mtp "$BONSAI" "" 0
  S 7-03-bonsai-pq2-mtp-3 step_mtp "$BONSAI" "$MTP27" 3
  S 7-04-qwen27-mtp-off step_mtp "$Q27" "" 0
  S 7-05-qwen27-mtp-3 step_mtp "$Q27" "$MTP27" 3
  S 7-06-saluki-mtp-off step_mtp "$SALUKI" "" 0
  S 7-07-saluki-mtp-3 step_mtp "$SALUKI" "$MTP27" 3
  # A/B: 7-07 verifies 4 tokens per step with the perseus multi-column kernels, this without
  S 7-08-saluki-mtp-3-mc-off step_mtp_mc_off "$SALUKI" "$MTP27" 3

  phase 8 "extras" 60
  S 8-01-bonsai-pq2-stock-layout step_bench "$BONSAI" naive cores 32 16 2
  S 8-02-qwen27-stock-layout step_bench "$Q27" naive cores 32 16 2
  S 8-03-bonsai-ptq1-stage step_stage "$BONSAI_T"
  S 8-04-bonsai-ptq1-sanity step_sanity "$BONSAI_T"
  S 8-05-bonsai-ptq1-all-sockets step_bench "$BONSAI_T" all-sockets cores 32 16 2
  S 8-06-bonsai-ptq1-one-socket step_bench "$BONSAI_T" one-socket cores 32 16 2
  S 8-07-bonsai-pq2-hyperthreads step_bench "$BONSAI" all-sockets smt 32 16 2
  S 8-08-qwen27-hyperthreads step_bench "$Q27" all-sockets smt 32 16 2
  S 8-09-bonsai-pq2-mtp-2 step_mtp "$BONSAI" "$MTP27" 2
  S 8-10-qwen27-mtp-2 step_mtp "$Q27" "$MTP27" 2
  S 8-11-bonsai-pq2-cold-distribute step_bench "$BONSAI" distribute-cold cores 32 16 2
  S 8-12-qwen27-cold-distribute step_bench "$Q27" distribute-cold cores 32 16 2
  S 8-13-saluki-stock-layout step_bench "$SALUKI" naive cores 32 16 2
  # A/B: 5-07's prompt numbers use the tiled matmul (enabled on SSE4.1 by kernels/), this without
  S 8-14-saluki-tiled-off step_bench_tiled_off "$SALUKI" all-sockets cores 32 16 2
  # Saluki's IQ kernels are table-lookup heavy: do hyperthreads hide that, and is draft 2 better?
  S 8-15-saluki-hyperthreads step_bench "$SALUKI" all-sockets smt 32 16 2
  S 8-16-saluki-mtp-2 step_mtp "$SALUKI" "$MTP27" 2
  # longer prompts: Saluki on the tiled matmul vs Bonsai (no batched kernel for PQ2_0)
  S 8-17-saluki-prompt-128 step_bench "$SALUKI" all-sockets cores 128 0 1
  S 8-18-bonsai-pq2-prompt-128 step_bench "$BONSAI" all-sockets cores 128 0 1
  # A/B: 5-07 runs Saluki's IQ3_S weights repacked at load (kernels/), this in the file layout
  S 8-19-saluki-repack-off step_bench_repack_off "$SALUKI" all-sockets cores 32 16 2
}
# Background copy order: what the plan needs first.
PREFETCH_ORDER="$TINY_Q40 $TINY_Q4KM $FLASHNEXT $FLASHNEXT_NGRAM $BONSAI $SALUKI $Q27 $MTP27 $BONSAI_T"

# ---------- lock owner (shared by run / status / pause / stop) ----------
# The lock holds "HOST PID PGID". OWNER_* are set by owner_alive.
owner_alive() {
  OWNER_PID="" OWNER_PGID=""
  local owner; owner=$(cat "$LOCK/owner" 2>/dev/null) || return 1
  set -- $owner
  [ "${1:-}" = "$HOST" ] && pid_alive "${2:-}" || return 1
  OWNER_PID="$2" OWNER_PGID="${3:-}"
}

latest_run_dir() {
  local d last=""
  for d in "$KIT/results/$HOST-"*; do [ -d "$d" ] && last="$d"; done
  echo "$last"
}

# ---------- commands that don't run tests ----------
case "$CMD" in
  clean)
    if owner_alive; then echo "A run is active (pid $OWNER_PID); stop it first."; exit 1; fi
    # The record of the GPU's pre-run level lives in $LOCAL: never delete it unrestored.
    dpm_restore || { echo "Kept $LOCAL: it holds the GPU's original performance level ($DPM_RECORD). Try: sudo bash run.sh clean"; exit 1; }
    if [ -f "$LOCAL/.perseus-kit" ]; then rm -rf "$LOCAL" && echo "removed $LOCAL"; exit $?; fi
    echo "nothing to clean ($LOCAL is not a perseus kit folder)"; exit 0 ;;
  pause)
    owner_alive || { echo "Nothing is running."; exit 1; }
    touch "$PAUSE_FLAG" 2>/dev/null || { echo "Can't write $PAUSE_FLAG. If you started the run with sudo, use: sudo bash run.sh pause"; exit 1; }
    echo "OK: it will stop after the current test finishes (the run window says 'Paused')."
    echo "For an immediate stop: bash run.sh stop"; exit 0 ;;
  stop)
    owner_alive || { echo "Nothing is running."; exit 1; }
    kill -INT -- "-${OWNER_PGID:-$OWNER_PID}" 2>/dev/null \
      || { echo "Couldn't signal the run. If you started it with sudo, use: sudo bash run.sh stop"; exit 1; }
    echo "Stopping. The current test will rerun next time."; exit 0 ;;
  status)
    RUN_DIR=$(latest_run_dir)
    [ -n "$RUN_DIR" ] || { echo "No runs on this machine yet."; exit 0; }
    if [ -f "$RUN_DIR/FINISHED" ]; then state="finished"
    elif owner_alive; then state="running (pid $OWNER_PID)"; [ -f "$PAUSE_FLAG" ] && state="$state, pausing after the current test"
    else state="stopped; '$UI_CMD' continues it"; fi
    if [ "$UI" = pretty ]; then
      printf ' %sperseus lab kit%s  %s\n   %sstate%s  %s\n\n' "$C_B" "$C_0" "$(basename "$RUN_DIR")" "$C_DIM" "$C_0" "$state"
    else
      echo "run:   $(basename "$RUN_DIR")"
      echo "state: $state"
    fi
    P_TITLE="" PROBLEMS=""
    flush_phase() {
      [ -n "$P_TITLE" ] || return 0
      if [ "$UI" = pretty ]; then
        local n_done=$((P_OK + P_SKIP + P_BAD)) g=" " c="" notes=""
        if [ -n "$P_NOW" ]; then g=$G_RUN c=$C_RUN
        elif [ "$P_BAD" -gt 0 ]; then g=$G_BAD c=$C_BAD
        elif [ "$n_done" -eq "$P_TOTAL" ]; then g=$G_OK c=$C_OK
        elif [ $((n_done + P_ABS)) -eq "$P_TOTAL" ]; then g=$G_ABS c=$C_DIM
        fi
        local fill=$((n_done * 10 / P_TOTAL))
        [ "$P_SKIP" -gt 0 ] && notes="$P_SKIP skipped"
        [ "$P_ABS" -gt 0 ] && notes="${notes:+$notes, }$P_ABS need models not on this USB"
        printf ' %s%s%s %-44s %s%s%s%s%s %2d/%-2d %s%s%s\n' "$c" "$g" "$C_0" "$(ui_fit 44 "$P_TITLE")" \
          "$c" "$(ui_repeat "$fill" "$G_FULL")" "$C_DIM" "$(ui_repeat $((10 - fill)) "$G_EMPTY")" "$C_0" \
          "$n_done" "$P_TOTAL" "$C_DIM" "$notes" "$C_0"
        [ -n "$P_NOW" ] && printf '   %snow%s %s\n' "$C_DIM" "$C_0" "$P_NOW"
        return 0
      fi
      local extra=""
      [ "$P_SKIP" -gt 0 ] && extra=" ($P_SKIP skipped)"
      [ "$P_ABS" -gt 0 ] && extra="$extra ($P_ABS need models not on this USB)"
      [ -n "$P_NOW" ] && extra="$extra  <- now: $P_NOW"
      printf '  %-42s %2d/%-2d done%s\n' "$P_TITLE" $((P_OK + P_SKIP + P_BAD)) "$P_TOTAL" "$extra"
    }
    phase() { flush_phase; P_TITLE="$1. $2"; P_TOTAL=0 P_OK=0 P_SKIP=0 P_BAD=0 P_ABS=0 P_NOW=""; }
    S() {
      local d="$RUN_DIR/steps/$1" st=""
      P_TOTAL=$((P_TOTAL + 1))
      [ -f "$d/status" ] && st=$(cat "$d/status")
      case "$st" in
        ok) P_OK=$((P_OK + 1)) ;;
        skipped) P_SKIP=$((P_SKIP + 1)) ;;
        absent) P_ABS=$((P_ABS + 1)) ;;
        failed|timeout|crashed) P_BAD=$((P_BAD + 1))
          if [ "$UI" = pretty ]; then PROBLEMS="$PROBLEMS   $C_BAD$G_BAD$C_0 $1  $st"$'\n'
          else PROBLEMS="$PROBLEMS  $1: $st"$'\n'; fi ;;
        '') if [ -f "$d/started" ] && owner_alive; then
              local age=$(( $(date +%s) - $(stat -c %Y "$d/started" 2>/dev/null || stat -f %m "$d/started") ))
              if [ "$UI" = pretty ]; then P_NOW="$1 ($(ui_dur "$age"))"; else P_NOW="$1 ($(fmt_dur "$age"))"; fi
            fi ;;
      esac
    }
    FIRST_PHYS=0 NPHYS=1 NCPUS=1  # the plan's arguments; status only counts steps
    plan; flush_phase
    if [ -n "$PROBLEMS" ] && [ "$UI" = pretty ]; then
      printf '\n %sproblems%s  %sdetails: results/<run>/steps/<test>/stderr.txt%s\n%s' "$C_B" "$C_0" "$C_DIM" "$C_0" "$PROBLEMS"
    elif [ -n "$PROBLEMS" ]; then
      printf 'problems (details in results/<run>/steps/<test>/stderr.txt):\n%s' "$PROBLEMS"
    fi
    exit 0 ;;
esac

# ---------- run: results folder, lock, signals ----------
# Refusing to start still puts back GPU clocks a cut-off run left pinned (lib/gpu.sh), unless a live
# run owns them.
refuse() { echo "$1"; owner_alive || dpm_restore; exit 1; }
mkdir -p "$KIT/results" 2>/dev/null
probe="$KIT/results/.write-test.$$"
( echo ok > "$probe" ) 2>/dev/null || refuse "Can't write to $KIT/results. Is the USB read-only or full?"
rm -f "$probe"
usb_free_kb=$(df -Pk "$KIT/results" | awk 'NR==2 {print $4}')
[ "${usb_free_kb:-0}" -ge 204800 ] || refuse "USB has under 200 MB free; results won't fit."

my_pgid() {
  if [ -r "/proc/$$/stat" ]; then awk '{print $5}' "/proc/$$/stat"; else ps -o pgid= -p $$ | tr -d ' '; fi
}
acquire_lock() {
  if mkdir "$LOCK" 2>/dev/null; then echo "$HOST $$ $(my_pgid)" > "$LOCK/owner"; return 0; fi
  [ -f "$LOCK/owner" ] || sleep 2  # owner may be mid-write
  if owner_alive; then echo "Another run is active on this machine (pid $OWNER_PID). Wait for it or stop it first."; return 1; fi
  # Stale lock from a crash or another machine. rename is atomic: only one taker wins.
  mv "$LOCK" "$LOCK.stale.$$" 2>/dev/null && rm -rf "$LOCK.stale.$$"
  mkdir "$LOCK" 2>/dev/null || { echo "Lost a race for the lock; try again."; return 1; }
  echo "$HOST $$ $(my_pgid)" > "$LOCK/owner"
}
acquire_lock || exit 1
rm -f "$PAUSE_FLAG"  # a pause asked for while nothing ran is stale

PREFETCH_PID=""
cleanup() {
  [ -n "$PREFETCH_PID" ] && kill -TERM -- "-$PREFETCH_PID" 2>/dev/null
  dpm_restore
  rm -rf "$LOCK"
}
trap cleanup EXIT

pick_run_dir() {
  local last; last=$(latest_run_dir)
  if [ "$NEW_RUN" = 0 ] && [ -n "$last" ] && [ ! -f "$last/FINISHED" ]; then echo "$last"; return; fi
  echo "$KIT/results/$HOST-$(date -u +%Y%m%dT%H%M%SZ)"
}
RUN_DIR=$(pick_run_dir)
mkdir -p "$RUN_DIR/steps"
CURRENT_STEP="" STEP_T0=0

on_interrupt() {
  ui_tick_stop
  if [ -n "$CURRENT_STEP" ]; then
    echo "stopped by user" > "$RUN_DIR/steps/$CURRENT_STEP/reason"
    write_status "$RUN_DIR/steps/$CURRENT_STEP" interrupted 130 "$(now_s)"
    [ "$UI" = pretty ] && ui_step_line interrupted "$CURRENT_STEP" $(( $(now_s) - STEP_T0 )) "stopped; reruns next time"
  fi
  dpm_restore
  write_summary
  say "Stopped. Run '$UI_CMD' again to continue from here."
  [ "$UI" = pretty ] && ui_end "$C_WARN" "Stopped." "'$UI_CMD' continues from here; an interrupted test reruns."
  exit 130
}
trap on_interrupt INT TERM HUP

DEADLINE=""
[ -n "$BUDGET_MIN" ] && DEADLINE=$(( $(now_s) + BUDGET_MIN * 60 ))
export DEADLINE PAUSE_FLAG

# Clean stop between tests: pause requested or time budget used up.
gate() {
  local why=""
  [ -f "$PAUSE_FLAG" ] && why="Paused"
  [ -n "$DEADLINE" ] && [ "$(now_s)" -ge "$DEADLINE" ] && why="Time limit (--for $BUDGET_ARG) reached"
  [ -n "$why" ] || return 0
  rm -f "$PAUSE_FLAG"
  write_summary
  say "$why. Results so far are saved. Run '$UI_CMD' to continue from here."
  [ "$UI" = pretty ] && ui_end "$C_WARN" "$why." "Results so far are saved; '$UI_CMD' continues from here."
  exit 0
}

# ---------- tools: binaries, topology, CPU variant ----------
mkdir -p "$LOCAL" && touch "$LOCAL/.perseus-kit" || { echo "Can't create $LOCAL"; exit 1; }
dpm_restore  # a run cut off mid-step (kill -9, crash) may have left the GPU's clocks pinned
BIN="$LOCAL/bin"
VK_BIN="$BIN/stock-vk/nehalem"  # Vulkan build (the lab's GPU); nehalem only, see need_vk_build
# GPU drivers keep a shader cache on disk (Mesa: $HOME/.cache/mesa_shader_cache, /root/.cache under
# sudo). Keep it, and any other XDG cache, inside $LOCAL, which 'run.sh clean' removes.
MESA_SHADER_CACHE_DIR="$LOCAL/cache/mesa_shader_cache" XDG_CACHE_HOME="$LOCAL/cache"
export MESA_SHADER_CACHE_DIR XDG_CACHE_HOME
MEMBW=""
VARIANT=none
NODES=1 NCPUS=1 NPHYS=1 NODE0_NCPUS=1 NODE0_NPHYS=1 ALL_PHYS=0

setup_linux() {
  rm -rf "$BIN" && mkdir -p "$BIN" && cp -R "$KIT/bin/." "$BIN/" && chmod -R u+x "$BIN" \
    || { log "!! could not copy binaries to $BIN"; return 1; }
  MEMBW="$BIN/tools/membw"
  local topo
  if topo=$("$MEMBW" topo --shell 2>&1) && ! echo "$topo" | grep -qvE '^[A-Z0-9_]+=("[0-9, ]*"|-?[0-9]+)$'; then
    eval "$topo"
  else
    log "!! membw topo failed, assuming one NUMA node: $topo"
    NCPUS=$(getconf _NPROCESSORS_ONLN); NPHYS=$NCPUS; NODE0_NCPUS=$NCPUS; NODE0_NPHYS=$NCPUS
  fi
  local flags; flags=" $(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2) "
  has() { case "$flags" in *" $1 "*) return 0 ;; esac; return 1; }
  if has avx2 && has fma && has f16c && has bmi2; then VARIANT=haswell
  elif has sse4_2 && has popcnt; then VARIANT=nehalem
  elif has ssse3 && has sse4_1; then VARIANT=penryn
  fi
  case "${PERSEUS_VARIANT:-}" in
    '') ;;
    penryn|nehalem|haswell) VARIANT="$PERSEUS_VARIANT"; log "CPU build forced by PERSEUS_VARIANT: $VARIANT" ;;
    *) log "!! PERSEUS_VARIANT must be penryn, nehalem or haswell (got '$PERSEUS_VARIANT')"; exit 2 ;;
  esac
  # Prove the chosen build actually runs here; step down a level on SIGILL. A tiny real
  # benchmark exercises the SIMD kernels (--help alone might never reach them); without a model
  # (diag kit) one trial of the kernel self-test does.
  local tiny="$KIT/models/$TINY_Q40" rc
  while [ "$VARIANT" != none ]; do
    if [ -f "$tiny" ]; then "$BIN/stock/$VARIANT/llama-bench" -m "$tiny" -p 4 -n 2 -r 1 -t 2 >/dev/null 2>&1
    else "$BIN/stock/$VARIANT/test-ssse3-kernels" 1 >/dev/null 2>&1; fi
    rc=$?
    [ "$rc" != 132 ] && [ "$rc" != 126 ] && break
    log "!! $VARIANT build won't run here (exit $rc), trying an older one"
    case "$VARIANT" in haswell) VARIANT=nehalem ;; nehalem) VARIANT=penryn ;; *) VARIANT=none ;; esac
  done
}

setup_macos() {
  if command -v cc >/dev/null 2>&1 && cc -O2 -pthread -o "$LOCAL/membw" "$KIT/src/membw.c" 2>/dev/null; then
    MEMBW="$LOCAL/membw"
  fi
  NCPUS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 1); NPHYS=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 1)
  NODE0_NCPUS=$NCPUS; NODE0_NPHYS=$NPHYS
}

case "$OS" in
  Linux) setup_linux ;;
  Darwin) setup_macos ;;
esac
FIRST_PHYS=$(echo "$ALL_PHYS" | cut -d, -f1)

# ---------- model catalog ----------
N_MODELS=0
while IFS=$'\t' read -r tier runner file size sha url; do
  case "$tier" in ''|'#'*) continue ;; esac
  M_RUNNER[$N_MODELS]=$runner; M_FILE[$N_MODELS]=$file; M_SIZE[$N_MODELS]=$size; M_SHA[$N_MODELS]=$sha
  N_MODELS=$((N_MODELS + 1))
done < "$KIT/models.tsv"

model_index() { # file -> index, or nothing
  local k=0
  while [ "$k" -lt "$N_MODELS" ]; do [ "${M_FILE[$k]}" = "$1" ] && { echo "$k"; return; }; k=$((k + 1)); done
}

# Echoes the path to use for a model file, or nothing if it isn't usable.
model_path() {
  if [ -f "$LOCAL/models/$1.ok" ]; then echo "$LOCAL/models/$1"
  elif [ -f "$KIT/models/$1" ] && [ ! -f "$LOCAL/models/$1.bad" ]; then echo "$KIT/models/$1"
  fi
}

prefetch_alive() { pid_alive "$(cat "$LOCAL/models/.prefetch.pid" 2>/dev/null)"; }

start_prefetch() {
  [ "$OS" = Linux ] && [ "$NO_COPY" = 0 ] && [ "$VARIANT" != none ] || return 0
  # A copier left over from a crashed run would race this one on the same .part files.
  local old; old=$(cat "$LOCAL/models/.prefetch.pid" 2>/dev/null)
  if pid_alive "$old"; then kill -TERM -- "-$old" 2>/dev/null; sleep 1; fi
  local specs=() f i
  for f in $PREFETCH_ORDER; do
    [ -f "$KIT/models/$f" ] || continue
    i=$(model_index "$f"); [ -n "$i" ] || continue
    specs[${#specs[@]}]="$f ${M_SIZE[$i]} ${M_SHA[$i]}"
  done
  [ ${#specs[@]} -gt 0 ] || return 0  # diag kit: nothing to stage, nothing on the local disk
  command -v setsid >/dev/null 2>&1 || { log "!! setsid missing: models will be copied in the foreground"; return 0; }
  mkdir -p "$LOCAL/models"
  # .bad/.nospace from an earlier day get a fresh attempt (disk may have been freed, a USB
  # read error may have been transient); the copier re-marks them if they still fail.
  rm -f "$LOCAL/models/"*.part "$LOCAL/models/"*.bad "$LOCAL/models/"*.nospace "$LOCAL/models/.prefetch.pid" "$LOCAL/models/.copying"
  local pri=(nice -n 19)
  command -v ionice >/dev/null 2>&1 && pri=(nice -n 19 ionice -c 3)
  setsid "${pri[@]}" bash "$KIT/lib/prefetch.sh" "$KIT" "$LOCAL/models" "$RUN_DIR/prefetch.log" ${specs[@]+"${specs[@]}"} </dev/null >/dev/null 2>&1 &
  PREFETCH_PID=$!
  # Registered here too, not only by the copier: at idle priority it can start seconds later, and
  # a stage step that saw no copier would copy the same file into the same .part beside it.
  echo "$PREFETCH_PID" > "$LOCAL/models/.prefetch.pid"
}

mem_avail_kb() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }

ram_check() { # size_bytes
  local need_kb=$(( $1 / 1024 * 115 / 100 + 1572864 )) have_kb; have_kb=$(mem_avail_kb)
  [ "${have_kb:-0}" -ge "$need_kb" ] || skip "needs ~$((need_kb / 1048576)) GB free RAM, only $((have_kb / 1048576)) GB available"
}

# llama-bench spelled "don't mmap" differently across versions; ask the binary.
load_flag() { # llama-bench want(none|mmap)
  local help; help=$("$1" --help 2>&1)
  if echo "$help" | grep -q -- '--load-mode'; then echo "-lm $2"
  elif echo "$help" | grep -q -- '-mmp'; then [ "$2" = none ] && echo "-mmp 0" || echo "-mmp 1"
  fi
}

need_linux_llama() {
  [ "$OS" = Linux ] || skip "llama.cpp benchmarks need Linux (this kit carries Linux binaries)"
  [ "$VARIANT" != none ] || skip "no kit build runs on this CPU (needs at least SSSE3 + SSE4.1)"
}

# The USB decides which models a run uses: a local copy left by an earlier, bigger kit doesn't
# count, so a diag kit never runs model tests.
need_model() { # file -> sets MI (catalog index) and MP (path), or ends the step
  MI=$(model_index "$1"); [ -n "$MI" ] || skip "$1 is not in this kit's catalog"
  [ -f "$KIT/models/$1" ] || absent "$1 is not on this USB"
  MP=$(model_path "$1"); [ -n "$MP" ] || skip "the copy of $1 failed its checksum"
}

# Records what else was going on, so a slow number can be explained later. Every timed step calls
# it right before its command, after its own checks (model, GPU, tools): so a config that pins the
# GPU's clocks (step_cfg) is applied here, and a step that can't run never touches the GPU.
note_env() {
  if [ "${FN_PIN:-0}" = 1 ]; then dpm_set_high; FN_PINNED=1; fi
  { echo "loadavg: $(cat /proc/loadavg)"
    echo "background copy: $(cat "$LOCAL/models/.copying" 2>/dev/null || echo none)"
    echo "overrides: $(env | grep -E '^(PERSEUS|GGML)_' | tr '\n' ' ')"
    echo "gpu performance level: $(dpm_level)"
  } > "$STEP_DIR/env.txt"
}

# ---------- steps ----------
step_env() {
  echo "utc: $(ts)"; echo "host: $HOST"; echo "os: $OS"; echo "id: $(id)"
  echo "variant: $VARIANT"; echo "local: $LOCAL"; echo "budget_min: ${BUDGET_MIN:-none}"
  echo "nodes=$NODES ncpus=$NCPUS nphys=$NPHYS node0: ncpus=$NODE0_NCPUS nphys=$NODE0_NPHYS"
  echo "manifest:"; cat "$KIT/MANIFEST.json" 2>/dev/null || echo "(no MANIFEST.json)"
  [ -n "$MEMBW" ] && "$MEMBW" topo
  return 0
}

# Each command's output goes to its own file; _index.tsv records every exit code, so a
# missing tool shows up as a non-zero row instead of vanishing.
cap() { # name command-string
  local t0; t0=$(now_s)
  run_cmd 120 sh -c "$2" > "$STEP_DIR/$1.txt" 2>&1
  printf '%s\t%s\t%ss\n' "$1" "$?" "$(( $(now_s) - t0 ))" >> "$STEP_DIR/_index.tsv"
}

step_sysinfo() {
  if [ "$OS" = Darwin ]; then
    cap sw_vers 'sw_vers; uname -a'
    cap sysctl-hw 'sysctl hw; sysctl machdep.cpu'
    cap profiler 'system_profiler SPHardwareDataType SPMemoryDataType SPStorageDataType SPSoftwareDataType SPUSBDataType SPPCIDataType'
    cap df 'df -h'
    cap vm 'vm_stat; pmset -g'
    cap tools 'for t in gcc cc clang cmake make git python3 brew; do printf "%-10s %s\n" $t "$(command -v $t || echo MISSING)"; done'
    cap load 'uptime; ps aux -r | head -15'
    return 0
  fi
  cap uname 'uname -a'
  cap os-release 'cat /etc/os-release'
  cap dmi-id 'cd /sys/class/dmi/id && for f in sys_vendor product_name product_version board_vendor board_name bios_vendor bios_version bios_date; do printf "%s: " $f; cat $f; done'
  cap cpuinfo 'cat /proc/cpuinfo'
  cap lscpu 'lscpu; echo; lscpu -e'
  cap meminfo 'cat /proc/meminfo'
  cap free 'free -m'
  cap numa-nodes 'for n in /sys/devices/system/node/node*; do echo "== $n"; echo "cpus: $(cat $n/cpulist)"; echo "distance: $(cat $n/distance)"; grep -E "MemTotal|MemFree" $n/meminfo; done'
  cap dmidecode-memory 'dmidecode -t 16 -t 17'
  cap dmidecode-system 'dmidecode -t 0 -t 1 -t 2 -t 4'
  cap edac 'for m in /sys/devices/system/edac/mc/mc*; do echo "== $m"; for f in mc_name size_mb ce_count ue_count; do printf "%s: " $f; cat $m/$f; done; done'
  cap cpufreq 'cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c; for f in cpuinfo_min_freq cpuinfo_max_freq scaling_driver; do printf "%s: " $f; cat /sys/devices/system/cpu/cpu0/cpufreq/$f; done; grep MHz /proc/cpuinfo | sort | uniq -c'
  cap smt 'cat /sys/devices/system/cpu/smt/control /sys/devices/system/cpu/smt/active'
  cap vm-knobs 'for f in /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag /proc/sys/kernel/numa_balancing /proc/sys/vm/zone_reclaim_mode /proc/sys/vm/swappiness /proc/sys/vm/overcommit_memory; do printf "%s: " $f; cat $f; done; grep -i huge /proc/meminfo'
  cap swap 'cat /proc/swaps'
  cap ulimit 'ulimit -a'
  cap disks 'lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL,ROTA,TRAN'
  cap df 'df -hT'
  cap pci 'lspci -nn'
  # GPU: board maker (subsystem ids), VRAM, clocks, PCIe link, kernel driver, who may open the
  # render node, and which Vulkan drivers are installed (the GPU steps need Mesa's RADV)
  cap gpu 'lspci -nnk | grep -A4 -E "VGA|3D controller|Display controller"; for c in /sys/class/drm/card[0-9]*; do [ -e $c/device/vendor ] || continue; echo "== $c"; for f in vendor device subsystem_vendor subsystem_device revision mem_info_vram_total mem_info_vram_used mem_info_vis_vram_total current_link_speed current_link_width max_link_speed max_link_width power_dpm_force_performance_level pp_dpm_sclk pp_dpm_mclk; do printf "%s: " $f; tr "\n" " " < $c/device/$f; echo; done; done'
  cap vulkan 'ls -l /dev/dri; getfacl -p /dev/dri/renderD* 2>&1; id; ls -l /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1; dpkg -l 2>/dev/null | grep -i -E "vulkan|mesa|amdgpu"; vulkaninfo --summary'
  cap usb 'lsusb; lsusb -t'
  cap glibc 'ldd --version'
  cap tools 'for t in gcc cc clang cmake make git python3 numactl perf dmidecode lscpu sha256sum timeout sudo setsid ionice; do printf "%-10s %s\n" $t "$(command -v $t || echo MISSING)"; done'
  cap cmdline 'cat /proc/cmdline'
  cap dmesg-hw 'dmesg | grep -iE "numa|edac|mce|dmi:|memory|microcode" | head -300'
  cap load 'uptime; cat /proc/loadavg; ps aux --sort=-%cpu | head -15'
  cap sensors 'sensors'
  return 0
}

# Perseus SSSE3 kernels (kernels/test_*.c) against llama.cpp's generic reference, on this CPU.
step_kernel_selftest() { # test binary
  need_linux_llama
  local t="$BIN/stock/$VARIANT/$1"
  [ -x "$t" ] || skip "$1 not in this kit"
  run_cmd 300 "$t" 100
}

# Per-format kernel cost on this CPU (1 thread, weights in L2): which formats are compute-bound
# here, and what the multi-column kernels save per extra token. kernels/gap.py reads every
# *-kernel-bench step.
step_kernel_bench() { # test binary
  need_linux_llama
  local t="$BIN/stock/$VARIANT/$1"
  [ -x "$t" ] || skip "$1 not in this kit"
  note_env
  run_cmd 300 "$t" bench
}

# The IQ3_S load-time repack against llama.cpp's own path, through ggml's graph API, on this CPU.
step_repack_selftest() {
  need_linux_llama
  local t="$BIN/stock/$VARIANT/test-repack-mulmat"
  [ -x "$t" ] || skip "no repack self-test for this build (the IQ3_S repack is for pre-AVX CPUs)"
  run_cmd 300 "$t"
}

# Lookup-table kernels (the T-MAC idea) vs the multiply kernel for 2-bit weights, on this CPU.
step_lut_bench() {
  need_linux_llama
  local t="$BIN/stock/$VARIANT/test-lut"
  [ -x "$t" ] || skip "lookup-table bench not in this kit"
  note_env
  run_cmd 300 "$t"
}

# llama.cpp's tiled matmul (prompt batches of 8+) against its vec_dot path, on this CPU.
step_tiled_selftest() {
  need_linux_llama
  local t="$BIN/stock/$VARIANT/test-tiled-mulmat"
  [ -x "$t" ] || skip "tiled matmul self-test not in this kit"
  run_cmd 300 "$t"
}

step_membw() { # label args...
  [ -n "$MEMBW" ] || skip "membw not available (no compiler on macOS?)"
  local label="$1"; shift
  run_cmd 300 "$MEMBW" read --label "$label" --mb 2048 --secs 2 --reps 3 "$@"
}

step_membw_multi() { # label args... (needs >1 NUMA node)
  [ "$NODES" -gt 1 ] || skip "single NUMA node"
  step_membw "$@"
}

# Waits for the background copier to stage the model; copies it here if the copier isn't
# running. Pause/stop/time limit while waiting ends the step as interrupted (reruns later).
step_stage() { # file
  need_linux_llama
  local f="$1" i; i=$(model_index "$f")
  [ -n "$i" ] || skip "$f is not in this kit's catalog"
  [ -f "$KIT/models/$f" ] || absent "$f is not on this USB"
  [ "$NO_COPY" = 1 ] && skip "--no-copy: running from USB"
  local waited=0 dst="$LOCAL/models/$f"
  while :; do
    [ -f "$dst.ok" ] && { echo "staged (waited ${waited}s for the background copy)"; return 0; }
    [ -f "$dst.bad" ] && { echo "checksum mismatch for $f: the USB copy is damaged" >&2; return 1; }
    [ -f "$dst.nospace" ] && skip "local disk too full to copy; running from USB (slow loads)"
    prefetch_alive || break
    [ -f "$PAUSE_FLAG" ] && exit 130
    [ -n "$DEADLINE" ] && [ "$(now_s)" -ge "$DEADLINE" ] && exit 130
    sleep 3; waited=$((waited + 3))
  done
  copy_model "$LOCAL/models" "$KIT/models" "$f" "${M_SIZE[$i]}" "${M_SHA[$i]}"
  case $? in
    0) return 0 ;;
    3) skip "local disk too full to copy; running from USB (slow loads)" ;;
    *) return 1 ;;
  esac
}

step_sanity() { # file
  need_linux_llama
  need_model "$1"
  ram_check "${M_SIZE[$MI]}"
  # -c matters: these tools default to the model's trained context (262k on Qwen3.8),
  # which reserves ~17 GB of KV cache. llama-bench sizes its own context.
  run_cmd 2400 "$BIN/${M_RUNNER[$MI]}/$VARIANT/llama-completion" -m "$MP" -c 512 -p "$PROMPT" -n 64 --temp 0 --seed 1 -no-cnv -t "$NPHYS" < /dev/null
}

# mode: all-sockets (interleaved memory + threads on both sockets; plain on 1-socket boxes),
#       one-socket, naive (stock llama.cpp defaults), distribute-cold (mmap after dropping caches)
# threads: cores (one per physical core) | smt (all hardware threads)
step_bench() { # file mode threads n_prompt n_gen reps
  need_linux_llama
  need_model "$1"
  local mode="$2" np="$4" ng="$5" reps="$6" t
  ram_check "${M_SIZE[$MI]}"
  if [ "$3" = smt ]; then
    [ "$NCPUS" -gt "$NPHYS" ] || skip "no hyperthreads on this CPU"
    t=$NCPUS; [ "$mode" = one-socket ] && t=$NODE0_NCPUS
  else
    t=$NPHYS; [ "$mode" = one-socket ] && t=$NODE0_NPHYS
  fi
  local lb="$BIN/${M_RUNNER[$MI]}/$VARIANT/llama-bench"
  local common=(-m "$MP" -p "$np" -n "$ng" -r "$reps" -t "$t" -o jsonl)
  note_env
  case "$mode" in
    all-sockets)
      if [ "$NODES" -gt 1 ]; then
        echo "interleave" > "$STEP_DIR/mode.txt"
        run_cmd 3600 "$MEMBW" exec --interleave -- "$lb" "${common[@]}" --numa distribute $(load_flag "$lb" none)
      else
        echo "single-node default" > "$STEP_DIR/mode.txt"
        run_cmd 3600 "$lb" "${common[@]}" $(load_flag "$lb" none)
      fi ;;
    naive)
      [ "$NODES" -gt 1 ] || skip "same as all-sockets on a single-socket machine"
      run_cmd 3600 "$lb" "${common[@]}" $(load_flag "$lb" none) ;;
    one-socket)
      [ "$NODES" -gt 1 ] || skip "single NUMA node"
      run_cmd 3600 "$MEMBW" exec --membind 0 --cpunode 0 -- "$lb" "${common[@]}" --numa numactl $(load_flag "$lb" none) ;;
    distribute-cold)
      [ "$NODES" -gt 1 ] || skip "single NUMA node"
      [ "$(id -u)" = 0 ] || skip "needs sudo (drops the page cache so pages land near the threads that use them)"
      sync && echo 3 > /proc/sys/vm/drop_caches || return 1
      run_cmd 3600 "$lb" "${common[@]}" --numa distribute $(load_flag "$lb" mmap) ;;
  esac
}

# Same prompt and settings with and without MTP drafting, so the speedup is like-for-like.
step_mtp() { # file draft_file|"" n_max(0 = MTP off)
  need_linux_llama
  need_model "$1"
  local m="$MP" i="$MI" n="$3" d="" need="${M_SIZE[$MI]}"
  if [ -n "$2" ]; then
    need_model "$2"
    d="$MP"; need=$(( need + ${M_SIZE[$MI]} ))
  fi
  ram_check "$need"
  note_env
  local wrap=() numa=()
  if [ "$NODES" -gt 1 ]; then wrap=("$MEMBW" exec --interleave --); numa=(--numa distribute); fi
  local bdir="$BIN/${M_RUNNER[$i]}/$VARIANT"
  local gen=(-c 1024 -p "$PROMPT_LONG" -n 128 --temp 0 --top-k 1 --seed 42 -t "$NPHYS")
  # ${arr[@]+...} keeps bash 3.2 with set -u happy when the array is empty.
  if [ "$n" = 0 ]; then
    run_cmd 3600 ${wrap[@]+"${wrap[@]}"} "$bdir/llama-completion" -m "$m" "${gen[@]}" -no-cnv ${numa[@]+"${numa[@]}"} < /dev/null
  else
    run_cmd 3600 ${wrap[@]+"${wrap[@]}"} "$bdir/llama-speculative-simple" -m "$m" -md "$d" \
      --spec-type draft-mtp --spec-draft-n-max "$n" "${gen[@]}" ${numa[@]+"${numa[@]}"} < /dev/null
  fi
}

# ---------- GPU (Vulkan build) ----------
# Ends the step unless this box can start the Vulkan build: it is the nehalem build (SSE4.2), and
# it links Ubuntu 24.04's glibc (glibc-min: the newest symbol version it uses).
need_vk_build() {
  [ -x "$VK_BIN/llama-completion" ] || skip "no Vulkan build in this kit"
  case "$VARIANT" in nehalem|haswell) ;; *) skip "the Vulkan build needs SSE4.2; this CPU runs the $VARIANT build" ;; esac
  local need have
  need=$(cat "$VK_BIN/glibc-min" 2>/dev/null)
  case "$need" in 2.[0-9]*) ;; *) echo "kit error: $VK_BIN/glibc-min is missing or malformed ('$need')" >&2; exit 1 ;; esac
  have=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')
  [ -n "$have" ] || skip "can't tell this box's glibc version; the Vulkan build needs glibc $need"
  [ "$(printf '%s\n%s\n' "$need" "$have" | sort -V | head -1)" = "$need" ] \
    || skip "the Vulkan build needs glibc $need or newer; this box has $have"
}

# Picks the GPU every later GPU step uses: the first discrete Vulkan device (uma: 0), so an iGPU
# or a software rasteriser never stands in for it. Records it by llama.cpp's device name (VulkanN,
# what -dev / -b take) in $RUN_DIR/gpu.env, and its full name in gpu-name.txt. Not by
# GGML_VK_VISIBLE_DEVICES: that indexes Vulkan's raw device list, which can differ from VulkanN.
step_gpu_probe() {
  need_linux_llama
  need_vk_build
  rm -f "$RUN_DIR/gpu.env" "$RUN_DIR/gpu-name.txt"
  unset GGML_VK_VISIBLE_DEVICES  # llama.cpp's default device list, the one VulkanN names refer to
  run_cmd 120 "$VK_BIN/llama-bench" --list-devices > "$STEP_DIR/devices.txt" 2>&1
  local rc=$?
  cat "$STEP_DIR/devices.txt"
  [ "$rc" = 0 ] || return "$rc"
  # ggml_vulkan: 0 = AMD Radeon RX 580 Series (RADV POLARIS10) (radv) | uma: 0 | fp16: ...
  #   Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10) (8192 MiB, 7930 MiB free)
  local idx line name total free
  idx=$(sed -n 's/^ggml_vulkan: \([0-9][0-9]*\) = .* | uma: 0 | .*/\1/p' "$STEP_DIR/devices.txt" | head -1)
  if [ -z "$idx" ]; then
    local seen; seen=$(grep -E '^ggml_vulkan: [0-9]+ = ' "$STEP_DIR/devices.txt" | tr '\n' ' ' | head -c 400)
    [ -n "$seen" ] || skip "no Vulkan devices: no GPU driver, or no access to /dev/dri (see steps/1-02-sysinfo/vulkan.txt)"
    skip "no discrete Vulkan GPU, only: $seen"
  fi
  line=$(grep -E "^ *Vulkan$idx: " "$STEP_DIR/devices.txt" | head -1)
  name=$(gpu_line_name "$line")
  total=$(echo "$line" | sed -n 's/.*(\([0-9][0-9]*\) MiB, [0-9][0-9]* MiB free)$/\1/p')
  free=$(echo "$line" | sed -n 's/.*([0-9][0-9]* MiB, \([0-9][0-9]*\) MiB free)$/\1/p')
  [ -n "$name" ] && [ -n "$total" ] && [ -n "$free" ] || { echo "can't read name/VRAM from: $line" >&2; return 1; }
  printf '%s\n' "$name" > "$RUN_DIR/gpu-name.txt"
  printf 'GPU_DEV=Vulkan%s\nGPU_VRAM_MIB=%s\nGPU_FREE_MIB=%s\n' "$idx" "$total" "$free" > "$RUN_DIR/gpu.env"
  echo "picked: $line"
}

# "  Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10) (8192 MiB, 7930 MiB free)" -> the name
gpu_line_name() { echo "$1" | sed -n 's/^ *Vulkan[0-9][0-9]*: \(.*\) ([0-9][0-9]* MiB, [0-9][0-9]* MiB free)$/\1/p'; }

# For GPU steps: ends the step unless 3-01 picked a GPU, and fails it unless that GPU is still the
# one llama.cpp calls $GPU_DEV (a driver gone or a device added since 3-01, e.g. across a resume,
# must not turn into CPU numbers or a different GPU's). Commands select it with -dev / -b $GPU_DEV.
need_gpu() {
  # Clocks still pinned by an earlier step whose restore failed (see run.log): numbers taken now
  # would be at the wrong level, so the step waits (interrupted: it reruns next time).
  if [ -f "$DPM_RECORD" ]; then
    echo "the GPU's clocks are still pinned (putting them back failed, see run.log): deferred to the next run" > "$STEP_DIR/reason"
    exit 130
  fi
  need_vk_build
  [ -f "$RUN_DIR/gpu.env" ] || skip "no usable GPU (see steps/3-01-gpu-probe)"
  # Our own file, but check its shape before sourcing it.
  if grep -qvE '^GPU_DEV=Vulkan[0-9]+$|^GPU_(VRAM|FREE)_MIB=[0-9]+$' "$RUN_DIR/gpu.env"; then
    echo "malformed $RUN_DIR/gpu.env" >&2; exit 1
  fi
  . "$RUN_DIR/gpu.env"
  unset GGML_VK_VISIBLE_DEVICES  # VulkanN names refer to llama.cpp's default device list
  local want now rc
  want=$(cat "$RUN_DIR/gpu-name.txt" 2>/dev/null)
  # bounded like every command (a driver stuck after a GPU reset can hang device listing)
  run_cmd 120 "$VK_BIN/llama-bench" --list-devices > "$STEP_DIR/devices-now.txt" 2>&1; rc=$?
  case "$rc" in 124|137) echo "listing Vulkan devices hung (120 s)" >&2; exit "$rc" ;; esac
  now=$(gpu_line_name "$(grep -E "^ *$GPU_DEV: " "$STEP_DIR/devices-now.txt" | head -1)")
  [ -n "$want" ] && [ "$now" = "$want" ] || { echo "3-01 picked $GPU_DEV = '$want'; now $GPU_DEV is '${now:-not there}'" >&2; exit 1; }
}

# test-backend-ops on the picked GPU. Its exit status alone proves nothing: it returns 0 when the
# named backend isn't there (it skips it), so the output must show that backend ran real cases.
gpu_tbo() { # timeout mode args...
  local t="$1" mode="$2" out="$STEP_DIR/tbo.out" rc; shift 2
  run_cmd "$t" "$VK_BIN/test-backend-ops" "$mode" -b "$GPU_DEV" "$@" > "$out" 2>&1; rc=$?
  cat "$out"
  [ "$rc" = 0 ] || return "$rc"
  local plain; plain=$(sed 's/\x1b\[[0-9;]*m//g' "$out")
  echo "$plain" | grep -qx "  Backend $GPU_DEV: OK" || { echo "test-backend-ops did not run on $GPU_DEV" >&2; return 1; }
  case "$mode" in
    test) echo "$plain" | grep -qE '^ +([1-9][0-9]*)/\1 tests passed$' || { echo "no test cases ran on $GPU_DEV" >&2; return 1; } ;;
    # perf mode carries on past a case that fails and still exits 0
    perf) echo "$plain" | grep -q ' GB/s' || { echo "no op was timed on $GPU_DEV" >&2; return 1; }
          if echo "$plain" | grep -qE 'failed to allocate|graph_compute failed'; then echo "a perf case failed on $GPU_DEV" >&2; return 1; fi ;;
  esac
}

# The GPU against the CPU for every op type the GPU will run (test-backend-ops: same op, same
# inputs, results compared). gpu-ops/MODEL.txt (made by build-kit.sh) replays the model's own ops
# and shapes; the op types a random-input replay can't judge (kit/build/gpu_ops.py says why) are in
# MODEL.txt.builtin, and llama.cpp's own cases for those ops and weight types run instead.
step_gpu_selftest() {
  need_linux_llama
  ls "$KIT"/gpu-ops/*.txt >/dev/null 2>&1 || skip "no GPU op lists in this kit (gpu-ops/)"
  need_gpu
  local f op filt left_out worst=0
  keep() { [ "$1" = 0 ] || { [ "$worst" = 124 ] || [ "$worst" = 137 ] || worst=$1; }; }
  # without its .builtin the ops a replay leaves out would go unchecked: a damaged kit, not a pass
  for f in "$KIT"/gpu-ops/*.txt; do
    [ -r "$f.builtin" ] || { echo "gpu-ops/$(basename "$f").builtin is missing or unreadable" >&2; return 1; }
  done
  left_out=$(sort -u "$KIT"/gpu-ops/*.txt.builtin) || { echo "can't read gpu-ops/*.builtin" >&2; return 1; }
  for f in "$KIT"/gpu-ops/*.txt; do
    echo "== ops of $(basename "$f" .txt)"
    gpu_tbo 1800 test -j "$NPHYS" --test-file "$f"; keep $?
  done
  while IFS="$(printf '\t')" read -r op filt; do
    [ -n "$op$filt" ] || continue  # no ops left out
    case "$op" in ''|*[!A-Z0-9_]*) echo "malformed gpu-ops/*.builtin line: $op" >&2; return 1 ;; esac
    echo "== llama.cpp's $op cases${filt:+ ($filt)}"
    if [ "$filt" = - ]; then gpu_tbo 1800 test -j "$NPHYS" -o "$op"; else gpu_tbo 1800 test -j "$NPHYS" -o "$op" -p "$filt"; fi
    keep $?
  done <<EOF_LEFT_OUT
$left_out
EOF_LEFT_OUT
  return "$worst"
}

# Time and GB/s of every matmul Flash-Next runs on the GPU, one op at a time (perf mode).
step_gpu_op_bench() {
  need_linux_llama
  need_gpu
  local f="$KIT/gpu-ops/${FLASHNEXT%.gguf}.txt"
  [ -f "$f" ] || skip "no Flash-Next op list in this kit (gpu-ops/)"
  note_env
  gpu_tbo 900 perf --test-file "$f" -o MUL_MAT
}

# A model's whole graph on the GPU: readable text means the GPU path works end to end. -lv 4 logs
# where the weights went ("offloaded N/N layers to GPU").
step_gpu_sanity() { # file
  need_linux_llama
  need_model "$1"
  need_gpu
  run_cmd 1200 "$VK_BIN/llama-completion" -m "$MP" -dev "$GPU_DEV" -ngl 99 -c 512 -p "$PROMPT" -n 64 --temp 0 --seed 1 -no-cnv -t "$NPHYS" -lv 4 < /dev/null
}

step_gpu_bench() { # file n_prompt n_gen reps
  need_linux_llama
  need_model "$1"
  need_gpu
  note_env
  run_cmd 1800 "$VK_BIN/llama-bench" -m "$MP" -dev "$GPU_DEV" -ngl 99 -p "$2" -n "$3" -r "$4" -t "$NPHYS" -o jsonl
}

# Flash-Next's two files, staged side by side: sets MP (the file to open) and FN_NEED (bytes).
need_flashnext() {
  need_linux_llama
  need_model "$FLASHNEXT_NGRAM"
  local ngram="$MP" n="${M_SIZE[$MI]}"
  need_model "$FLASHNEXT"
  FN_NEED=$(( n + ${M_SIZE[$MI]} ))
  # llama.cpp finds the n-gram file by name next to the first one
  [ "$(dirname "$ngram")" = "$(dirname "$MP")" ] || skip "Flash-Next's two files are in different places ($(dirname "$MP"), $(dirname "$ngram")): one was staged, the other not"
  need_vk_build
}

# Flash-Next, same binary, prompt, context, threads and greedy settings; only the GPU differs.
#   gpu: llama.cpp's --fit fills the GPU back to front with each layer's attention, shared experts
#        and router (routed experts stay in RAM) as far as free VRAM goes; -lv 4 logs its choices
#   cpu: -dev none, so nothing reaches the GPU (with -ngl 0 the prompt's matmuls still would)
# mmap (the default), not -lm none: a 75.8 GB model read into RAM would sit beside its page cache.
step_flashnext() { # gpu|cpu
  need_flashnext  # the Vulkan build runs both halves of this A/B
  local dev
  # cpu: Vulkan not even loaded, so a GPU driver left wedged by an earlier step can't take the
  # CPU baseline down with it
  if [ "$1" = gpu ]; then need_gpu; dev=(-dev "$GPU_DEV" -fit on -lv 4); else dev=(-dev none); GGML_DISABLE_VULKAN=1; export GGML_DISABLE_VULKAN; fi
  ram_check "$FN_NEED"
  note_env
  run_cmd 3600 "$VK_BIN/llama-completion" -m "$MP" "${dev[@]}" -c 4096 -p "$PROMPT_LONG" -n 128 \
    --temp 0 --top-k 1 --seed 42 -t "$NPHYS" -no-cnv < /dev/null
}

# ---------- phase 4b: Flash-Next profile ----------
# Each GPU step: everything but the routed experts on the GPU (8 GB holds that: 5.4 GB), the experts
# in RAM. Fixed instead of -fit: llama-bench re-fits by freeing and reloading the model for every
# test it runs, and a fixed split keeps the A/B steps comparable.
FN_GPU_SPLIT="-ngl 99 -ot exps=CPU"

# Runs a step under one of 4b's configs (see the plan). best = what 4b-05 picked, for Flash-Next
# steps only: Flash-Next not on this USB ends it absent (it runs once a USB has it); otherwise
# nothing picked ends it skipped, never quietly run as base. dpm-high pins the GPU's clocks for this
# step only, from note_env (after the step's own checks); the runner puts them back when the step
# ends, however it ends (lib/gpu.sh). A step that never reached note_env can't report pinned numbers.
step_cfg() { # base|novis|dpm-high|novis-dpm-high|best step args...
  local cfg="$1" rc=0; shift
  if [ "$cfg" = best ]; then
    need_flashnext
    [ -f "$RUN_DIR/fn-best.env" ] || skip "4b-05 picked no config"
    cfg=$(sed -n 's/^FN_BEST=//p' "$RUN_DIR/fn-best.env")
  fi
  echo "$cfg" > "$STEP_DIR/config.txt"
  case "$cfg" in
    base) ;;
    novis) export GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 ;;
    dpm-high) FN_PIN=1 ;;
    novis-dpm-high) export GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1; FN_PIN=1 ;;
    *) echo "unknown config '$cfg' (fn-best.env damaged?)" >&2; exit 1 ;;
  esac
  "$@" || rc=$?
  if [ "${FN_PIN:-0}" = 1 ] && [ "${FN_PINNED:-0}" != 1 ] && [ "$rc" = 0 ]; then
    echo "kit error: $cfg asked for pinned clocks but $1 never applied them (no note_env)" >&2; return 1
  fi
  return "$rc"
}

# llama-bench on Flash-Next. gpu: the fixed split, with the GPU's busy % and clocks sampled into
# gpu-samples.txt; cpu: nothing on the GPU, Vulkan not loaded (as 4-04).
step_fn_bench() { # gpu|cpu llama-bench args...
  local where="$1" dev rc=0; shift
  need_flashnext
  if [ "$where" = gpu ]; then need_gpu; dev=(-dev "$GPU_DEV" $FN_GPU_SPLIT)
  else dev=(-dev none); GGML_DISABLE_VULKAN=1; export GGML_DISABLE_VULKAN; fi
  ram_check "$FN_NEED"
  note_env
  [ "$where" = gpu ] && gpu_sampler_start "$STEP_DIR/gpu-samples.txt"
  run_cmd 3600 "$VK_BIN/llama-bench" -m "$MP" "${dev[@]}" "$@" -o jsonl < /dev/null || rc=$?
  gpu_sampler_stop
  return "$rc"
}

# The fastest decode (tg) of the given steps that finished ok, into $RUN_DIR/fn-best.env for the
# 'best' steps. Flash-Next not on this USB: absent, like them. A bench that ran (ok without a
# number, failed, timeout, crashed) yet none has a number: failed. All of them skipped (no GPU):
# skipped. Either way the 'best' steps then skip.
step_fn_pick() { # step-id:config...
  rm -f "$RUN_DIR/fn-best.env"
  need_flashnext
  local spec id cfg st ts best="" best_ts=0 ran=0
  for spec in "$@"; do
    id=${spec%%:*} cfg=${spec#*:}
    st=$(cat "$RUN_DIR/steps/$id/status" 2>/dev/null)
    ts=$(grep '"n_prompt": 0,' "$RUN_DIR/steps/$id/stdout.txt" 2>/dev/null | sed -n 's/.*"avg_ts": *\([0-9.]*\).*/\1/p' | head -1)
    echo "$cfg ($id, ${st:-not run}): ${ts:-no decode number} tok/s"
    case "$st" in ok|failed|timeout|crashed) ran=1 ;; esac
    [ "$st" = ok ] && [ -n "$ts" ] || continue
    if awk -v a="$ts" -v b="$best_ts" 'BEGIN { exit !(a > b) }'; then best=$cfg best_ts=$ts; fi
  done
  if [ -z "$best" ]; then
    [ "$ran" = 1 ] || skip "none of the benches to pick from ran (see their reasons)"
    echo "no decode number from any of: $*" >&2; return 1
  fi
  echo "FN_BEST=$best" > "$RUN_DIR/fn-best.env"
  echo "picked: $best ($best_ts tok/s)"
}

# 64 tokens of Flash-Next through llama-completion, the fixed split, profiled.
#   gpu-ops: ggml-vulkan's per-op GPU timer (GGML_VK_PERF_LOGGER), for every graph (warmup and prompt
#            included). It puts a barrier between ops, so the sum overstates the GPU's share; use it
#            for the split between ops.
#   perf:    where the CPU threads spend their time while generating (perf record, 999 Hz), reported
#            here with symbols (the Vulkan build's llama-completion keeps them). Needs sudo on Ubuntu.
#            -N: no build-id cache in root's home; the kit writes nothing outside $LOCAL.
step_fn_profile() { # gpu-ops|perf
  need_flashnext
  need_gpu
  ram_check "$FN_NEED"
  local gen=(-m "$MP" -dev "$GPU_DEV" $FN_GPU_SPLIT -c 4096 -p "$PROMPT_LONG" -n 64 --temp 0 --top-k 1 --seed 42 -t "$NPHYS" -no-cnv)
  case "$1" in
    gpu-ops)
      GGML_VK_PERF_LOGGER=1; export GGML_VK_PERF_LOGGER
      note_env
      run_cmd 1800 "$VK_BIN/llama-completion" "${gen[@]}" < /dev/null ;;
    perf)
      command -v perf >/dev/null 2>&1 || skip "perf is not installed"
      local v; v=$(perf version 2>&1) || skip "perf doesn't run here: $(echo "$v" | tr '\n' ' ' | head -c 200)"
      fn_perf_probe
      note_env
      fn_perf_attach
      local rc=0 prc=0
      run_cmd 1800 "$VK_BIN/llama-completion" "${gen[@]}" < /dev/null || rc=$?
      touch "$STEP_DIR/.lc-done"; wait "$PERF_ATTACH" || prc=$?
      [ "$rc" = 0 ] || return "$rc"
      [ "$prc" = 0 ] || { echo "perf record exited $prc (see perf-record.txt)" >&2; return 1; }
      [ -s "$STEP_DIR/perf.data" ] || { echo "perf recorded nothing (see perf-record.txt)" >&2; return 1; }
      run_cmd 600 perf report -i "$STEP_DIR/perf.data" --stdio --no-children --sort dso,sym --percent-limit 0.2 > "$STEP_DIR/report-sym.txt" 2>&1 || return $?
      run_cmd 600 perf report -i "$STEP_DIR/perf.data" --stdio --no-children --sort pid,dso --percent-limit 0.5 > "$STEP_DIR/report-thread.txt" 2>&1 ;;
    *) echo "unknown profile '$1'" >&2; return 1 ;;
  esac
}

# Can perf attach to another process here, as 4b-09 will? Ubuntu forbids it without sudo
# (perf_event_paranoid, Yama ptrace scope). Tried on a 1 s 'sleep', so a refusal skips the step
# before Flash-Next loads (30-190 s).
fn_perf_probe() {
  local t out rc=0
  sleep 1 & t=$!
  out=$(perf record -q -N -F 99 -p "$t" -o "$STEP_DIR/perf-probe.data" 2>&1) || rc=$?
  kill "$t" 2>/dev/null; wait "$t" 2>/dev/null; rm -f "$STEP_DIR/perf-probe.data"
  [ "$rc" = 0 ] || skip "perf can't attach to a process here (sudo bash run.sh): $(echo "$out" | tr '\n' ' ' | head -c 160)"
}

# Attaches perf to the step's llama-completion once it has loaded and warmed up (its "generate:"
# line), so the profile is generation (17 prompt tokens + 64 decoded), not 30-190 s of loading.
# perf record -p ends when that process ends. In the background, where SIGINT is ignored: the
# benchmark stays in the foreground for Ctrl-C, and this loop polls its step and a done flag.
fn_perf_attach() {
  local step=$BASHPID
  rm -f "$STEP_DIR/.lc-done"
  (
    until grep -q 'generate: n_ctx' "$STEP_DIR/stderr.txt" 2>/dev/null; do
      [ -f "$STEP_DIR/.lc-done" ] && exit 0
      kill -0 "$step" 2>/dev/null || exit 0
      sleep 0.2
    done
    # this step's own llama-completion: run_cmd's 'timeout' child, or the step's direct child
    t=$(pgrep -P "$step" -x timeout | head -1)
    if [ -n "$t" ]; then pid=$(pgrep -P "$t" | head -1); else pid=$(pgrep -P "$step" -f llama-completion | head -1); fi
    [ -n "$pid" ] || { echo "llama-completion logged 'generate:' but its pid wasn't found"; exit 0; }
    echo "perf record -F 999 -N -p $pid -o $STEP_DIR/perf.data"
    exec perf record -F 999 -N -p "$pid" -o "$STEP_DIR/perf.data"
  ) > "$STEP_DIR/perf-record.txt" 2>&1 &
  PERF_ATTACH=$!
}

# A/B variants of the steps above (the kernels/ patches read these at startup)
step_mtp_mc_off() { export PERSEUS_MC=0; step_mtp "$@"; }
step_bench_tiled_off() { export GGML_CPU_TILED_MM=0; step_bench "$@"; }
step_bench_repack_off() { export PERSEUS_REPACK=0; step_bench "$@"; }

# ---------- go ----------
# A dry pass over the plan for the screen: step count, widest id, steps done on an earlier day.
UI_DONE=0
phase() { :; }
S() {
  UI_TOTAL=$((UI_TOTAL + 1))
  [ ${#1} -gt "$UI_IDW" ] && UI_IDW=${#1}
  case "$(cat "$RUN_DIR/steps/$1/status" 2>/dev/null)" in ok|skipped) UI_DONE=$((UI_DONE + 1)) ;; esac
}
plan
ui_fit_live
phase() { ui_phase "$@"; }
S() {
  UI_N=$((UI_N + 1))
  gate; run_step "$@"
  dpm_restore
  if [ "$1" = "${PERSEUS_PAUSE_AFTER:-}" ]; then touch "$PAUSE_FLAG"; fi
}

UI_T0=$(now_s)
ON_USB=0
for f in ${M_FILE[@]+"${M_FILE[@]}"}; do [ -f "$KIT/models/$f" ] && ON_USB=$((ON_USB + 1)); done
say "perseus lab kit -> $RUN_DIR"
say "CPU build: $VARIANT | NUMA nodes: $NODES | cores: $NPHYS (threads $NCPUS)${BUDGET_MIN:+ | stops starting tests after $BUDGET_MIN min}"
if [ "$ON_USB" = 0 ]; then
  say "Diagnostics kit: no models on this USB, so only phases 1-3 run (~15 min); model tests are marked 'absent'."
else
  say "Phases 1-4b are the important data (~1 h; ~30 min more if Flash-Next isn't on this disk yet)."
fi
say "Stop any time: Ctrl-C here, or '$UI_CMD pause' / '$UI_CMD stop' in another terminal."
if [ "$UI" = pretty ]; then
  printf '\n %sperseus lab kit%s  %s%s%s\n' "$C_B" "$C_0" "$C_DIM" "$(basename "$RUN_DIR")" "$C_0"
  nodes="$NODES NUMA node"; [ "$NODES" = 1 ] || nodes="${nodes}s"
  ui_kv cpu "$VARIANT build, $NPHYS cores / $NCPUS threads, $nodes"
  if [ "$ON_USB" = 0 ]; then ui_kv models "none on this USB: phases 1-3 only (~15 min)"
  else ui_kv models "$ON_USB files on this USB; phases 1-4b are the key data (~1 h)"; fi
  if [ -n "$BUDGET_MIN" ]; then
    end=$(( $(date +%s) + DEADLINE - $(now_s) ))  # the enforced deadline, set before setup ran
    ui_kv time "no new tests after $(date -d "@$end" +%H:%M 2>/dev/null || date -r "$end" +%H:%M) (--for $BUDGET_ARG)"
  fi
  [ "$UI_DONE" -gt 0 ] && ui_kv resume "$UI_DONE of $UI_TOTAL tests done earlier"
  ui_kv stop "Ctrl-C, or from a second terminal: $UI_CMD pause | stop"
fi
if [ "$ON_USB" != 0 ] && [ "$(id -u)" != 0 ]; then
  log "!! not under sudo: phase 4b's GPU clock tests and CPU profile will be skipped. Ctrl-C and run: sudo bash run.sh"
fi
start_prefetch
plan
write_summary
touch "$RUN_DIR/FINISHED"
n_absent=$(grep -l '^absent$' "$RUN_DIR"/steps/*/status 2>/dev/null | wc -l | tr -d ' ')
absent_note=""
[ "$n_absent" -gt 0 ] && absent_note=" ($n_absent tests need models this USB doesn't carry)"
say "DONE$absent_note. Results: $RUN_DIR (summary.txt has the headline numbers). Eject the USB and bring it back."
if [ "$UI" = pretty ]; then
  count() { grep -l -x -E "$1" "$RUN_DIR"/steps/*/status 2>/dev/null | wc -l | tr -d ' '; }
  n_ok=$(count ok) n_skip=$(count skipped) n_bad=$(count 'failed|timeout|crashed')
  tally="$n_ok ok, $n_skip skipped"
  [ "$n_bad" -gt 0 ] && tally="$tally, $n_bad failed"
  [ "$n_absent" -gt 0 ] && tally="$tally, $n_absent need models not on this USB"
  if [ "$n_bad" -gt 0 ]; then ui_end "$C_WARN" "$G_BAD Done in $(ui_dur $(( $(now_s) - UI_T0 )))" "$tally"
  else ui_end "$C_OK" "$G_OK Done in $(ui_dur $(( $(now_s) - UI_T0 )))" "$tally"; fi
  ui_kv results "$RUN_DIR/summary.txt"
  printf '\n Eject the USB and bring it back.\n'
fi
