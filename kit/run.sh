#!/usr/bin/env bash
# perseus lab kit: collects hardware facts and inference benchmarks on the lab box and
# logs everything to results/ on the USB. Plain-language instructions: README.txt.
#
#   bash run.sh [--for MINUTES]   run, most important tests first; continues an unfinished run
#   bash run.sh status            progress of the current run (works from a second terminal)
#   bash run.sh pause             finish the current test, then stop (second terminal)
#   bash run.sh stop              stop now; the current test reruns next time (same as Ctrl-C)
#   bash run.sh clean             delete the model copies on the local disk
# Options:
#   --for MINUTES    don't start new tests after this many minutes
#   --retry-failed   also rerun tests that failed, timed out or crashed last time
#   --new            start a fresh results folder instead of continuing
#   --no-copy        run models straight off the USB (slow loads, no local disk use)
# Troubleshooting: PERSEUS_VARIANT=penryn|nehalem|haswell forces the CPU build.
# Tests only: PERSEUS_PAUSE_AFTER=STEP-ID pauses after that step, as 'run.sh pause' would.
set -u

KIT="$(cd "$(dirname "$0")" && pwd)"
. "$KIT/lib/common.sh"

CMD=run BUDGET_MIN="" RETRY_FAILED=0 NEW_RUN=0 NO_COPY=0
while [ $# -gt 0 ]; do
  case "$1" in
    status|pause|stop|clean) CMD="$1" ;;
    --for) BUDGET_MIN="${2:-}"; shift ;;
    --retry-failed) RETRY_FAILED=1 ;;
    --new) NEW_RUN=1 ;;
    --no-copy) NO_COPY=1 ;;
    *) echo "unknown argument: $1 (see the top of run.sh)"; exit 2 ;;
  esac
  shift
done
case "$BUDGET_MIN" in
  '') ;;
  *[!0-9]*|0) echo "--for needs a whole number of minutes, e.g. --for 95"; exit 2 ;;
esac

OS=$(uname -s)
HOST=$(hostname 2>/dev/null | cut -d. -f1)
[ -n "$HOST" ] || HOST=unknown
# Fixed path (not $HOME) so 'sudo bash run.sh' and 'bash run.sh clean' agree on it.
LOCAL="${PERSEUS_LOCAL:-/var/tmp/perseus-kit-local}"
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

# ---------- the plan: most decision-relevant data first ----------
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
  # Flash-Next formats (q6_K, q5_K, iq4_nl, q2_0) + q4_1; q6_K/q5_K/q4_1 run in phase 4's Qwen3.8-27B
  S 2-00f-flashnext-kernel-selftest step_kernel_selftest test-fn-kernels
  S 2-00g-flashnext-kernel-bench step_kernel_bench test-fn-kernels
  S 2-01-tiny-q4_0-stage step_stage "$TINY_Q40"
  S 2-02-tiny-q4_0-sanity step_sanity "$TINY_Q40"
  S 2-03-tiny-q4_0-bench step_bench "$TINY_Q40" all-sockets cores 128 32 3
  S 2-04-tiny-q4km-stage step_stage "$TINY_Q4KM"
  S 2-05-tiny-q4km-bench step_bench "$TINY_Q4KM" all-sockets cores 128 32 3

  phase 3 "compact 27Bs: Bonsai 2 (PQ2_0) vs Saluki (IQ2-mix)" 25
  S 3-01-bonsai-pq2-stage step_stage "$BONSAI"
  S 3-02-bonsai-pq2-sanity step_sanity "$BONSAI"
  S 3-03-bonsai-pq2-all-sockets step_bench "$BONSAI" all-sockets cores 32 16 2
  S 3-04-bonsai-pq2-one-socket step_bench "$BONSAI" one-socket cores 32 16 2
  # Saluki runs on the perseus SSSE3 kernels (kernels/), checked by 2-00 on this CPU.
  S 3-05-saluki-stage step_stage "$SALUKI"
  S 3-06-saluki-sanity step_sanity "$SALUKI"
  S 3-07-saluki-all-sockets step_bench "$SALUKI" all-sockets cores 32 16 2
  S 3-08-saluki-one-socket step_bench "$SALUKI" one-socket cores 32 16 2

  phase 4 "Qwen3.8-27B Q4_0 reference" 20
  S 4-01-qwen27-stage step_stage "$Q27"
  S 4-02-qwen27-sanity step_sanity "$Q27"
  S 4-03-qwen27-all-sockets step_bench "$Q27" all-sockets cores 32 16 2
  S 4-04-qwen27-one-socket step_bench "$Q27" one-socket cores 32 16 2

  phase 5 "MTP speculative decoding" 15
  S 5-01-mtp-head-stage step_stage "$MTP27"
  S 5-02-bonsai-pq2-mtp-off step_mtp "$BONSAI" "" 0
  S 5-03-bonsai-pq2-mtp-3 step_mtp "$BONSAI" "$MTP27" 3
  S 5-04-qwen27-mtp-off step_mtp "$Q27" "" 0
  S 5-05-qwen27-mtp-3 step_mtp "$Q27" "$MTP27" 3
  S 5-06-saluki-mtp-off step_mtp "$SALUKI" "" 0
  S 5-07-saluki-mtp-3 step_mtp "$SALUKI" "$MTP27" 3
  # A/B: 5-07 verifies 4 tokens per step with the perseus multi-column kernels, this without
  S 5-08-saluki-mtp-3-mc-off step_mtp_mc_off "$SALUKI" "$MTP27" 3

  phase 6 "extras" 60
  S 6-01-bonsai-pq2-stock-layout step_bench "$BONSAI" naive cores 32 16 2
  S 6-02-qwen27-stock-layout step_bench "$Q27" naive cores 32 16 2
  S 6-03-bonsai-ptq1-stage step_stage "$BONSAI_T"
  S 6-04-bonsai-ptq1-sanity step_sanity "$BONSAI_T"
  S 6-05-bonsai-ptq1-all-sockets step_bench "$BONSAI_T" all-sockets cores 32 16 2
  S 6-06-bonsai-ptq1-one-socket step_bench "$BONSAI_T" one-socket cores 32 16 2
  S 6-07-bonsai-pq2-hyperthreads step_bench "$BONSAI" all-sockets smt 32 16 2
  S 6-08-qwen27-hyperthreads step_bench "$Q27" all-sockets smt 32 16 2
  S 6-09-bonsai-pq2-mtp-2 step_mtp "$BONSAI" "$MTP27" 2
  S 6-10-qwen27-mtp-2 step_mtp "$Q27" "$MTP27" 2
  S 6-11-bonsai-pq2-cold-distribute step_bench "$BONSAI" distribute-cold cores 32 16 2
  S 6-12-qwen27-cold-distribute step_bench "$Q27" distribute-cold cores 32 16 2
  S 6-13-saluki-stock-layout step_bench "$SALUKI" naive cores 32 16 2
  # A/B: 3-07's prompt numbers use the tiled matmul (enabled on SSE4.1 by kernels/), this without
  S 6-14-saluki-tiled-off step_bench_tiled_off "$SALUKI" all-sockets cores 32 16 2
  # Saluki's IQ kernels are table-lookup heavy: do hyperthreads hide that, and is draft 2 better?
  S 6-15-saluki-hyperthreads step_bench "$SALUKI" all-sockets smt 32 16 2
  S 6-16-saluki-mtp-2 step_mtp "$SALUKI" "$MTP27" 2
  # longer prompts: Saluki on the tiled matmul vs Bonsai (no batched kernel for PQ2_0)
  S 6-17-saluki-prompt-128 step_bench "$SALUKI" all-sockets cores 128 0 1
  S 6-18-bonsai-pq2-prompt-128 step_bench "$BONSAI" all-sockets cores 128 0 1
  # A/B: 3-07 runs Saluki's IQ3_S weights repacked at load (kernels/), this in the file layout
  S 6-19-saluki-repack-off step_bench_repack_off "$SALUKI" all-sockets cores 32 16 2
}
# Background copy order: what the plan needs first.
PREFETCH_ORDER="$TINY_Q40 $TINY_Q4KM $BONSAI $SALUKI $Q27 $MTP27 $BONSAI_T"

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
    else state="stopped; 'bash run.sh' continues it"; fi
    echo "run:   $(basename "$RUN_DIR")"
    echo "state: $state"
    P_TITLE="" PROBLEMS=""
    flush_phase() {
      [ -n "$P_TITLE" ] || return 0
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
        failed|timeout|crashed) P_BAD=$((P_BAD + 1)); PROBLEMS="$PROBLEMS  $1: $st"$'\n' ;;
        '') if [ -f "$d/started" ] && owner_alive; then
              local age=$(( $(date +%s) - $(stat -c %Y "$d/started" 2>/dev/null || stat -f %m "$d/started") ))
              P_NOW="$1 ($(fmt_dur "$age"))"
            fi ;;
      esac
    }
    FIRST_PHYS=0
    plan; flush_phase
    [ -z "$PROBLEMS" ] || printf 'problems (details in results/<run>/steps/<test>/stderr.txt):\n%s' "$PROBLEMS"
    exit 0 ;;
esac

# ---------- run: results folder, lock, signals ----------
mkdir -p "$KIT/results" 2>/dev/null
probe="$KIT/results/.write-test.$$"
if ! ( echo ok > "$probe" ) 2>/dev/null; then echo "Can't write to $KIT/results. Is the USB read-only or full?"; exit 1; fi
rm -f "$probe"
usb_free_kb=$(df -Pk "$KIT/results" | awk 'NR==2 {print $4}')
if [ "${usb_free_kb:-0}" -lt 204800 ]; then echo "USB has under 200 MB free; results won't fit."; exit 1; fi

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
CURRENT_STEP=""

on_interrupt() {
  if [ -n "$CURRENT_STEP" ]; then
    echo "stopped by user" > "$RUN_DIR/steps/$CURRENT_STEP/reason"
    write_status "$RUN_DIR/steps/$CURRENT_STEP" interrupted 130 "$(now_s)"
  fi
  write_summary
  log "Stopped. Run 'bash run.sh' again to continue from here."
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
  [ -n "$DEADLINE" ] && [ "$(now_s)" -ge "$DEADLINE" ] && why="Time limit (--for $BUDGET_MIN) reached"
  [ -n "$why" ] || return 0
  rm -f "$PAUSE_FLAG"
  write_summary
  log "$why. Results so far are saved. Run 'bash run.sh' to continue from here."
  exit 0
}

# ---------- tools: binaries, topology, CPU variant ----------
mkdir -p "$LOCAL" && touch "$LOCAL/.perseus-kit" || { echo "Can't create $LOCAL"; exit 1; }
BIN="$LOCAL/bin"
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

# Records what else was going on, so a slow number can be explained later.
note_env() {
  { echo "loadavg: $(cat /proc/loadavg)"
    echo "background copy: $(cat "$LOCAL/models/.copying" 2>/dev/null || echo none)"
    echo "overrides: $(env | grep -E '^(PERSEUS|GGML)_' | tr '\n' ' ')"
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

# A/B variants of the steps above (the kernels/ patches read these at startup)
step_mtp_mc_off() { export PERSEUS_MC=0; step_mtp "$@"; }
step_bench_tiled_off() { export GGML_CPU_TILED_MM=0; step_bench "$@"; }
step_bench_repack_off() { export PERSEUS_REPACK=0; step_bench "$@"; }

# ---------- go ----------
phase() { log "== Phase $1: $2 (~$3 min on the Mac Pro, rough)"; }
S() {
  gate; run_step "$@"
  if [ "$1" = "${PERSEUS_PAUSE_AFTER:-}" ]; then touch "$PAUSE_FLAG"; fi
}

log "perseus lab kit -> $RUN_DIR"
log "CPU build: $VARIANT | NUMA nodes: $NODES | cores: $NPHYS (threads $NCPUS)${BUDGET_MIN:+ | stops starting tests after $BUDGET_MIN min}"
ON_USB=0
for f in ${M_FILE[@]+"${M_FILE[@]}"}; do [ -f "$KIT/models/$f" ] && ON_USB=$((ON_USB + 1)); done
if [ "$ON_USB" = 0 ]; then
  log "Diagnostics kit: no models on this USB, so only phases 1-2 run (~10 min); model tests are marked 'absent'."
else
  log "Phases 1-5 are the important data (~80 min)."
fi
log "Stop any time: Ctrl-C here, or 'bash run.sh pause' / 'bash run.sh stop' in another terminal."
start_prefetch
plan
write_summary
touch "$RUN_DIR/FINISHED"
n_absent=$(grep -l '^absent$' "$RUN_DIR"/steps/*/status 2>/dev/null | wc -l | tr -d ' ')
absent_note=""
[ "$n_absent" -gt 0 ] && absent_note=" ($n_absent tests need models this USB doesn't carry)"
log "DONE$absent_note. Results: $RUN_DIR (summary.txt has the headline numbers). Eject the USB and bring it back."
