# Terminal output for run.sh. Bash 3.2-compatible, like common.sh.
#
# run.log always gets plain "[HH:MM:SS] text" lines: the record (the e2e tests grep it). A screen
# that is an interactive terminal gets the readable form instead: phase rules, one glyph per step,
# [n/total] progress, a live timer on the running step, colour unless NO_COLOR is set or TERM=dumb
# (clig.dev). Output into a file or pipe (nohup, tests) gets the same plain lines as run.log.

UI=plain UI_TERM_LIVE=0 UI_LIVE=0 UI_COLS=80 UI_RESIZED=0 UI_TICK="" UI_TICKW=0 UI_N=0 UI_TOTAL=0 UI_IDW=20 UI_PHASE="" UI_STTY=""
C_OK="" C_BAD="" C_WARN="" C_RUN="" C_DIM="" C_B="" C_0=""
G_OK=+ G_BAD=x G_SKIP=- G_ABS=. G_RUN='>' G_HR=- G_FULL='#' G_EMPTY=.
# How to reach this run from a second terminal: same sudo as this one.
UI_CMD="bash run.sh"
[ -n "${SUDO_USER:-}" ] && UI_CMD="sudo bash run.sh"

# The terminal's width into UI_COLS, capped at 80. stty asks the terminal itself (tput would trust
# a stale COLUMNS, and inside $(...) its stdout isn't the terminal).
ui_term_cols() {
  local s; s=$(stty size < /dev/tty 2>/dev/null)
  s=${s##* }
  case "$s" in ''|*[!0-9]*|0) s=80 ;; esac
  [ "$s" -gt 80 ] && s=80
  UI_COLS=$s
}

ui_init() {
  [ -t 1 ] || return 0
  UI=pretty
  ui_term_cols
  # A resize is picked up before the next step's timer (and by a running timer: ui_tick_start).
  trap 'UI_RESIZED=1' WINCH
  # A dumb terminal may not know the escape that clears a line, so it gets no live timer either.
  [ "${TERM:-dumb}" != dumb ] && UI_TERM_LIVE=1 UI_LIVE=1
  # The terminal's modes, put back after each live timer (it reads keys: ui_tick_start).
  UI_STTY=$(stty -g < /dev/tty 2>/dev/null)
  if [ -z "${NO_COLOR:-}" ] && [ "$UI_LIVE" = 1 ]; then
    C_OK=$'\033[32m' C_BAD=$'\033[31m' C_WARN=$'\033[33m' C_RUN=$'\033[36m' C_DIM=$'\033[2m' C_B=$'\033[1m' C_0=$'\033[0m'
  fi
  case "$(locale charmap 2>/dev/null)" in
    UTF-8|utf8) G_OK='✓' G_BAD='✗' G_SKIP='–' G_ABS='·' G_RUN='▸' G_HR='─' G_FULL='█' G_EMPTY='░' ;;
  esac
}

# 3s, 4m05s, 2h07m
ui_dur() {
  if [ "$1" -lt 60 ]; then printf '%ds' "$1"
  elif [ "$1" -lt 3600 ]; then printf '%dm%02ds' $(($1 / 60)) $(($1 % 60))
  else printf '%dh%02dm' $(($1 / 3600)) $(($1 % 3600 / 60)); fi
}

# Cuts TEXT to N characters, marking the cut.
ui_fit() { # n text
  if [ "$1" -lt 8 ] || [ "${#2}" -le "$1" ]; then printf '%s' "$2"; else printf '%s~' "${2:0:$(($1 - 1))}"; fi
}

ui_repeat() { # n char
  local i=0 s=""
  while [ "$i" -lt "$1" ]; do s="$s$2"; i=$((i + 1)); done
  printf '%s' "$s"
}

# In run.log only; the caller draws the screen.
record() {
  [ -n "${RUN_DIR:-}" ] && echo "[$(date +%H:%M:%S)] $*" >> "$RUN_DIR/run.log"
  return 0
}

# Records TEXT; a plain screen shows it too. The caller draws the pretty screen.
say() {
  record "$1"
  [ "$UI" = plain ] && echo "[$(date +%H:%M:%S)] $1"
  return 0
}

# Everything else run.sh says: warnings start with "!!".
log() {
  say "$*"
  [ "$UI" = pretty ] || return 0
  ui_flush_phase
  case "$*" in
    '!! '*) printf '  %s!%s %s\n' "$C_WARN" "$C_0" "${*#!! }" ;;
    *) printf '  %s%s%s\n' "$C_DIM" "$*" "$C_0" ;;
  esac
}

# The phase rule is drawn when its first step does something, so a resumed run doesn't list
# phases that finished earlier.
ui_phase() { # num title minutes
  say "== Phase $1: $2 (~$3 min on the Mac Pro, rough)"
  [ "$UI" = pretty ] || return 0
  local left="$G_HR$G_HR $1  $2 " right=" ~$3 min"
  local fill=$((UI_COLS - ${#left} - ${#right}))
  [ "$fill" -lt 3 ] && fill=3
  UI_PHASE=$(printf '\n%s%s%s %s%s %s%s%s%s' "$C_DIM" "$G_HR$G_HR" "$C_0" "$C_B" "$1  $2" "$C_0" "$C_DIM" "$(ui_repeat "$fill" "$G_HR")$right" "$C_0")
}
ui_flush_phase() {
  [ -n "$UI_PHASE" ] || return 0
  printf '%s\n' "$UI_PHASE"
  UI_PHASE=""
}

ui_counter() { printf '%s[%*d/%d]%s' "$C_DIM" ${#UI_TOTAL} "$UI_N" "$UI_TOTAL" "$C_0"; }

# One finished (or not rerun) step: glyph, id, progress, duration, note.
ui_step_line() { # status id seconds|"" note
  local g c n=$C_DIM note="${4%%$'\n'*}"
  # The banner and the end of the run count these; one line each would bury a diag kit's run.
  [ "$1" = absent ] && return 0
  case "$1" in
    ok) g=$G_OK c=$C_OK ;;
    skipped) g=$G_SKIP c=$C_WARN ;;
    interrupted) g=$G_SKIP c=$C_WARN ;;
    *) g=$G_BAD c=$C_BAD n=$C_BAD ;;
  esac
  ui_flush_phase
  local dur=""; [ -n "$3" ] && dur=$(ui_dur "$3")
  # 2 + glyph + 1 + id + 1 + counter (3 + 2 * digits) + 1 + duration 6 + 2
  local room=$((UI_COLS - 17 - UI_IDW - 2 * ${#UI_TOTAL}))
  printf '  %s%s%s %-*s %s %6s' "$c" "$g" "$C_0" "$UI_IDW" "$2" "$(ui_counter)" "$dur"
  [ -n "$note" ] && printf '  %s%s%s' "$n" "$(ui_fit "$room" "$note")" "$C_0"
  printf '\n'
}

# The running step's line, redrawn every second with its elapsed time until ui_tick_stop. A step
# that ends within a second never gets one (no flicker; an 'absent' step stays silent). The ticker
# draws the pending phase rule itself and leaves $RUN_DIR/.ui-ticked so the runner doesn't draw it
# again. It runs in the background, where SIGINT is ignored: Ctrl-C reaches only the runner, which
# stops it; it also ends by itself within a second of the runner dying (kill -9), zombie or not.
# It also takes single keys (ui_key): its wait is a one-second 'read' from the terminal, which
# returns early on a key. bash puts the terminal back when that read is killed (SIGTERM, untrapped);
# ui_tick_stop puts the saved modes back as well. It shares the box with benchmarks, so the loop is
# builtins only (without a terminal to read: one 'sleep' process a second).
ui_tick_start() { # id step_dir
  if [ "$UI_RESIZED" = 1 ]; then UI_RESIZED=0; ui_term_cols; ui_fit_live; fi
  [ "$UI_LIVE" = 1 ] || return 0
  local s0=$SECONDS mark="$RUN_DIR/.ui-ticked"
  rm -f "$mark"
  local head; head=$(printf '  %s%s%s %-*s %s' "$C_RUN" "$G_RUN" "$C_0" "$UI_IDW" "$1" "$(ui_counter)")
  (
    # Narrowed below the line mid-step: clear it and stop redrawing (stty: one process, per resize).
    local e d resized=0
    trap 'resized=1' WINCH
    # No /proc (macOS): ps, one process a second.
    runner_alive() {
      kill -0 $$ 2>/dev/null || return 1
      local _p _c st=""
      if [ -r /proc/$$/stat ]; then read -r _p _c st _ < /proc/$$/stat
      else st=$(ps -o stat= -p $$ 2>/dev/null); fi
      case "$st" in Z*) return 1 ;; esac
    }
    local key="" keys=0
    { : < /dev/tty; } 2>/dev/null && keys=1
    wait1() { # a second, or less if a key comes
      key=""
      [ "$keys" = 1 ] || { sleep 1; return; }
      read -rs -n 1 -t 1 key < /dev/tty 2>/dev/null
      [ $? -gt 128 ] || [ -n "$key" ] || sleep 1  # not a timeout and no key: don't spin
    }
    wait1
    runner_alive || exit 0
    [ -n "$UI_PHASE" ] && printf '%s\n' "$UI_PHASE"
    : > "$mark"
    while runner_alive; do
      if [ "$resized" = 1 ]; then
        resized=0; ui_term_cols
        [ "$UI_TICKW" -lt "$UI_COLS" ] || { printf '\r\033[K'; exit 0; }
      fi
      [ -n "$key" ] && ui_key "$key" "$2"
      e=$((SECONDS - s0))
      if [ "$e" -lt 60 ]; then d="${e}s"
      elif [ "$e" -lt 3600 ]; then printf -v d '%dm%02ds' $((e / 60)) $((e % 60))
      else printf -v d '%dh%02dm' $((e / 3600)) $((e % 3600 / 60)); fi
      printf '\r%s %6s ' "$head" "$d" || exit 0
      wait1
    done
  ) &
  UI_TICK=$!
}

# A redraw that wraps leaves a trail ("\r" only rewinds the last row): no live timer on a
# terminal narrower than its line. Call after the dry pass has set UI_IDW and UI_TOTAL, and after
# a resize.
ui_fit_live() {
  UI_TICKW=$((16 + UI_IDW + 2 * ${#UI_TOTAL}))
  UI_LIVE=$UI_TERM_LIVE
  [ "$UI_TICKW" -lt "$UI_COLS" ] || UI_LIVE=0
}
ui_tick_stop() {
  [ -n "$UI_TICK" ] || return 0
  kill "$UI_TICK" 2>/dev/null
  wait "$UI_TICK" 2>/dev/null
  UI_TICK=""
  [ -n "$UI_STTY" ] && stty "$UI_STTY" < /dev/tty 2>/dev/null
  if [ -f "$RUN_DIR/.ui-ticked" ]; then
    rm -f "$RUN_DIR/.ui-ticked"
    UI_PHASE=""
    printf '\r\033[K'
  fi
}

# ---------- keys in the run window (read by the live timer) ----------
UI_KEYS="p pause · s status · l output · m machine · ? keys"

# Clears the timer's line and prints above it; the timer redraws its line right after.
ui_key() { # key step_dir
  printf '\r\033[K'
  case "$1" in
    p|P)
      if [ -f "$PAUSE_FLAG" ]; then rm -f "$PAUSE_FLAG"; ui_note "pause cancelled: the run goes on"
      else : > "$PAUSE_FLAG"; ui_note "pausing after this test (p again cancels)"; fi ;;
    s|S) ui_view_status ;;
    l|L) ui_view_output "$2" ;;
    m|M) ui_view_machine ;;
    '?'|h|H) ui_note "keys: $UI_KEYS · Ctrl-C stop" ;;
  esac
}
ui_note() { printf '  %s!%s %s\n' "$C_WARN" "$C_0" "$1"; }
ui_view_rule() { # title
  local fill=$((UI_COLS - ${#1} - 7)); [ "$fill" -lt 3 ] && fill=3
  printf '  %s%s %s %s%s\n' "$C_DIM" "$G_HR$G_HR" "$1" "$(ui_repeat "$fill" "$G_HR")" "$C_0"
}

# s: every phase's progress: 'run.sh status', as a second terminal shows it.
ui_view_status() {
  ui_view_rule "status"
  PERSEUS_STATUS_BARE=1 bash "$KIT/run.sh" status 2>/dev/null
}

# l: the last lines the running test wrote.
ui_view_output() { # step_dir
  local f n=$((UI_COLS - 6))
  ui_view_rule "output of $(basename "$1")"
  for f in stdout stderr; do
    [ -s "$1/$f.txt" ] || continue
    printf '  %s%s.txt%s\n' "$C_DIM" "$f" "$C_0"
    tail -n 5 "$1/$f.txt" | tr -d '\r' | sed $'s/\033\\[[0-9;]*[A-Za-z]//g' | cut -c1-"$n" | sed "s/^/  $C_DIM|$C_0 /"
  done
  [ -s "$1/stdout.txt" ] || [ -s "$1/stderr.txt" ] || printf '  %s(no output yet)%s\n' "$C_DIM" "$C_0"
}

# m: what the box is doing now: CPU load, RAM, the GPU, the background copy. Linux only.
ui_view_machine() {
  ui_view_rule "machine"
  if [ ! -r /proc/loadavg ]; then printf '   (Linux only)\n'; return 0; fi
  local l1 l5 l15 tot avail
  read -r l1 l5 l15 _ < /proc/loadavg
  tot=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo); avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  ui_kv cpu "load $l1 $l5 $l15 (1/5/15 min) on ${NCPUS:-?} threads"
  ui_kv ram "$(awk -v t="$tot" -v a="$avail" 'BEGIN {printf "%.1f of %.1f GB used", (t - a) / 1048576, t / 1048576}')"
  local p dev busy s m vu vt lvl
  if p=$(dpm_path 2>/dev/null); then
    dev=${p%/*}
    read -r busy < "$dev/gpu_busy_percent" 2>/dev/null || busy="?"
    s=$(sed -n 's/.*: *\([0-9]*[MmGg][Hh]z\) *\*.*/\1/p' "$dev/pp_dpm_sclk" 2>/dev/null)
    m=$(sed -n 's/.*: *\([0-9]*[MmGg][Hh]z\) *\*.*/\1/p' "$dev/pp_dpm_mclk" 2>/dev/null)
    vu=$(cat "$dev/mem_info_vram_used" 2>/dev/null); vt=$(cat "$dev/mem_info_vram_total" 2>/dev/null)
    read -r lvl < "$p" 2>/dev/null || lvl="?"
    ui_kv gpu "busy ${busy}% · core ${s:-?} · mem ${m:-?} · clocks $lvl"
    [ -n "$vu" ] && [ -n "$vt" ] && ui_kv vram "$(awk -v u="$vu" -v t="$vt" 'BEGIN {printf "%.1f of %.1f GB used", u / 1e9, t / 1e9}')"
  else
    ui_kv gpu "no single AMD GPU in sysfs"
  fi
  local c; c=$(cat "$LOCAL/models/.copying" 2>/dev/null)
  ui_kv copy "${c:-none running}"
}

# "   label   value" for the banner and the end of a run.
ui_kv() { printf '   %s%-7s%s %s\n' "$C_DIM" "$1" "$C_0" "$2"; }

# The last word of a run: a headline and what to do next.
ui_end() { # colour headline detail
  printf '\n %s%s%s%s  %s\n' "$C_B" "$1" "$2" "$C_0" "$3"
}

LIB_END=ui  # last line: run.sh checks it to catch a copy cut short
