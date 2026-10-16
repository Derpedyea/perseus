#!/usr/bin/env bash
# run.sh's command line and screen (lib/ui.sh), on a diag kit:
#  (a) --for parsing (minutes, 90m, 8h, 1h30m; rejects 0, fractions, more than a week), --help,
#      unknown arguments: usage error on stderr, exit 2
#  (b) in a terminal (a pty from 'script'): phase rules, a glyph and [n/total] per step, the live
#      timer on a step that takes seconds, colour; run.log keeps its plain lines. NO_COLOR drops
#      the colour; a non-UTF-8 locale gets ASCII glyphs; output into a pipe stays plain
#  (c) 'status' in a terminal while a step runs: per-phase bars and the running step
#  (d) 'stop' mid-step: the timer is gone before "Stopped." and nothing of the run is left;
#      kill -9 of the runner alone: its timer ends by itself
# usage: e2e-tty.sh   (E2E_TAG keeps container names apart)
set -uo pipefail
KITSRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BK="$KITSRC/build/build-kit.sh"
W="${PERSEUS_KIT_WORK:-$HOME/.cache/perseus-kit}"
C="perseus-e2e-tty-${E2E_TAG:-0}"
T=$(mktemp -d "$W/tty-test.XXXX")
P="/media/lab/USB/PERSEUS-KIT"
pass=0; fail=0
ok() { echo "PASS  $1"; pass=$((pass + 1)); }
bad() { echo "FAIL  $1"; fail=$((fail + 1)); }
expect() { [ "$2" = "$3" ] && ok "$1 ($3)" || bad "$1: expected '$2', got '$3'"; }
has() { case "$3" in *"$2"*) ok "$1" ;; *) bad "$1: '$2' not in: $(echo "$3" | tail -5 | tr '\n' '|')" ;; esac; }
hasnt() { case "$3" in *"$2"*) bad "$1: '$2' found" ;; *) ok "$1" ;; esac; }
inside() { docker exec "$C" bash -c "$1"; }
# A terminal 80 columns wide; env assignments may precede the args. Prints the screen bytes.
in_tty() { # out_file env... -- args
  local out="$1"; shift
  inside "cd '$P' && TERM=xterm-256color script -qefc 'stty cols 80; $* bash run.sh $TTY_ARGS' $out >/dev/null 2>&1; echo \$?"
}
cleanup() { docker rm -f "$C" >/dev/null 2>&1; rm -rf "$T"; }
trap cleanup EXIT
ESC=$'\033'

# ---------- (a) ----------
. "$KITSRC/lib/common.sh"
res=""
for x in 95 95m 8h 1h30m 08h 0h30m 007m 10080 0 0h 1.5h h 1h30 '' 8H 99999h 10081; do
  res="$res $x=$(parse_minutes "$x" || echo no)"
done
expect "parse_minutes" " 95=95 95m=95 8h=480 1h30m=90 08h=480 0h30m=30 007m=7 10080=10080 0=no 0h=no 1.5h=no h=no 1h30=no =no 8H=no 99999h=no 10081=no" "$res"

"$BK" assemble "$T" diag > "$T/assemble.out" 2>&1 || { cat "$T/assemble.out"; exit 1; }
K="$T/PERSEUS-KIT"
mkdir -p "$T/local"
# Host-side calls: a regression that starts a run must not touch this machine's kit dir.
export PERSEUS_LOCAL="$T/local-host"
help=$(env -u PERSEUS_LOCAL bash "$K/run.sh" --help 2>&1); rc=$?
expect "--help exits 0" 0 "$rc"
has "--help shows usage and the --for forms" "--for TIME      don't start new tests after TIME: 95 (min), 90m, 8h, 1h30m" "$help"
expect "-h is --help" "$help" "$(env -u PERSEUS_LOCAL bash "$K/run.sh" -h 2>&1)"
expect "--help fits 80 columns" "" "$(echo "$help" | awk 'length > 79')"
# A trailing 'status': if parsing regresses, the call reports instead of starting a run.
for args in "--bogus" "--for" "--for 0" "--for 1.5h" "--for=99999h" "--for="; do
  out=$(bash "$K/run.sh" $args status 2>&1 >/dev/null); rc=$?
  expect "'$args' is a usage error" "2 Run \"bash run.sh --help\" for usage." "$rc $(echo "$out" | tail -1)"
done
# accepted: 'status' after the options parses them without starting a run
for args in "--for 95" "--for=8h" "--for 1h30m"; do
  out=$(bash "$K/run.sh" $args status 2>&1); rc=$?
  expect "'$args' accepted" "0 No runs on this machine yet." "$rc $out"
done

# ---------- (b) ----------
docker rm -f "$C" >/dev/null 2>&1
docker run -d --init --memory=8g --memory-swap=8g -e PERSEUS_VARIANT=nehalem -e LANG=C.UTF-8 --name "$C" --network none \
  -u "$(id -u):$(id -g)" -v "$T:/media/lab/USB" -v "$T/local:/var/tmp/perseus-kit-local" perseus-kit-build:bookworm sleep infinity >/dev/null
TTY_ARGS="--for 8h"
expect "tty run exits 0 (paused)" 0 "$(in_tty /tmp/a.out PERSEUS_PAUSE_AFTER=1-03-membw-1-thread)"
scr=$(inside "cat /tmp/a.out")
has "phase rule" "──$ESC[0m $ESC[1m1  machine facts + memory bandwidth $ESC[0m$ESC[2m───" "$scr"
has "ok step: glyph, id, counter" "$ESC[32m✓$ESC[0m 1-01-env " "$scr"
has "counter shows the plan's size" "[ 1/$(grep -c '^  S ' "$K/run.sh")]" "$scr"
has "live timer drew the running step" "$ESC[36m▸$ESC[0m 1-03-membw-1-thread" "$scr"
has "banner shows the --for deadline" "no new tests after" "$scr"
has "pause headline" "Paused." "$scr"
hasnt "no plain step lines on a terminal" ">> 1-01-env" "$scr"
log=$(inside "cat \"\$(ls -d '$P'/results/\$(hostname)-*/ | tail -1)run.log\"")
has "run.log keeps plain lines" ">> 1-01-env" "$log"
has "run.log keeps plain phase lines" "== Phase 1: machine facts" "$log"
hasnt "run.log has no escape codes" "$ESC" "$log"

# Results folders are named to the second: a 1 s gap keeps each --new run's own.
TTY_ARGS="--new"
sleep 1
in_tty /tmp/b.out NO_COLOR=1 PERSEUS_PAUSE_AFTER=1-02-sysinfo >/dev/null
scr=$(inside "cat /tmp/b.out")
has "NO_COLOR keeps the glyphs" "✓ 1-01-env" "$scr"
hasnt "NO_COLOR: no colour codes" "$ESC[3" "$scr"
sleep 1
in_tty /tmp/c.out LANG=C PERSEUS_PAUSE_AFTER=1-02-sysinfo >/dev/null
has "non-UTF-8 locale: ASCII glyphs" "+$ESC[0m 1-01-env" "$(inside "cat /tmp/c.out")"
sleep 1
pipe=$(inside "cd '$P' && PERSEUS_PAUSE_AFTER=1-02-sysinfo bash run.sh --new 2>&1 | cat")
has "into a pipe: plain lines" "] >> 1-01-env" "$pipe"
hasnt "into a pipe: no escape codes" "$ESC" "$pipe"

# ---------- (c) + (d) ----------
runs() { inside "ls -d '$P'/results/\$(hostname)-*/ 2>/dev/null | wc -l"; }
# Waits until a run newer than the N0 results folders is inside 1-03 (started, no status yet).
in_new_1_03() { # n0
  local waited=0 rd
  while [ "$waited" -lt 60 ]; do
    if [ "$(runs)" -gt "$1" ]; then
      rd=$(inside "ls -d '$P'/results/\$(hostname)-*/ | tail -1")
      inside "[ -f '${rd}steps/1-03-membw-1-thread/started' ] && [ ! -f '${rd}steps/1-03-membw-1-thread/status' ]" && return 0
    fi
    sleep 1; waited=$((waited + 1))
  done
  return 1
}
sleep 1
n0=$(runs)
inside "cd '$P' && TERM=xterm-256color script -qefc 'stty cols 80; bash run.sh --new' /tmp/d.out >/dev/null 2>&1 &"
in_new_1_03 "$n0" || bad "the run never reached 1-03"
sleep 2
TTY_ARGS="status"
in_tty /tmp/s.out >/dev/null
scr=$(inside "cat /tmp/s.out")
has "status: running phase with a bar" "$ESC[36m▸$ESC[0m 1. machine facts + memory bandwidth" "$scr"
has "status: the running step" "now$ESC[0m 1-03-membw-1-thread" "$scr"
inside "cd '$P' && bash run.sh stop" >/dev/null
sleep 3
expect "stop leaves nothing of the run" none "$(inside "pgrep -f '^bash run.sh' >/dev/null && echo running || echo none")"
scr=$(inside "cat /tmp/d.out")
has "stop headline" "Stopped." "$scr"
has "the stopped step keeps its line" "1-03-membw-1-thread" "$(echo "$scr" | grep -F 'stopped; reruns next time')"
expect "no timer redraw after Stopped." "" "$(echo "${scr##*Stopped.}" | grep -F '▸')"

# kill -9 the runner alone (not its group) mid-step: its timer must end by itself. The step's own
# subshell lives on until the step ends (~7 s), so wait that out.
sleep 1
n0=$(runs)
inside "cd '$P' && TERM=xterm-256color script -qefc 'stty cols 80; bash run.sh --new' /tmp/e.out >/dev/null 2>&1 &"
in_new_1_03 "$n0" || bad "the run never reached 1-03"
sleep 2
inside "kill -KILL \$(cut -d' ' -f2 '$P/results/.lock/owner')"
sleep 15
expect "kill -9 of the runner: its timer ends too" none "$(inside "pgrep -f '^bash run.sh' >/dev/null && echo running || echo none")"

# kill -9 of the runner under a parent that never reaps it: the runner stays a zombie, which
# 'kill -0' still accepts. The timer must still end (perl's sleep stands in for that parent).
sleep 1
n0=$(runs)
inside "cd '$P' && TERM=xterm-256color script -qefc 'stty cols 80; perl -e \"if (fork() == 0) { exec qw(bash run.sh --new) } sleep 45\"' /tmp/z.out >/dev/null 2>&1 &"
in_new_1_03 "$n0" || bad "the run never reached 1-03"
sleep 2
inside "kill -KILL \$(cut -d' ' -f2 '$P/results/.lock/owner')"
sleep 15
expect "the killed runner is a zombie meanwhile" Z "$(inside "awk '{print \$3}' /proc/\$(cut -d' ' -f2 '$P/results/.lock/owner')/stat")"
expect "zombie runner: its timer ends too" none "$(inside "pgrep -f '^bash run.sh' >/dev/null && echo running || echo none")"
inside "pkill -f '^perl -e'; sleep 1; rm -rf '$P/results/.lock'"

# A terminal narrower than the timer's line: no live timer (a wrapped redraw leaves a trail), the
# finished lines still print.
TTY_ARGS="--new"
sleep 1
inside "cd '$P' && TERM=xterm-256color script -qefc 'stty cols 40; PERSEUS_PAUSE_AFTER=1-03-membw-1-thread bash run.sh --new' /tmp/n.out >/dev/null 2>&1"
scr=$(inside "cat /tmp/n.out")
hasnt "40 columns: no live timer" "▸" "$scr"
has "40 columns: steps still listed" "✓$ESC[0m 1-03-membw-1-thread" "$scr"

# Narrowed to 40 columns mid-step: the timer stops redrawing (each redraw would wrap and leave a
# trail). A marker written to the pty at the resize splits the screen: 1-03 runs ~7 s and the
# resize lands 2-4 s in, so a timer that ignores it draws 3+ frames after the marker.
sleep 1
n0=$(runs)
inside "cd '$P' && TERM=xterm-256color script -qefc 'stty cols 80; PERSEUS_PAUSE_AFTER=1-03-membw-1-thread bash run.sh --new' /tmp/r.out >/dev/null 2>&1 &"
in_new_1_03 "$n0" || bad "the run never reached 1-03"
sleep 1.5
inside "t=\$(readlink /proc/\$(cut -d' ' -f2 '$P/results/.lock/owner')/fd/1); printf '\\nRESIZE-MARK\\n' > \$t; stty -F \$t cols 40"
waited=0
until [ "$(inside "pgrep -f '^bash run.sh' >/dev/null && echo running || echo none")" = none ] || [ "$waited" -ge 60 ]; do sleep 1; waited=$((waited + 1)); done
frames=$(inside "sed -n '/RESIZE-MARK/,\$p' /tmp/r.out | grep -o '▸' | wc -l")
# at most one frame already on its way when the resize landed
[ "$frames" -le 1 ] && ok "resized to 40 columns: timer stops ($frames frames after the resize)" \
  || bad "resized to 40 columns: timer kept redrawing ($frames frames after the resize)"
has "resized: the step's line still printed" "✓$ESC[0m 1-03-membw-1-thread" "$(inside "cat /tmp/r.out")"

# Setup slower than --for (a 90 s cp): the banner's cutoff is the enforced one, ~60 s after the
# start, not 60 s after setup.
inside "mkdir -p /tmp/slowcp && printf '#!/bin/sh\nsleep 90\nexec /bin/cp \"\$@\"\n' > /tmp/slowcp/cp && chmod +x /tmp/slowcp/cp"
sleep 1
launch=$(date +%s)
inside "cd '$P' && TERM=xterm-256color script -qefc 'stty cols 80; PATH=/tmp/slowcp:\$PATH bash run.sh --new --for 1' /tmp/l.out >/dev/null 2>&1"
scr=$(inside "cat /tmp/l.out")
hhmm=$(echo "$scr" | sed -n 's/.*no new tests after \([0-9][0-9]:[0-9][0-9]\).*/\1/p' | head -1)
shown=$(date -u -d "$(date -u -d "@$launch" +%F) $hhmm" +%s 2>/dev/null || echo 0)  # the container's clock is UTC
[ "$shown" -lt $((launch - 43200)) ] && shown=$((shown + 86400))  # past midnight
# enforced cutoff ~launch+61; the minute shown is at most that. Computed after setup it is >= launch+91.
[ "$shown" -gt 0 ] && [ "$shown" -le $((launch + 75)) ] && ok "slow setup: banner shows the enforced cutoff ($hhmm)" \
  || bad "slow setup: banner cutoff $hhmm is past the enforced one (launch $(date -u -d "@$launch" +%H:%M:%S) UTC)"
has "slow setup: the time limit stops it" "Time limit (--for 1) reached." "$scr"

echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
