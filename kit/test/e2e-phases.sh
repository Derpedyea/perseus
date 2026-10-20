#!/usr/bin/env bash
# Phased-run e2e in copy mode (background copier on): status, stop and pause from a second
# terminal, --for time limit, orphaned copier cleanup after kill -9, clean refusing while running.
# A core kit's models (1 GB) copy in seconds, so the copier checks allow for one that already ended.
# Container is memory-capped (12 GB, no swap); the kit's local dir is a fresh host folder.
# usage: e2e-phases.sh ASSEMBLED_KIT_DIR
set -uo pipefail
K="$1"; C=perseus-e2e-phases; P="/media/lab/MY USB/PERSEUS-KIT"
LD="$HOME/.cache/perseus-kit/e2e-local"; rm -rf "$LD"; mkdir -p "$LD"
pass=0; fail=0
ok() { echo "PASS  $1"; pass=$((pass + 1)); }
bad() { echo "FAIL  $1"; fail=$((fail + 1)); }
expect() { [ "$2" = "$3" ] && ok "$1 ($3)" || bad "$1: expected '$2', got '$3'"; }
has() { case "$3" in *"$2"*) ok "$1" ;; *) bad "$1: '$2' not in: $(echo "$3" | tail -5 | tr '\n' '|')" ;; esac; }
inside() { docker exec "$C" bash -c "$1"; }
kit() { inside "cd '$P' && bash run.sh $* 2>&1"; }
RD="\"\$(ls -d '$P'/results/*/ | tail -1)\""
st() { inside "cat $RD/steps/$1/status 2>/dev/null || echo none"; }
wait_for() { local i=0; while [ "$i" -lt "$2" ]; do inside "$1" >/dev/null 2>&1 && return 0; sleep 1; i=$((i + 1)); done; return 1; }
start_run() { docker exec -d "$C" bash -c "cd '$P' && exec setsid bash run.sh $* >> /tmp/run.out 2>&1"; sleep 1; }
runner_gone() { wait_for "! pgrep -f '^bash run.sh'" "$1"; }
# Distinct copier sessions (the copier's own $(...) subshells share its command line).
copiers() { inside "ps -eo sid,args | grep '[l]ib/prefetch.sh' | awk '{print \$1}' | sort -u | wc -l"; }

docker rm -f "$C" >/dev/null 2>&1
docker run -d --init --memory=10g --memory-swap=10g -e PERSEUS_VARIANT=nehalem --name "$C" --network none -u "$(id -u):$(id -g)" \
  -v "$K:$P" -v "$LD:/var/tmp/perseus-kit-local" perseus-kit-build:bookworm sleep infinity >/dev/null
inside "rm -rf '$P'/results/* '$P'/results/.lock '$P'/results/.pause"

has "status with no runs" "No runs on this machine yet" "$(kit status)"

# 1) Copy mode: copier runs in the background in plan order; status reports progress.
start_run
wait_for "grep -q 'copy Qwen3.5-0.8B-Q4_0' $RD/prefetch.log" 300 || bad "copier never started"
s=$(kit status)
has "status: running" "running (pid" "$s"
has "status: phase 1 listed" "1. machine facts + memory bandwidth" "$s"
expect "at most one copier running" yes "$(c=$(copiers); [ "$c" -le 1 ] && echo yes || echo "no ($c)")"
has "clean refuses while running" "A run is active" "$(kit clean)"

# 2) stop from a second terminal: runner and copier both gone, current step interrupted.
has "stop acknowledged" "Stopping" "$(kit stop)"
runner_gone 20 || bad "runner still alive 20 s after stop"
sleep 2
expect "stop kills the copier too" 0 "$(copiers)"
has "status after stop" "stopped; 'bash run.sh' continues it" "$(kit status)"

# 3) kill -9 the runner while the copier works: the next start kills the orphan copier. Without
# their .ok markers the models are copied again.
inside "rm -f /var/tmp/perseus-kit-local/models/*.ok"
start_run
wait_for "[ -f /var/tmp/perseus-kit-local/models/.copying ]" 120 || bad "copier did not restart"
inside "kill -KILL -\$(pgrep -f '^bash run.sh' | head -1)"; sleep 2
orphans=$(copiers)
[ "$orphans" -ge 1 ] && ok "copier outlives kill -9 of the runner (orphan: $orphans)" || ok "copier finished before kill -9 (nothing orphaned)"
start_run
sleep 5
expect "restart leaves exactly one copier (or none if all copied)" yes "$(c=$(copiers); [ "$c" -le 1 ] && echo yes || echo "no ($c)")"

# 4) pause during a model test: that test finishes ok, the run stops cleanly before the next.
wait_for "ls $RD/steps/2-03-tiny-q4_0-bench/started" 900 || bad "never reached 2-03"
has "pause acknowledged" "will stop after the current test" "$(kit pause)"
runner_gone 900 || bad "runner did not stop after pause"
expect "paused test completed normally" ok "$(st 2-03-tiny-q4_0-bench)"
expect "next test not started" none "$(inside "[ -f $RD/steps/2-04-tiny-q4km-stage/started ] && echo started || echo none")"
has "pause message" "Paused. Results so far are saved" "$(inside 'tail -3 /tmp/run.out')"
expect "pause flag cleared" absent "$(inside "[ -f '$P/results/.pause' ] && echo present || echo absent")"

# 5) What got copied, in plan order, verified.
inside "cat $RD/prefetch.log" | grep -E '^\[.*\] copy ' | awk '{print $3}' | uniq | tr '\n' ' '; echo
expect "tiny Q4_K_M staged + verified" yes "$(inside "[ -f /var/tmp/perseus-kit-local/models/Qwen3.5-0.8B-Q4_K_M.gguf.ok ] && echo yes || echo no")"
has "bench records the background copy state" "background copy:" "$(inside "cat $RD/steps/2-03-tiny-q4_0-bench/env.txt")"
expect "one results folder (resumed each time)" 1 "$(inside "ls -d '$P'/results/*/ | wc -l")"

# 6) --for 1 on a fresh run, with 20 s minutes: no new test starts after 20 s; whatever was running
# finishes cleanly. (Resumed, the rest of a core kit's plan would end before the limit.)
t0=$(date +%s)
docker exec -d -e PERSEUS_MINUTE_S=20 "$C" bash -c "cd '$P' && exec setsid bash run.sh --new --for 1 >> /tmp/run.out 2>&1"; sleep 1
runner_gone 600 || bad "runner did not stop under --for 1"
el=$(( $(date +%s) - t0 ))
has "time-limit message" "Time limit (--for 1) reached" "$(inside 'tail -3 /tmp/run.out')"
expect "ran for at least the 20 s" yes "$([ "$el" -ge 19 ] && echo yes || echo "no (${el}s)")"
expect "nothing left cut off" 0 "$(inside "cd $RD/steps && for d in */; do [ -f \$d/started ] && [ ! -f \$d/status ] && echo \$d; done | wc -l")"
expect "plan not marked finished" absent "$(inside "[ -f $RD/FINISHED ] && echo present || echo absent")"

docker rm -f "$C" >/dev/null
rm -rf "$LD"
echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
