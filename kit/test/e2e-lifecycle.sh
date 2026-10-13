#!/usr/bin/env bash
# Lifecycle e2e for the lab kit in an offline container: skip-vs-fail, Ctrl-C + resume,
# duplicate run refused, kill -9 mid-step -> 'crashed' and not retried, --retry-failed reruns it.
# usage: e2e-lifecycle.sh ASSEMBLED_KIT_DIR
set -uo pipefail
K="$1"; C=perseus-e2e-life; P="/media/lab/MY USB/PERSEUS-KIT"
pass=0; fail=0
ok() { echo "PASS  $1"; pass=$((pass + 1)); }
bad() { echo "FAIL  $1"; fail=$((fail + 1)); }
expect() { [ "$2" = "$3" ] && ok "$1 ($3)" || bad "$1: expected '$2', got '$3'"; }
inside() { docker exec "$C" bash -c "$1"; }
st() { inside "cat \"\$(ls -d '$P'/results/*/ | tail -1)/steps/$1/status\" 2>/dev/null || echo none"; }
wait_for() { # bash-condition seconds
  local i=0; while [ "$i" -lt "$2" ]; do inside "$1" >/dev/null 2>&1 && return 0; sleep 1; i=$((i + 1)); done; return 1
}
# docker exec -d, not '&': background jobs of a non-interactive shell start with SIGINT
# ignored, which would make the Ctrl-C test meaningless. A terminal run has SIGINT live.
start_run() { # args...
  docker exec -d "$C" bash -c "cd '$P' && exec setsid bash run.sh $* >> /tmp/run.out 2>&1"
  sleep 1
}
run_pgid() { inside "pgrep -f '^bash run.sh' | head -1"; }

docker rm -f "$C" >/dev/null 2>&1
INIT=--init; [ "${E2E_NO_INIT:-0}" = 1 ] && INIT=""
docker run -d $INIT --memory=10g --memory-swap=10g -e PERSEUS_VARIANT=nehalem --name "$C" --network none -u "$(id -u):$(id -g)" -v "$K:$P" perseus-kit-build:bookworm sleep infinity >/dev/null
inside "rm -rf '$P'/results/* '$P'/results/.lock"

BONSAI=3-03-bonsai-pq2-all-sockets
start_probe() { docker exec -d "$C" bash -c "exec setsid sleep 31"; sleep 1; }
start_probe
expect "harness: SIGINT not ignored in started process" 0 "$(inside "printf '%d' \$(( 0x\$(grep SigIgn /proc/\$(pgrep -f '^sleep 31' | head -1)/status | awk '{print \$2}') & 2 ))")"
inside "pkill -f '^sleep 31'"
# 1) Ctrl-C mid-step: the step becomes 'interrupted', the lock is released, exit code 130.
start_run --no-copy
wait_for "ls '$P'/results/*/steps/$BONSAI/started" 900 || bad "never reached $BONSAI"
sleep 5
inside "kill -INT -\$(pgrep -f '^bash run.sh' | head -1)"
wait_for "! pgrep -f '^bash run.sh'" 20 || bad "run.sh did not stop within 20 s of SIGINT"
expect "Ctrl-C stops the benchmark too" none "$(inside "pgrep -f '[l]lama-(bench|completion|speculative)' >/dev/null && echo running || echo none")"
expect "Ctrl-C marks the running step" interrupted "$(st $BONSAI)"
expect "Ctrl-C releases the lock" absent "$(inside "[ -d '$P/results/.lock' ] && echo present || echo absent")"
expect "single-node steps are skipped, not failed" skipped "$(st 1-06-membw-node0-local)"
expect "summary written on stop" yes "$(inside "[ -f \"\$(ls -d '$P'/results/*/ | tail -1)/summary.txt\" ] && echo yes || echo no")"

if [ "${E2E_ONLY_PART1:-0}" = 1 ]; then
  inside "p=\$(pgrep -f '^bash run.sh' | head -1); [ -n \"\$p\" ] && kill -KILL -\$p"
  docker rm -f "$C" >/dev/null; echo "passed $pass, failed $fail"; [ "$fail" -eq 0 ]; exit
fi

# 2) Rerun resumes the same folder and repeats the interrupted step; a second run is refused.
start_run --no-copy
wait_for "ls '$P'/results/*/steps/$BONSAI/started && [ ! -f \"\$(ls -d '$P'/results/*/ | tail -1)/steps/$BONSAI/status\" ]" 120 || bad "interrupted step not rerun"
expect "resume reuses one results folder" 1 "$(inside "ls -d '$P'/results/*/ | wc -l")"
expect "earlier ok steps not rerun" 1 "$(inside "grep -c '>> 1-01-env' \"\$(ls -d '$P'/results/*/ | tail -1)/run.log\"")"
dup=$(inside "cd '$P' && bash run.sh --no-copy 2>&1; echo rc=\$?" | tail -2 | tr '\n' ' ')
case "$dup" in *"Another run is active"*rc=1*) ok "duplicate run refused" ;; *) bad "duplicate run not refused: $dup" ;; esac

# 3) kill -9 the whole run mid-step (power cut): the step is 'crashed' next time and skipped.
inside "kill -KILL -\$(pgrep -f '^bash run.sh' | head -1)"
sleep 2
expect "kill -9 of the run leaves no benchmark behind" none "$(inside "pgrep -f '[l]lama-(bench|completion|speculative)' >/dev/null && echo running || echo none")"
start_run --no-copy
wait_for "[ -f \"\$(ls -d '$P'/results/*/ | tail -1)/steps/$BONSAI/status\" ]" 60 || bad "no status after crash"
expect "kill -9 mid-step -> crashed" crashed "$(st $BONSAI)"
expect "stale lock taken over" yes "$(inside "grep -q 'was cut off' /tmp/run.out && echo yes || echo no")"
wait_for "ls '$P'/results/*/steps/5-02-bonsai-pq2-mtp-off/started" 900 || bad "did not move on past the crashed step"
expect "crashed step not retried" crashed "$(st $BONSAI)"
inside "p=\$(pgrep -f '^bash run.sh' | head -1); [ -n \"\$p\" ] && kill -INT -\$p"; wait_for "! pgrep -f '^bash run.sh'" 120

# 4) --retry-failed reruns the crashed step.
start_run --no-copy --retry-failed
wait_for "[ ! -f \"\$(ls -d '$P'/results/*/ | tail -1)/steps/$BONSAI/status\" ] && ls '$P'/results/*/steps/$BONSAI/started" 60 \
  && ok "--retry-failed reruns crashed step" || { bad "--retry-failed did not rerun crashed step"; inside "tail -15 /tmp/run.out; d=\$(ls -d '$P'/results/*/ | tail -1)/steps/$BONSAI; ls -la \"\$d\"; cat \"\$d/status\" \"\$d/reason\" 2>&1"; }
inside "p=\$(pgrep -f '^bash run.sh' | head -1); [ -n \"\$p\" ] && kill -INT -\$p"; wait_for "! pgrep -f '^bash run.sh'" 120

docker rm -f "$C" >/dev/null
echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
