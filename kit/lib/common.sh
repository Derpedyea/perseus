# Shared helpers for run.sh. Bash 3.2-compatible (macOS ships 3.2): no associative
# arrays, no mapfile, no ${x^^}.

# Exit code a step returns to mean "deliberately not run"; the reason goes in $STEP_DIR/reason.
SKIP=3

now_s() {
  if [ -r /proc/uptime ]; then cut -d' ' -f1 /proc/uptime | cut -d. -f1; else date +%s; fi
}

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# log/record (run.log + screen) live in ui.sh.

# "95" (minutes), "90m", "8h" or "1h30m" -> whole minutes. Fails on anything else, on 0 and on
# more than a week.
parse_minutes() { # text
  local h=0 m=0
  if [[ "$1" =~ ^([0-9]+)m?$ ]]; then m=${BASH_REMATCH[1]}
  elif [[ "$1" =~ ^([0-9]+)h(([0-9]+)m)?$ ]]; then h=${BASH_REMATCH[1]}; m=${BASH_REMATCH[3]:-0}
  else return 1; fi
  [ ${#h} -le 4 ] && [ ${#m} -le 6 ] || return 1  # before the arithmetic can overflow
  m=$(( 10#$h * 60 + 10#$m ))  # 10#: "08" is not octal
  [ "$m" -gt 0 ] && [ "$m" -le 10080 ] || return 1
  echo "$m"
}

json_str() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr '\n\t' '  '; }

fmt_dur() { printf '%dm%02ds' $(($1 / 60)) $(($1 % 60)); }

# Ends the current step as skipped. Steps run in a subshell, so exit leaves only the step.
skip() { # reason
  echo "$1" > "$STEP_DIR/reason"
  exit $SKIP
}

# Ends the current step as absent: it needs a file this USB doesn't carry (a model on a diag or
# smaller kit). Unlike skipped it is not final; every resume re-evaluates it, so a USB
# re-assembled with a bigger tier runs it. A marker, not an exit code, so a benchmark's own
# exit status can't be mistaken for it.
absent() { # reason
  echo "$1" > "$STEP_DIR/reason"
  touch "$STEP_DIR/absent"
  exit $SKIP
}

# Records a command in cmd.txt and runs it with a timeout when one is available.
# Exit 124 means the timeout fired.
run_cmd() { # timeout_seconds cmd args...
  local t="$1"; shift
  printf '%q ' "$@" >> "$STEP_DIR/cmd.txt"; echo >> "$STEP_DIR/cmd.txt"
  if command -v timeout >/dev/null 2>&1; then
    # --foreground keeps the command in our process group; without it timeout makes its
    # own group and a terminal Ctrl-C never reaches the benchmark.
    timeout --foreground --kill-after=30 "$t" "$@"
  else
    "$@"
  fi
}

# True if PID is a live process here. A killed process can linger as a zombie, which
# kill -0 still accepts; that doesn't count.
pid_alive() {
  [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null || return 1
  [ -r "/proc/$1/stat" ] && [ "$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null)" = Z ] && return 1
  return 0
}

# Copies one model from the USB to the local disk and verifies it.
# Returns 0 ok, 1 checksum mismatch (.bad marker), 2 copy failed, 3 not enough disk (.nospace).
copy_model() { # local_models_dir usb_models_dir file size sha256
  local dst="$1/$3" src="$2/$3" size="$4" sha="$5"
  [ -f "$dst.ok" ] && return 0
  mkdir -p "$1" || return 2
  local free_kb; free_kb=$(df -Pk "$1" | awk 'NR==2 {print $4}')
  if [ "${free_kb:-0}" -lt $(( size / 1024 + 5242880 )) ]; then
    touch "$dst.nospace"; echo "not enough local disk for $3"; return 3
  fi
  rm -f "$dst.part" "$dst.bad" "$dst.nospace"
  cp "$src" "$dst.part" || { rm -f "$dst.part"; echo "copy of $3 failed"; return 2; }
  if command -v sha256sum >/dev/null 2>&1; then
    if [ "$(sha256sum "$dst.part" | cut -d' ' -f1)" != "$sha" ]; then
      rm -f "$dst.part"; touch "$dst.bad"; echo "checksum mismatch for $3: the USB copy is damaged"; return 1
    fi
  fi
  mv "$dst.part" "$dst" && touch "$dst.ok" && echo "staged $3"
}

write_status() { # dir status rc started_epoch
  local d="$1" st="$2" rc="$3" t0="$4" reason=""
  [ -f "$d/reason" ] && reason=$(cat "$d/reason")
  local dur=$(( $(now_s) - t0 ))
  echo "$st" > "$d/status"
  printf '{"step":"%s","status":"%s","exit_code":%s,"duration_s":%s,"ended_utc":"%s","reason":"%s"}\n' \
    "$(basename "$d")" "$st" "$rc" "$dur" "$(ts)" "$(json_str "$reason")" > "$d/status.json"
}

# run_step ID FUNC [ARGS...]
# A step that already finished ok/skipped is not rerun. A step that has a 'started'
# marker but no status was killed mid-way (power loss, OOM, hard hang): it is marked
# 'crashed' and not retried automatically, so one bad step can't wedge every rerun.
# failed/timeout/crashed rerun only with --retry-failed; interrupted (Ctrl-C) and absent (input
# not on this USB) always rerun.
run_step() {
  local id="$1"; shift
  local d="$RUN_DIR/steps/$id" prev=""
  [ -f "$d/status" ] && prev=$(cat "$d/status")
  if [ -z "$prev" ] && [ -f "$d/started" ]; then
    write_status "$d" crashed -1 "$(now_s)"
    prev=crashed
    log "!! $id was cut off last run (crash/power loss/OOM?)"
  fi
  case "$prev" in
    ok|skipped) return 0 ;;
    failed|timeout|crashed)
      if [ "$RETRY_FAILED" != 1 ]; then
        say "-- $id: previously $prev, not retrying (use --retry-failed)"
        [ "$UI" = pretty ] && ui_step_line "$prev" "$id" "" "$prev earlier, not retried (--retry-failed retries)"
        return 0
      fi ;;
  esac
  rm -rf "$d"; mkdir -p "$d"
  echo "$(ts)" > "$d/started"
  CURRENT_STEP="$id"
  local t0; t0=$(now_s)
  STEP_T0=$t0
  say ">> $id"
  ui_tick_start "$id" "$d"
  local rc=0
  ( STEP_DIR="$d"; export STEP_DIR; "$@" ) < /dev/null > "$d/stdout.txt" 2> "$d/stderr.txt" || rc=$?
  ui_tick_stop
  local st=ok
  case "$rc" in
    0) st=ok ;;
    "$SKIP") st=skipped; [ -f "$d/absent" ] && st=absent ;;
    124|137) st=timeout ;;
    130) st=interrupted ;;  # pause/stop/time budget while the step was waiting
    *) st=failed ;;
  esac
  write_status "$d" "$st" "$rc" "$t0"
  CURRENT_STEP=""
  local extra="" note="" took=$(( $(now_s) - t0 ))
  [ -f "$d/reason" ] && note=$(head -c 160 "$d/reason") && extra=" ($note)"
  [ "$st" = failed ] && extra=" (exit $rc; see steps/$id/stderr.txt)" && note="exit $rc, see steps/$id/stderr.txt"
  say "   $id: $st in $(fmt_dur "$took")$extra"
  [ "$UI" = pretty ] && ui_step_line "$st" "$id" "$took" "$note"
  return 0
}

# Writes summary.txt: one line per step plus the headline numbers we came for.
write_summary() {
  local out="$RUN_DIR/summary.txt"
  {
    echo "perseus lab kit results: $(basename "$RUN_DIR")"
    echo "written $(ts)"
    echo
    local n_absent; n_absent=$(grep -l '^absent$' "$RUN_DIR"/steps/*/status 2>/dev/null | wc -l)
    [ "$n_absent" -gt 0 ] && echo "$n_absent tests need models this USB doesn't carry (status 'absent'); they run on the next 'bash run.sh' once the USB has them." && echo
    [ -f "${DPM_RECORD:-}" ] && echo "!! The GPU's performance level is still pinned (record: $DPM_RECORD); 'sudo bash run.sh clean' puts it back, or reboot." && echo
    printf '%-46s %-11s %s\n' STEP STATUS NOTE
    for d in "$RUN_DIR"/steps/*/; do
      [ -d "$d" ] || continue
      local st="running/cut-off" note=""
      [ -f "$d/status" ] && st=$(cat "$d/status")
      [ -f "$d/reason" ] && note=$(head -c 90 "$d/reason")
      printf '%-46s %-11s %s\n' "$(basename "$d")" "$st" "$note"
    done
    echo
    echo "== memory bandwidth (GB/s, best rep)"
    grep -h '"mode":"read"' "$RUN_DIR"/steps/*/stdout.txt 2>/dev/null \
      | sed -n 's/.*"label":"\([^"]*\)".*"gbps_best":\([0-9.]*\).*"placement_pages":\({[^}]*}\).*/  \1: \2   pages \3/p'
    echo
    echo "== llama-bench (tokens/s): step | threads | test | avg"
    for f in "$RUN_DIR"/steps/*/stdout.txt; do
      grep -q '"avg_ts"' "$f" 2>/dev/null || continue
      local step; step=$(basename "$(dirname "$f")")
      [ -f "$(dirname "$f")/config.txt" ] && step="$step [$(head -c 20 "$(dirname "$f")/config.txt")]"
      grep '"avg_ts"' "$f" | while read -r line; do
        local th np ng avg ub dp extra=""
        th=$(echo "$line" | sed -n 's/.*"n_threads": *\([0-9]*\).*/\1/p')
        np=$(echo "$line" | sed -n 's/.*"n_prompt": *\([0-9]*\).*/\1/p')
        ng=$(echo "$line" | sed -n 's/.*"n_gen": *\([0-9]*\).*/\1/p')
        avg=$(echo "$line" | sed -n 's/.*"avg_ts": *\([0-9.]*\).*/\1/p')
        # phase 6 varies these; the default ubatch (512) and an empty context go unlabelled
        ub=$(echo "$line" | sed -n 's/.*"n_ubatch": *\([0-9]*\).*/\1/p')
        dp=$(echo "$line" | sed -n 's/.*"n_depth": *\([0-9]*\).*/\1/p')
        [ -n "$ub" ] && [ "$ub" != 512 ] && extra=" ub=$ub"
        [ -n "$dp" ] && [ "$dp" != 0 ] && extra="$extra at depth $dp"
        if [ "${np:-0}" != 0 ]; then printf '  %-44s t=%-3s pp%-5s %s%s\n' "$step" "$th" "$np" "$avg" "$extra"
        else printf '  %-44s t=%-3s tg%-5s %s%s\n' "$step" "$th" "$ng" "$avg" "$extra"; fi
      done
    done
    echo
    echo "== GPU"
    sed -n 's/^picked: /  picked: /p' "$RUN_DIR"/steps/*-gpu-probe/stdout.txt 2>/dev/null
    for f in "$RUN_DIR"/steps/*-gpu-probe/reason; do [ -f "$f" ] && echo "  no GPU: $(head -c 300 "$f")"; done
    grep -h -E '^== |tests passed' "$RUN_DIR"/steps/*-gpu-selftest/stdout.txt 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g; s/^ */  /'
    local ob
    for ob in "$RUN_DIR"/steps/*-gpu-op-bench*/stdout.txt; do
      [ -f "$ob" ] || continue
      echo "  $(basename "$(dirname "$ob")"): Flash-Next decode matmuls on the GPU, slowest first: us per call, GB/s,"
      echo "  weight format, tensor. Over VRAM speed: the tensor stayed in the GPU's cache. A few GB/s: it sat in"
      echo "  system RAM, read over PCIe (CPU-visible VRAM is 256 MB without Resizable BAR; compare the -novis run)"
      # test-backend-ops colours the GB/s figure even into a file: strip ANSI codes first
      sed 's/\x1b\[[0-9;]*m//g' "$ob" \
        | sed -n 's/.*MUL_MAT(name=\([^,]*\),type=[^,]*,ne=\[[0-9]*,1,1,1\],.*sources=\([A-Za-z0-9_]*\)\[.*- *\([0-9.]*\) us\/run - .* - *\([0-9.]*\) GB\/s.*/\3 \4 \2 \1/p' \
        | sort -rn | head -15 | awk '{printf "    %9s us %8s GB/s  %-8s %s\n", $1, $2, $3, $4}'
    done
    echo
    echo "== Flash-Next and Flint (llama-completion, 128 tokens: cpu-gpu = attention etc. on the GPU, experts on the CPU)"
    grep -H -E 'prompt eval time =|[^t] eval time =|common_params_fit_impl: +- ' "$RUN_DIR"/steps/*-flashnext-cpu-*/stderr.txt "$RUN_DIR"/steps/*-flint-cpu-*/stderr.txt 2>/dev/null \
      | sed "s|$RUN_DIR/steps/||; s|/stderr.txt:|: |; s/[0-9.]* I common_params_fit_impl: *- /GPU placement: /"
    echo
    echo "== llama-server (phase 6): load time; per request prompt/generated tok/s, n-gram drafts accepted/drafted; parallel totals"
    local sf
    for sf in "$RUN_DIR"/steps/*-serve-*/stdout.txt; do
      [ -f "$sf" ] || continue
      echo "  $(basename "$(dirname "$sf")")"
      sed 's/, "text_head": .*}$/}/; s/^/    /' "$sf"
    done
    echo
    echo "== Flash-Next profile (phase 4b; tokens/s are in the llama-bench section)"
    sed 's/^/  4b-05 /' "$RUN_DIR"/steps/*-fn-pick/stdout.txt 2>/dev/null
    local gf; gf=$(ls "$RUN_DIR"/steps/*-fn-gpu-op-times/stderr.txt 2>/dev/null | head -1)
    if [ -n "$gf" ]; then
      # one "Total time: N us." per graph the GPU ran (a decode token is several graphs)
      awk '/^Total time: / { n++; us += $3 } END { if (n) printf "  GPU op time, all graphs: %d graphs, %.1f ms (barrier per op: an overstatement)\n", n, us / 1000 }' "$gf"
    fi
    local pf; pf=$(ls "$RUN_DIR"/steps/*-fn-cpu-profile/report-sym.txt 2>/dev/null | head -1)
    if [ -n "$pf" ]; then
      echo "  CPU profile while generating (perf, all threads), top entries: % of samples, library, function"
      grep -E '^ +[0-9.]+%' "$pf" | head -15 | sed 's/^ */    /'
    fi
    echo
    echo "== kernel bench (ns per 256 weights, 1 thread; nc = tokens per weight read)"
    grep -h '^bench ' "$RUN_DIR"/steps/*-kernel-bench/stdout.txt 2>/dev/null | sed 's/^bench /  /'
    echo
    echo "== lookup-table kernels vs multiply (ns per 256 weights, 1 thread)"
    grep -h '^lut' "$RUN_DIR"/steps/*-lut-bench/stdout.txt 2>/dev/null | sed 's/^/  /'
  } > "$out"
}

LIB_END=common  # last line: run.sh checks it to catch a copy cut short
