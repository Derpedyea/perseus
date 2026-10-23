#!/usr/bin/env bash
# Phase 4b's wiring, with stub binaries and a fake sysfs GPU: no docker, models, GPU or benchmark,
# a few seconds per case, so it runs on a dev box that is busy with other work.
#  (a) a whole run: 4b-01..13 end ok; 4b-05 picks the config the stub makes fastest and the 'best'
#      steps run under it; the GPU's level is 'high' only inside the steps that pin it, 'auto' after,
#      no record left; each GPU bench sampled the GPU; summary has the 4b lines; nothing left running
#  (b) Ctrl-C mid 4b-03 (level pinned): interrupted, level back, record gone, nothing left running
#  (c) kill -9 of the whole run mid 4b-03: level stays pinned and the record stays; the next start
#      restores it before any step; so does a start that refuses to run (results dir read-only)
#  (d) restore fails (switch read-only): the record stays; 'clean' refuses and keeps the local dir;
#      writable again: 'clean' restores, then removes it
#  (e) the switch is gone (GPU renumbered): the next start drops the record and says so
#  (f) no switch in sysfs: 4b-03/04 skipped; 4b-05 picks among 4b-01/02
#  (g) every Flash-Next bench fails: 4b-05 failed, the 'best' steps skipped, not run as 'base'
#  (l) putting the clocks back fails mid-run: later GPU steps are deferred ('interrupted', not run at
#      the wrong level), CPU-only 4b-12 still runs, summary.txt says the GPU is still pinned
#  (m) perf can't attach (no sudo): 4b-09 skipped before Flash-Next is even launched
#  (n) perf record fails after writing data: 4b-09 failed, not ok
#  (k) Ctrl-C while perf is attached to 4b-09's benchmark: interrupted, nothing left running
#  (h) lib/gpu.sh alone: pinning while an earlier restore is still owed keeps the true original
#  (i) Flash-Next not on the USB (and no clock switch): its 4b steps end 'absent' (4b-05, the pinning
#      and the 'best' steps too, not failed/skipped), e2e-diag's model_ids classifies exactly those;
#      once it is back they run
#  (j) no GPU picked: 4b-01..04 skipped, 4b-05 skipped (not failed), CPU-only 4b-12 still runs
#  (p) keys in the run window (a pty): ? l s m each print their view while 4b-03 runs; p sets the
#      pause flag, p again clears it
#  (s) start-up: 'sh run.sh' re-runs itself in bash; a PATH without /usr/bin etc. still works, with
#      sh too; an unreadable or cut-short lib/common.sh stops it with one message naming the file,
#      before any test
#  (o) not root, at a terminal: run.sh asks first. Enter (no) starts nothing and prints the sudo
#      command with the same arguments; y goes on. (Skipped when the tests run as root.)
# usage: stub-4b.sh            CASES="b c" stub-4b.sh runs only those cases
set -uo pipefail
KITSRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T=$(mktemp -d /tmp/perseus-stub4b.XXXX)
K="$T/usb/PERSEUS-KIT"
pass=0; fail=0
ok() { echo "PASS  $1"; pass=$((pass + 1)); }
bad() { echo "FAIL  $1"; fail=$((fail + 1)); }
expect() { [ "$2" = "$3" ] && ok "$1 ($3)" || bad "$1: expected '$2', got '$3'"; }
has() { case "$3" in *"$2"*) ok "$1" ;; *) bad "$1: '$2' not in: $(echo "$3" | tail -5 | tr '\n' '|')" ;; esac; }
cleanup() { pkill -KILL -f -- "$T/" 2>/dev/null; [ -n "${KEEP_T:-}" ] || rm -rf "$T"; }
trap cleanup EXIT  # KEEP_T=1 keeps the temp dir
set -m  # background runs get their own process group with SIGINT live, as in a terminal

# ---------- the fake kit ----------
mkdir -p "$K/models" "$K/gpu-ops" "$T/path" "$T/drm/card0/device"
cp "$KITSRC/run.sh" "$KITSRC/README.txt" "$K/"; cp -r "$KITSRC/lib" "$KITSRC/src" "$K/"
TINY=Qwen3.5-0.8B-Q4_0.gguf TINYK=Qwen3.5-0.8B-Q4_K_M.gguf
FN1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf FN2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
FL=flint-UD-Q4_K_XL.gguf
{ printf '#tier\trunner\tfile\tsize\tsha256\turl\n'
  for f in $TINY $TINYK $FN1 $FN2 $FL; do
    echo "stub model $f" > "$K/models/$f"
    printf 'core\tstock\t%s\t%s\t%s\tx\n' "$f" "$(stat -c %s "$K/models/$f")" "$(sha256sum "$K/models/$f" | cut -d' ' -f1)"
  done; } > "$K/models.tsv"
for f in ${TINY%.gguf} ${FN1%.gguf}; do echo op > "$K/gpu-ops/$f.txt"; : > "$K/gpu-ops/$f.txt.builtin"; done

DPM="$T/drm/card0/device/power_dpm_force_performance_level"
fake_gpu() {
  rm -rf "$T/drm"; mkdir -p "$T/drm/card0/device"
  local d="$T/drm/card0/device"
  echo 0x1002 > "$d/vendor"; echo auto > "$d/power_dpm_force_performance_level"; echo 42 > "$d/gpu_busy_percent"
  printf '0: 300Mhz\n1: 1366Mhz *\n' > "$d/pp_dpm_sclk"; printf '0: 300Mhz *\n1: 1750Mhz\n' > "$d/pp_dpm_mclk"
}

# One stub for every binary: what it does depends on its name and arguments.
cat > "$T/stub" <<EOF
#!/usr/bin/env bash
DPM="$DPM"
EOF
cat >> "$T/stub" <<'EOF'
me=$(basename "$0")
arg() { local want="$1"; shift; while [ $# -gt 0 ]; do [ "$1" = "$want" ] && { echo "$2"; return; }; shift; done; }
case "$me" in
  membw)
    case "$1" in
      topo) [ "${2:-}" = --shell ] && printf 'NODES=1\nNCPUS=4\nNPHYS=2\nNODE0_NCPUS=4\nNODE0_NPHYS=2\nALL_PHYS="0,1"\n' || echo "1 node" ;;
      read) echo '{"mode":"read","label":"stub","gbps_best":9.9,"placement_pages":{"0":1}}' ;;
      exec) while [ "$1" != -- ]; do shift; done; shift; exec "$@" ;;
    esac ;;
  vulkaninfo) echo "stub vulkaninfo" ;;
  perf)
    case "$1" in
      version) [ -n "${STUB_PERF_BROKEN:-}" ] && { echo "WARNING: perf not found for kernel x"; exit 2; }; echo "perf version stub" ;;
      record) o=$(arg -o "$@"); pid=$(arg -p "$@"); d=$(dirname "$o")
        [ -n "${STUB_PERF_NOATTACH:-}" ] && { echo "Error: No permission to enable task-clock event." >&2; exit 255; }
        case "$o" in *perf-probe.data) echo probe > "$o"; exit 0 ;; esac  # '-- sleep 1' ends the probe
        grep -q 'generate: n_ctx' "$d/stderr.txt" && echo "stub perf: attached after generate" || echo "stub perf: attached before generate"
        touch "$d/.stub-attached"
        while kill -0 "$pid" 2>/dev/null; do sleep 0.1; done
        echo data > "$o"; [ -n "${STUB_PERF_FAIL_LATE:-}" ] && exit 2; exit 0 ;;
      report) echo "    41.20%  llama-completion  [.] ggml_vec_dot_iq3_s_q8_K"; echo "    12.00%  libvulkan_radeon.so  [.] stub" ;;
    esac ;;
  llama-bench)
    if [ "${1:-}" = --list-devices ]; then
      echo "ggml_vulkan: 0 = AMD Radeon RX 580 Series (RADV POLARIS10) (radv) | uma: 0 | fp16: 0 | bf16: 0"
      echo "Available devices:"; echo "  Vulkan0: AMD Radeon RX 580 Series (RADV POLARIS10) (8192 MiB, 7835 MiB free)"; exit 0
    fi
    lvl=$(cat "$DPM" 2>/dev/null || echo none)
    echo "stub saw: novis=${GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM:-0} level=$lvl args=$*" >&2
    [ "$lvl" = high ] && [ -n "${STUB_LOCK_SWITCH:-}" ] && chmod 444 "$DPM"  # the switch stops taking writes
    case "$*" in *Flash-Next*) [ -n "${STUB_FAIL_FN:-}" ] && exit 1
      [ "$lvl" = high ] && [ -n "${STUB_HANG_PINNED:-}" ] && sleep 60 ;; esac
    # a GPU bench: let the sampler take a sample first (a condition, not a sleep; bounded at 5 s)
    case "$*" in *"-dev Vulkan0"*) i=0; while [ "$(wc -l < "$STEP_DIR/gpu-samples.txt" 2>/dev/null || echo 0)" -lt 2 ] && [ $i -lt 50 ]; do sleep 0.1; i=$((i + 1)); done ;; esac
    ts=3; [ -n "${GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM:-}" ] && ts=$((ts + 1)); [ "$lvl" = high ] && ts=$((ts + 2))
    p=$(arg -p "$@"); n=$(arg -n "$@"); t=$(arg -t "$@")
    for th in ${t//,/ }; do
      [ "${p:-0}" != 0 ] && echo "{\"n_threads\": $th, \"n_prompt\": $p, \"n_gen\": 0, \"avg_ts\": $((ts * 10))}"
      [ "${n:-0}" != 0 ] && echo "{\"n_threads\": $th, \"n_prompt\": 0, \"n_gen\": $n, \"avg_ts\": $ts}"
    done; exit 0 ;;
  llama-completion)
    echo "generate: n_ctx = 4096, n_batch = 2048" >&2
    # the CPU profile attaches perf now: wait for it (a condition, bounded at 5 s)
    case "$STEP_DIR" in *fn-cpu-profile*) i=0; while [ ! -f "$STEP_DIR/.stub-attached" ] && [ $i -lt 50 ]; do sleep 0.1; i=$((i + 1)); done
      [ -n "${STUB_HANG_PROFILE:-}" ] && sleep 60 ;; esac
    echo "stub completion"
    echo "common_perf_print:        eval time =  100.00 ms /    31 runs" >&2
    [ -n "${GGML_VK_PERF_LOGGER:-}" ] && printf 'Vulkan Timings:\nMUL_MAT: 2 x 10 us = 20 us\nTotal time: 1000 us.\nTotal time: 1500 us.\n' >&2
    exit 0 ;;
  test-backend-ops)
    echo "  MUL_MAT(name=z-0,type=f32,ne=[6144,1,1,1],op_params=[],sources=iq3_s[2560,6144,1,1],f32[2560,1,1,1]):   100 runs -    90.83 us/run -   6634 kB/run -   69.65 GB/s"
    echo "  5/5 tests passed"; echo "  Backend Vulkan0: OK"; exit 0 ;;
  *) echo "stub $me ok"; exit 0 ;;
esac
EOF
chmod +x "$T/stub"
for p in tools/membw stock-vk/nehalem/llama-bench stock-vk/nehalem/llama-completion stock-vk/nehalem/test-backend-ops; do
  mkdir -p "$K/bin/$(dirname "$p")"; cp "$T/stub" "$K/bin/$p"
done
for f in llama-bench llama-completion llama-speculative-simple test-ssse3-kernels test-fn-kernels test-tiled-mulmat test-repack-mulmat test-lut; do
  mkdir -p "$K/bin/stock/nehalem"; cp "$T/stub" "$K/bin/stock/nehalem/$f"
done
echo 2.17 > "$K/bin/stock-vk/nehalem/glibc-min"; mkdir -p "$K/bin/stock-vk/nehalem/lib"
cp "$T/stub" "$T/path/perf"; cp "$T/stub" "$T/path/vulkaninfo"

export PERSEUS_VARIANT=nehalem PERSEUS_LOCAL="$T/local" PERSEUS_SYSFS_DRM="$T/drm" PATH="$T/path:$PATH" NO_COLOR=1
L="$T/local" REC="$T/local/gpu-dpm.restore"
rundir() { ls -d "$K"/results/*/ 2>/dev/null | tail -1; }
st() { cat "$(rundir)steps/$1/status" 2>/dev/null || echo none; }
wait_for() { local i=0; while ! eval "$1"; do sleep 0.2; i=$((i + 1)); [ "$i" -lt $(($2 * 5)) ] || return 1; done; }
left() { pgrep -f -- "$T/" >/dev/null && echo "$(pgrep -af -- "$T/" | head -3 | tr '\n' '|')" || echo none; }
want() { [ -z "${CASES:-}" ] || case " $CASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
run_fg() { (cd "$K" && bash "$K/run.sh" "$@") > "$T/run.out" 2>&1; }

# Phases 1-4 once (stubs), paused before 4b; each case resumes a copy of that state.
fake_gpu
(export PERSEUS_PAUSE_AFTER=4-04-flashnext-cpu-only; run_fg)
expect "phases 1-4 ran on stubs (4-04)" ok "$(st 4-04-flashnext-cpu-only)"
cp -a "$K/results" "$T/snap-results"; cp -a "$L" "$T/snap-local"
fresh() { rm -rf "$K/results" "$L"; cp -a "$T/snap-results" "$K/results"; cp -a "$T/snap-local" "$L"; fake_gpu; }

# ---------- (a) ----------
if want a; then
fresh
run_fg
for s in 4b-01-fn-base 4b-02-fn-novis 4b-03-fn-dpm-high 4b-04-fn-novis-dpm-high 4b-04b-fn-base-again \
         4b-04c-fn-novis-again 4b-05-fn-pick 4b-06-fn-threads \
         4b-07-fn-no-op-offload 4b-08-fn-gpu-op-times 4b-09-fn-cpu-profile 4b-10-tiny-gpu-novis \
         4b-11-gpu-op-bench-novis 4b-12-fn-cpu-only 4b-13-fn-load-dio \
         5-01-flint-cpu-gpu 5-02-flint-cpu-only 5-03-flint-bench-gpu 5-04-flint-bench-cpu; do
  expect "(a) $s" ok "$(st $s)"
done
R=$(rundir)
expect "(a) 4b-05 picks the fastest" "FN_BEST=novis-dpm-high" "$(cat "$R/fn-best.env")"
expect "(a) a 'best' step runs under it" novis-dpm-high "$(cat "$R/steps/4b-06-fn-threads/config.txt")"
has "(a) 4b-06 saw the pinned level and novis" "novis=1 level=high" "$(cat "$R/steps/4b-06-fn-threads/stderr.txt")"
has "(a) 4b-06 ran each thread count" '"n_threads": 4, "n_prompt": 0' "$(cat "$R/steps/4b-06-fn-threads/stdout.txt")"
has "(a) 4b-01 ran at the pre-run level" "novis=0 level=auto" "$(cat "$R/steps/4b-01-fn-base/stderr.txt")"
has "(a) base turns the novis default off" "novis=0 level=auto" "$(cat "$R/steps/4b-04b-fn-base-again/stderr.txt")"
has "(a) Flint's GPU bench runs novis, the fixed split" "novis=1 level=auto args=-m $L/models/$FL -dev Vulkan0 -ngl 99 -ot exps=CPU" "$(cat "$R/steps/5-03-flint-bench-gpu/stderr.txt")"
has "(a) 4b-10 runs unpinned after the pinned steps" "novis=1 level=auto" "$(cat "$R/steps/4b-10-tiny-gpu-novis/stderr.txt")"
has "(a) 4b-03 pinned" "level=high" "$(cat "$R/steps/4b-03-fn-dpm-high/stderr.txt")"
has "(a) env.txt records the level" "gpu performance level: high" "$(cat "$R/steps/4b-03-fn-dpm-high/env.txt")"
has "(a) 4b-12 (CPU) runs without Vulkan" "GGML_DISABLE_VULKAN=1" "$(cat "$R/steps/4b-12-fn-cpu-only/env.txt")"
expect "(a) level back to auto" auto "$(cat "$DPM")"
expect "(a) no record left" none "$([ -f "$REC" ] && echo left || echo none)"
expect "(a) GPU sampled during 4b-01" "42 1366Mhz 300Mhz" "$(sed -n '2p' "$R/steps/4b-01-fn-base/gpu-samples.txt" | cut -d' ' -f2-)"
has "(a) dio step passes -lm dio" "-lm dio" "$(cat "$R/steps/4b-13-fn-load-dio/cmd.txt")"
has "(a) perf attached to the benchmark, no build-id cache" "perf record -F 999 -N -p" "$(cat "$R/steps/4b-09-fn-cpu-profile/perf-record.txt")"
has "(a) after its load and warmup" "attached after generate" "$(cat "$R/steps/4b-09-fn-cpu-profile/perf-record.txt")"
sum=$(cat "$R/summary.txt")
has "(a) summary: pick" "4b-05 picked: novis-dpm-high" "$sum"
has "(a) summary: config label" "4b-06-fn-threads [novis-dpm-high]" "$sum"
has "(a) summary: GPU op time" "GPU op time, all graphs: 2 graphs, 2.5 ms" "$sum"
has "(a) summary: perf top" "ggml_vec_dot_iq3_s_q8_K" "$sum"
has "(a) summary: both op benches" "4b-11-gpu-op-bench-novis: Flash-Next decode matmuls" "$sum"
expect "(a) nothing left running" none "$(left)"
out=$(bash "$K/run.sh" status 2>&1); rc=$?
expect "(a) status exits 0" 0 "$rc"
expect "(a) status runs clean (no shell errors)" "" "$(echo "$out" | grep -E 'command not found|unbound variable')"
has "(a) status lists phase 4b" "4b. Flash-Next: where the time goes" "$out"

fi

# ---------- (b) ----------
if want b; then
fresh
(cd "$K" && STUB_HANG_PINNED=1 exec bash "$K/run.sh" > "$T/run.out" 2>&1) &
pid=$!
wait_for "[ \"\$(cat $DPM)\" = high ] && [ -f \"\$(rundir)steps/4b-03-fn-dpm-high/started\" ]" 30 || bad "(b) never reached a pinned 4b-03"
expect "(b) harness: SIGINT live in the run" 0 "$(( 0x$(awk '/^SigIgn/ {print $2}' /proc/$pid/status) & 2 ))"
kill -INT -- "-$pid"
wait_for "! kill -0 $pid 2>/dev/null" 20 || bad "(b) run did not stop within 20 s of SIGINT"
wait "$pid" 2>/dev/null; rc=$?
expect "(b) exit 130" 130 "$rc"
expect "(b) step interrupted" interrupted "$(st 4b-03-fn-dpm-high)"
expect "(b) level back to auto" auto "$(cat "$DPM")"
expect "(b) no record left" none "$([ -f "$REC" ] && echo left || echo none)"
expect "(b) nothing left running" none "$(left)"

fi

# ---------- (c) ----------
if want c; then
fresh
(cd "$K" && STUB_HANG_PINNED=1 exec bash "$K/run.sh" > "$T/run.out" 2>&1) &
pid=$!
wait_for "[ \"\$(cat $DPM)\" = high ] && [ -f \"\$(rundir)steps/4b-03-fn-dpm-high/started\" ]" 30 || bad "(c) never reached a pinned 4b-03"
kill -KILL -- "-$pid"; wait "$pid" 2>/dev/null
wait_for "[ \"\$(left)\" = none ]" 10
expect "(c) after kill -9 the level stays pinned" high "$(cat "$DPM")"
expect "(c) and the record stays" auto "$(cut -f2 "$REC" 2>/dev/null)"
PERSEUS_PAUSE_AFTER=1-01-env bash -c "cd '$K' && bash '$K/run.sh' --new" > "$T/run.out" 2>&1
expect "(c) next start restores it" auto "$(cat "$DPM")"
log=$(cat "$(rundir)run.log")
expect "(c) before its first step" "restored first" "$(echo "$log" | grep -E "is 'auto' again|>> 1-01-env" | head -1 | grep -q "again" && echo "restored first" || echo "step first")"
expect "(c) record gone" none "$([ -f "$REC" ] && echo left || echo none)"
printf '%s\tauto\n' "$DPM" > "$REC"; echo high > "$DPM"; chmod 555 "$K/results"  # cut off again; USB now read-only
out=$(bash "$K/run.sh" 2>&1); rc=$?
chmod 755 "$K/results"
expect "(c) a start that refuses to run exits 1" 1 "$rc"
has "(c) saying why" "Can't write to" "$out"
expect "(c) and still puts the clocks back" auto "$(cat "$DPM")"
expect "(c) record gone after it" none "$([ -f "$REC" ] && echo left || echo none)"

fi

# ---------- (d) ----------
if want d; then
fresh
printf '%s\tauto\n' "$DPM" > "$REC"; echo high > "$DPM"; chmod 444 "$DPM"
out=$(cd "$K" && bash "$K/run.sh" clean 2>&1); rc=$?
expect "(d) clean refuses while the level is owed" 1 "$rc"
has "(d) and says why" "couldn't put the GPU's performance level back to 'auto'" "$out"
expect "(d) local dir kept" yes "$([ -d "$L" ] && echo yes || echo no)"
chmod 644 "$DPM"
out=$(cd "$K" && bash "$K/run.sh" clean 2>&1); rc=$?
expect "(d) clean once writable" 0 "$rc"
expect "(d) level restored" auto "$(cat "$DPM")"
expect "(d) local dir removed" no "$([ -d "$L" ] && echo yes || echo no)"

fi

# ---------- (e) ----------
if want e; then
fresh
printf '%s\tauto\n' "$T/drm/card7/device/power_dpm_force_performance_level" > "$REC"
(export PERSEUS_PAUSE_AFTER=4b-01-fn-base; run_fg)
has "(e) a vanished switch is reported" "card7/device/power_dpm_force_performance_level is gone" "$(cat "$(rundir)run.log")"
expect "(e) its record dropped" none "$([ -f "$REC" ] && echo left || echo none)"

fi

# ---------- (f) ----------
if want f; then
fresh
rm -rf "$T/drm/card0"
run_fg
expect "(f) 4b-03 skipped without a switch" skipped "$(st 4b-03-fn-dpm-high)"
has "(f) with the reason" "no single AMD GPU performance-level switch" "$(cat "$(rundir)steps/4b-03-fn-dpm-high/reason")"
expect "(f) 4b-05 picks among the rest" "FN_BEST=novis" "$(cat "$(rundir)fn-best.env")"
expect "(f) a 'best' step runs" ok "$(st 4b-06-fn-threads)"

fi

# ---------- (g) ----------
if want g; then
fresh
(export STUB_FAIL_FN=1; run_fg)
expect "(g) 4b-01 failed" failed "$(st 4b-01-fn-base)"
expect "(g) 4b-05 failed: nothing to pick" failed "$(st 4b-05-fn-pick)"
expect "(g) a 'best' step is skipped" skipped "$(st 4b-06-fn-threads)"
has "(g) because nothing was picked" "4b-05 picked no config" "$(cat "$(rundir)steps/4b-06-fn-threads/reason")"
expect "(g) no fn-best.env" none "$([ -f "$(rundir)fn-best.env" ] && echo left || echo none)"
expect "(g) level back to auto" auto "$(cat "$DPM")"

fi

# ---------- (i) ----------
if want i; then
fresh
mkdir -p "$T/held"; mv "$K/models/$FN1" "$K/models/$FN2" "$T/held/"
rm -rf "$T/drm/card0"  # a box without the switch must not turn the model steps into final skips
run_fg
for s in 4b-01-fn-base 4b-03-fn-dpm-high 4b-04-fn-novis-dpm-high 4b-05-fn-pick 4b-06-fn-threads 4b-08-fn-gpu-op-times 4b-12-fn-cpu-only 4b-13-fn-load-dio; do
  expect "(i) $s absent" absent "$(st $s)"
done
expect "(i) the tiny-model step still runs" ok "$(st 4b-10-tiny-gpu-novis)"
expect "(i) no 4b step failed" "" "$(grep -l -x -E 'failed|skipped' "$(rundir)"steps/4b-*/status 2>/dev/null | grep -v 4b-11 | tr '\n' ' ')"
eval "$(grep '^model_ids()' "$KITSRC/test/e2e-diag.sh")"
expect "(i) e2e-diag's model_ids: every 4b step but the op bench" "4b-11-gpu-op-bench-novis" \
  "$(comm -23 <(grep -o '^  S 4b-[^ ]*' "$K/run.sh" | cut -c5- | sort) <(model_ids "$K" | sort) | tr '\n' ' ' | sed 's/ $//')"
mv "$T/held/$FN1" "$T/held/$FN2" "$K/models/"
fake_gpu
rm -f "$(rundir)FINISHED"  # as if paused: a resume reruns the absent steps once the USB has the model
run_fg
expect "(i) back on the USB: 4b-05 runs" ok "$(st 4b-05-fn-pick)"
expect "(i) and so does a 'best' step" ok "$(st 4b-06-fn-threads)"
expect "(i) and a pinning step" ok "$(st 4b-03-fn-dpm-high)"
fi

# ---------- (j) ----------
if want j; then
fresh
rm -f "$(rundir)gpu.env"
run_fg
expect "(j) 4b-01 skipped without a GPU" skipped "$(st 4b-01-fn-base)"
expect "(j) 4b-05 skipped, not failed" skipped "$(st 4b-05-fn-pick)"
has "(j) with the reason" "none of the benches to pick from ran" "$(cat "$(rundir)steps/4b-05-fn-pick/reason")"
expect "(j) a 'best' step skipped" skipped "$(st 4b-06-fn-threads)"
expect "(j) CPU-only still runs" ok "$(st 4b-12-fn-cpu-only)"
fi

# ---------- (l) ----------
if want l; then
fresh
(export STUB_LOCK_SWITCH=1; run_fg)
expect "(l) 4b-03 ran pinned" ok "$(st 4b-03-fn-dpm-high)"
expect "(l) the next GPU step deferred" interrupted "$(st 4b-04-fn-novis-dpm-high)"
has "(l) with the reason" "clocks are still pinned" "$(cat "$(rundir)steps/4b-04-fn-novis-dpm-high/reason")"
expect "(l) an unpinned GPU step deferred too" interrupted "$(st 4b-10-tiny-gpu-novis)"
expect "(l) and it never ran at the wrong level" "" "$(grep 'stub saw' "$(rundir)steps/4b-10-tiny-gpu-novis/stderr.txt" 2>/dev/null)"
expect "(l) CPU-only still runs" ok "$(st 4b-12-fn-cpu-only)"
has "(l) summary says it's still pinned" "still pinned" "$(cat "$(rundir)summary.txt")"
expect "(l) the record is kept for a retry" auto "$(cut -f2 "$REC" 2>/dev/null)"
chmod 644 "$DPM"
fi

# ---------- (m) ----------
if want m; then
fresh
(export STUB_PERF_NOATTACH=1; run_fg)
expect "(m) 4b-09 skipped" skipped "$(st 4b-09-fn-cpu-profile)"
has "(m) with perf's reason" "perf can't attach to a process here" "$(cat "$(rundir)steps/4b-09-fn-cpu-profile/reason")"
expect "(m) before Flash-Next was launched" "" "$(grep llama-completion "$(rundir)steps/4b-09-fn-cpu-profile/cmd.txt" 2>/dev/null)"
fi

# ---------- (n) ----------
if want n; then
fresh
(export STUB_PERF_FAIL_LATE=1; run_fg)
expect "(n) 4b-09 failed" failed "$(st 4b-09-fn-cpu-profile)"
has "(n) naming perf" "perf record exited 2" "$(cat "$(rundir)steps/4b-09-fn-cpu-profile/stderr.txt")"
fi

# ---------- (k) ----------
if want k; then
fresh
(cd "$K" && STUB_HANG_PROFILE=1 exec bash "$K/run.sh" > "$T/run.out" 2>&1) &
pid=$!
wait_for "[ -f \"\$(rundir)steps/4b-09-fn-cpu-profile/.stub-attached\" ]" 60 || bad "(k) perf never attached in 4b-09"
kill -INT -- "-$pid"
wait_for "! kill -0 $pid 2>/dev/null" 20 || bad "(k) run did not stop within 20 s of SIGINT"
wait "$pid" 2>/dev/null
expect "(k) step interrupted" interrupted "$(st 4b-09-fn-cpu-profile)"
wait_for "[ \"\$(left)\" = none ]" 5
expect "(k) nothing left running (benchmark, perf, attacher)" none "$(left)"
expect "(k) level back to auto" auto "$(cat "$DPM")"
fi

# ---------- (o) ----------
if want o && [ "$(id -u)" != 0 ]; then
fresh
n0=$(ls -d "$K"/results/*/ | wc -l)
(cd "$K" && printf '\n' | script -qefc "bash run.sh --for 95m" "$T/o1.out" >/dev/null 2>&1); rc=$?
expect "(o) Enter means no: exit 1" 1 "$rc"
has "(o) it asks" "Continue without sudo? [y/N]" "$(cat "$T/o1.out")"
has "(o) with the sudo command, same arguments" "sudo bash run.sh --for 95m" "$(cat "$T/o1.out")"
expect "(o) nothing ran: no step, folder or lock" "none $n0 none" "$(st 4b-01-fn-base) $(ls -d "$K"/results/*/ | wc -l) $([ -d "$K/results/.lock" ] && echo lock || echo none)"
(cd "$K" && printf 'y\n' | PERSEUS_PAUSE_AFTER=4b-01-fn-base script -qefc "bash run.sh" "$T/o2.out" >/dev/null 2>&1); rc=$?
expect "(o) y goes on" "0 ok" "$rc $(st 4b-01-fn-base)"
fi

# ---------- (p) ----------
if want p; then
fresh
rm -f "$T/keys"; mkfifo "$T/keys"
(cd "$K" && STUB_HANG_PINNED=1 TERM=xterm script -qefc 'stty cols 100; bash run.sh' "$T/p.out" < "$T/keys" >/dev/null 2>&1) &
pid=$!
exec 7> "$T/keys"
[ "$(id -u)" = 0 ] || printf 'y\n' >&7
seen() { grep -qF -- "$1" "$T/p.out"; }
press() { printf '%s' "$1" >&7; wait_for "seen '$2'" 10 || bad "(p) key $1: no '$2' on screen"; }
wait_for "grep -qs 'stub saw' \"\$(rundir)steps/4b-03-fn-dpm-high/stderr.txt\"" 30 || bad "(p) never reached 4b-03's benchmark"
has "(p) the banner lists the keys" "p pause · s status · l output · m machine · ? keys" "$(cat "$T/p.out")"
press '?' "keys: p pause"
press l "output of 4b-03-fn-dpm-high"
has "(p) l: the test's stderr" "| stub saw: novis=0 level=high" "$(cat "$T/p.out")"
press s "4b. Flash-Next: where the time goes"
expect "(p) s: phases only, no second header" 1 "$(grep -c 'perseus lab kit' "$T/p.out")"
press m "busy 42% · core 1366Mhz · mem 300Mhz · clocks high"
press p "pausing after this test"
expect "(p) p sets the pause flag" yes "$([ -f "$K/results/.pause" ] && echo yes || echo no)"
press p "pause cancelled"
expect "(p) p again clears it" no "$([ -f "$K/results/.pause" ] && echo yes || echo no)"
bash "$K/run.sh" stop >/dev/null
wait_for "! pgrep -f -- '$K/run.sh' >/dev/null" 20 || bad "(p) run did not stop"
exec 7>&-; wait "$pid"
expect "(p) nothing left running" none "$(left)"
fi

# ---------- (s) ----------
if want s; then
fresh
out=$(cd "$K" && sh run.sh status 2>&1); rc=$?
expect "(s) 'sh run.sh' runs it in bash" 0 "$rc"
has "(s) sh: the whole plan" "4b. Flash-Next: where the time goes" "$out"
expect "(s) sh: no shell errors" "" "$(echo "$out" | grep -iE 'syntax error|not found|unbound')"
out=$(cd "$K" && PATH=/nonexistent "$BASH" run.sh status 2>&1); rc=$?
expect "(s) a PATH without the system's directories (a trimmed sudo secure_path)" "0 " "$rc $(echo "$out" | grep -iE 'not found')"
out=$(cd "$K" && PATH=/nonexistent /bin/sh run.sh status 2>&1); rc=$?
expect "(s) both: sh finds bash to re-run in" "0 " "$rc $(echo "$out" | grep -iE 'not found')"
S="$T/s-kit"; rm -rf "$S"; cp -a "$K" "$S"; rm -rf "$S/results"
mv "$S/lib/common.sh" "$S/lib/common.sh.gone"
out=$(cd "$S" && bash run.sh --no-copy 2>&1); rc=$?
expect "(s) lib/common.sh unreadable: exit 1" 1 "$rc"
has "(s) says which file" "can't read $S/lib/common.sh" "$out"
expect "(s) and stops there: no 'command not found', no results folder" "0 none" \
  "$(echo "$out" | grep -c 'not found') $([ -d "$S/results" ] && echo made || echo none)"
# cut short right after run_step: valid bash, but write_summary and the rest are gone
sed '/^write_summary() {/,$d' "$S/lib/common.sh.gone" > "$S/lib/common.sh"
out=$(cd "$S" && bash run.sh --no-copy 2>&1); rc=$?
expect "(s) lib/common.sh cut short: exit 1, before any test" "1 none" "$rc $([ -d "$S/results" ] && echo made || echo none)"
has "(s) says which file is incomplete" "$S/lib/common.sh is incomplete" "$out"
# cut inside a function: a syntax error, reported as the same one message
{ sed '/^run_step() {/,$d' "$S/lib/common.sh.gone"; sed -n '/^run_step() {/,/^}/p' "$S/lib/common.sh.gone" | head -5; } > "$S/lib/common.sh"
out=$(cd "$S" && bash run.sh --no-copy 2>&1); rc=$?
expect "(s) cut inside a function: exit 1 and only the kit's message" "1 0" "$rc $(echo "$out" | grep -c -v -e '^run.sh: ' -e '^  ')"
has "(s) naming the file" "$S/lib/common.sh is incomplete" "$out"
fi

# ---------- (h) ----------
if want h; then
fake_gpu
(
  DPM_RECORD="$T/h.restore"
  skip() { echo "skipped: $1"; exit 3; }; log() { :; }; record() { :; }
  . "$KITSRC/lib/gpu.sh"
  printf '%s\tauto\n' "$DPM" > "$DPM_RECORD"; echo high > "$DPM"  # a restore still owed
  ( dpm_set_high )
  echo "record: $(cut -f2 "$DPM_RECORD")"
  dpm_restore && echo "level: $(cat "$DPM")"
) > "$T/h.out" 2>&1
has "(h) a later pin keeps the true original" "record: auto" "$(cat "$T/h.out")"
has "(h) which the restore then applies" "level: auto" "$(cat "$T/h.out")"
fi

echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
