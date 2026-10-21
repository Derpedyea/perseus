# The AMD GPU's sysfs knobs for run.sh's phase 4b: its performance level (DPM), pinned at 'high'
# by the steps that A/B it and always put back, and a sampler of its busy % and clocks.
# Bash 3.2-compatible, like common.sh.
#
# Who holds the level: the kernel, until reboot. Before the first change, the pre-run value and the
# switch's path go to $DPM_RECORD; the runner restores from it after every step, in its Ctrl-C/exit
# traps and at its next start (a run cut off by kill -9 or a crash), and deletes it only once the
# read-back matches. See kit/DESIGN.md, phase 4b.

# Tests point this at a fake sysfs tree.
DRM_ROOT="${PERSEUS_SYSFS_DRM:-/sys/class/drm}"

# The one AMD GPU's performance-level switch, or nothing (and false) unless there is exactly one.
dpm_path() {
  local d found=""
  for d in "$DRM_ROOT"/card[0-9]*/device; do
    [ -f "$d/power_dpm_force_performance_level" ] || continue
    [ "$(cat "$d/vendor" 2>/dev/null)" = 0x1002 ] || continue
    [ -z "$found" ] || return 1
    found="$d/power_dpm_force_performance_level"
  done
  [ -n "$found" ] && echo "$found"
}

# For env.txt: the level now.
dpm_level() {
  local p v
  p=$(dpm_path) && read -r v < "$p" 2>/dev/null && echo "$v" || echo "n/a"
}

# Inside a step: pin the level at 'high' or end the step. Records the pre-run value first (once:
# a record left by an earlier change still holds the true original).
dpm_set_high() {
  local p cur
  p=$(dpm_path) || skip "no single AMD GPU performance-level switch in sysfs"
  [ -w "$p" ] || skip "pinning the GPU's clocks needs sudo (sudo bash run.sh)"
  read -r cur < "$p" || { echo "can't read $p" >&2; exit 1; }
  if [ ! -f "$DPM_RECORD" ]; then
    printf '%s\t%s\n' "$p" "$cur" > "$DPM_RECORD.tmp" && mv "$DPM_RECORD.tmp" "$DPM_RECORD" \
      || { echo "can't write $DPM_RECORD" >&2; exit 1; }
  fi
  echo high > "$p" 2>/dev/null
  read -r cur < "$p"
  [ "$cur" = high ] || { echo "the GPU's performance level is '$cur' after writing 'high' to $p" >&2; exit 1; }
}

# In the runner: put the level back from the record. 0 = nothing to do or restored; 1 = still
# owed (the record stays, and the caller's next call retries).
dpm_restore() {
  [ -f "$DPM_RECORD" ] || return 0
  local p="" v="" cur="" n ok=1
  IFS="$(printf '\t')" read -r p v < "$DPM_RECORD"
  case "$p" in "$DRM_ROOT"/card*/device/power_dpm_force_performance_level) ;; *) ok=0 ;; esac
  n=${p#"$DRM_ROOT"/card}; n=${n%%/*}
  case "$n" in ''|*[!0-9]*) ok=0 ;; esac
  case "$v" in ''|*[!a-z_]*) ok=0 ;; esac
  if [ "$ok" = 0 ]; then
    log "!! $DPM_RECORD is malformed; check the GPU's performance level by hand, then delete it"
    return 1
  fi
  if [ ! -e "$p" ]; then
    # GPU removed, or renumbered by a reboot: the setting went with the device.
    log "!! $p is gone, so its level ('high') went with it; dropping $DPM_RECORD"
    rm -f "$DPM_RECORD"
    return 0
  fi
  read -r cur < "$p" 2>/dev/null
  if [ "$cur" != "$v" ]; then
    echo "$v" > "$p" 2>/dev/null
    read -r cur < "$p" 2>/dev/null
  fi
  if [ "$cur" = "$v" ]; then
    rm -f "$DPM_RECORD"
    record "GPU performance level is '$v' again ($p)"
    return 0
  fi
  log "!! couldn't put the GPU's performance level back to '$v' ($p says '$cur'); retrying after the next test, at exit and at the next start"
  return 1
}

# Inside a step: GPU busy % and current core/memory clocks, 5 times a second, into FILE as
# "uptime busy% sclk mclk", until gpu_sampler_stop. Builtins plus one 'sleep' per sample (it runs
# beside the benchmark). It polls the step's own pid, so a step killed with it still running (Ctrl-C
# reaches the step but not this background loop) can't leave it behind.
gpu_sampler_start() { # file
  local p dev
  GPU_SAMPLER=""
  p=$(dpm_path) || { echo "no single AMD GPU in sysfs: no samples" > "$1"; return 0; }
  dev=${p%/*}
  local step=$BASHPID
  (
    trap '' INT
    alive() {
      kill -0 "$step" 2>/dev/null || return 1
      local _p _c st=""
      [ -r "/proc/$step/stat" ] && read -r _p _c st _ < "/proc/$step/stat"
      [ "$st" != Z ]
    }
    clk=""
    cur_clock() { # pp_dpm file -> $clk, the starred level's clock ("1: 466Mhz *")
      local l
      clk="?"
      while read -r l; do case "$l" in *'*'*) l=${l#*: }; clk=${l%% *}; return ;; esac; done < "$1"
    }
    echo "# uptime_s busy_pct sclk mclk ($dev)"
    while alive; do
      read -r up _ < /proc/uptime
      read -r busy < "$dev/gpu_busy_percent" 2>/dev/null || busy="?"
      cur_clock "$dev/pp_dpm_sclk"; s=$clk
      cur_clock "$dev/pp_dpm_mclk"
      echo "$up $busy $s $clk"
      sleep 0.2
    done
  ) > "$1" 2>/dev/null &
  GPU_SAMPLER=$!
}

gpu_sampler_stop() {
  [ -n "${GPU_SAMPLER:-}" ] || return 0
  kill "$GPU_SAMPLER" 2>/dev/null
  wait "$GPU_SAMPLER" 2>/dev/null
  GPU_SAMPLER=""
}

LIB_END=gpu  # last line: run.sh checks it to catch a copy cut short
