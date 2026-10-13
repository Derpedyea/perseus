#!/usr/bin/env bash
# Background copier started by run.sh: stages models from the USB to the local disk in run
# order, so benchmark steps don't sit waiting on USB 2.0. run.sh starts it at idle CPU/IO
# priority in its own session and kills that session when it exits.
#   prefetch.sh KIT LOCAL_MODELS_DIR LOG "file size sha256"...
set -u
KIT="$1" DIR="$2" LOG="$3"; shift 3
. "$KIT/lib/common.sh"
mkdir -p "$DIR"
echo $$ > "$DIR/.prefetch.pid"
for spec in "$@"; do
  set -- $spec
  [ -f "$KIT/models/$1" ] || continue
  [ -f "$DIR/$1.ok" ] && continue
  echo "$1" > "$DIR/.copying"
  echo "[$(date +%H:%M:%S)] copy $1" >> "$LOG"
  copy_model "$DIR" "$KIT/models" "$1" "$2" "$3" >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] -> exit $?" >> "$LOG"
done
rm -f "$DIR/.copying" "$DIR/.prefetch.pid"
