#!/usr/bin/env bash
# Runs a command in the kit build image under a hard memory cap (no swap), with a host-side
# watchdog that kills it if host MemAvailable falls below a floor. Models/bins mounted read-only.
# usage: capped-run.sh CAP_GB FLOOR_MB NAME -- CMD...
set -uo pipefail
cap="$1"; floor="$2"; name="$3"; shift 4
W="$HOME/.cache/perseus-kit"
docker rm -f "$name" >/dev/null 2>&1
docker run -d --name "$name" --network none --memory="${cap}g" --memory-swap="${cap}g" \
  -u "$(id -u):$(id -g)" -v "$W/bin:/bin-kit:ro" -v "$W/models:/models:ro" \
  perseus-kit-build:bookworm "$@" >/dev/null || exit 1
min=999999
while [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" = true ]; do
  avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
  [ "$avail" -lt "$min" ] && min=$avail
  if [ "$avail" -lt "$floor" ]; then
    echo "WATCHDOG: host MemAvailable ${avail} MB < ${floor} MB, killing $name"
    docker kill "$name" >/dev/null
    break
  fi
  sleep 1
done
docker logs "$name" 2>&1
echo "container exit: $(docker inspect -f '{{.State.ExitCode}} oom={{.State.OOMKilled}}' "$name")  lowest host MemAvailable: ${min} MB"
docker rm -f "$name" >/dev/null
