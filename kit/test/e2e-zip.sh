#!/usr/bin/env bash
# The diag kit zip (build-kit.sh zip) works however it was extracted: unzip -t passes, the zip
# holds exactly one PERSEUS-KIT/ (MANIFEST.json, no results/, no models), and after extracting it
# and stripping every exec bit (chmod -R a-x, as FAT32/exFAT/NTFS round trips may) 'bash run.sh'
# reaches DONE with the phase 1-2 diagnostics done (phase 3 skips: Debian 12's glibc predates
# the Vulkan build's). Runs as root, like 'sudo bash run.sh' in the lab (root can still enter
# directories without x).
# usage: e2e-zip.sh KIT.zip
set -uo pipefail
ZIP="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
W="${PERSEUS_KIT_WORK:-$HOME/.cache/perseus-kit}"
T=$(mktemp -d "$W/zip-test.XXXX"); C="perseus-e2e-zip-${E2E_TAG:-0}"
pass=0; fail=0
ok() { echo "PASS  $1"; pass=$((pass + 1)); }
bad() { echo "FAIL  $1"; fail=$((fail + 1)); }
expect() { [ "$2" = "$3" ] && ok "$1 ($3)" || bad "$1: expected '$2', got '$3'"; }
inside() { docker exec "$C" bash -c "$1"; }
cleanup() { inside "rm -rf /x/*" >/dev/null 2>&1; docker rm -f "$C" >/dev/null 2>&1; rm -rf "$T"; }
trap cleanup EXIT

unzip -tq "$ZIP" >/dev/null; expect "unzip -t" 0 "$?"
names=$(unzip -Z1 "$ZIP")
expect "one top-level folder" "PERSEUS-KIT" "$(echo "$names" | cut -d/ -f1 | sort -u | tr '\n' ' ' | sed 's/ $//')"
expect "has MANIFEST.json" 1 "$(echo "$names" | grep -cx 'PERSEUS-KIT/MANIFEST.json')"
expect "no results/" 0 "$(echo "$names" | grep -c '^PERSEUS-KIT/results/')"
expect "no models" 0 "$(echo "$names" | grep '^PERSEUS-KIT/models/.' | wc -l)"
expect "no symlinks" 0 "$(unzip -Zl "$ZIP" | grep -c '^l')"

mkdir -p "$T/x" && (cd "$T/x" && unzip -q "$ZIP")
P=/x/PERSEUS-KIT
# read the plan now: after chmod -R a-x only root (the container) can enter these directories
models=$(sed -n 's/^  S \([^ ]*\) step_\(stage\|sanity\|bench\) .*/\1/p' "$T/x/PERSEUS-KIT/run.sh")
diag=$(sed -n 's/^  S \([12]-[^ ]*\) .*/\1/p' "$T/x/PERSEUS-KIT/run.sh" | grep -vxF -f <(echo "$models"))
expect "plan read (phase 1-2 diagnostic steps)" yes "$([ "$(echo "$diag" | wc -w)" -ge 10 ] && echo yes || echo "no: $diag")"
docker rm -f "$C" >/dev/null 2>&1
docker run -d --init --memory=10g --memory-swap=10g -e PERSEUS_VARIANT=nehalem --name "$C" --network none \
  -v "$T/x:/x" perseus-kit-build:bookworm sleep infinity >/dev/null
inside "chmod -R a-x /x/PERSEUS-KIT"
expect "no exec bit left anywhere" 0 "$(inside "find $P -perm /111 | wc -l")"
inside "cd $P && bash run.sh > /tmp/run.out 2>&1; echo \$? > /tmp/run.rc"
expect "run exits 0" 0 "$(inside 'cat /tmp/run.rc')"
expect "DONE reached" 1 "$(inside "grep -c 'DONE (' /tmp/run.out")"
RD=$(inside "ls -d $P/results/*/ | tail -1")
res=$(for s in $diag; do echo "$s $(inside "cat $RD/steps/$s/status 2>/dev/null || echo none")"; done)
expect "phase 1-2 diagnostics ok (multi-node membw skipped on 1 node)" "" \
  "$(echo "$res" | grep -v ' ok$' | grep -v -E '^1-0[6-9]-.* skipped$' | tr '\n' ' ')"
echo "$res" | sed 's/^/      /'

echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]
