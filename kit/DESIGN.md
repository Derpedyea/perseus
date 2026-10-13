# Lab kit design

**Invariant:** tests run in a fixed priority order (phases 1-6), and every planned step ends with exactly one recorded outcome (`ok`, `skipped` + reason, `absent` + reason, `failed`, `timeout`, `interrupted` or `crashed`), and its raw output sits next to it. Nothing is silently dropped, and the kit never changes the lab machine beyond `/var/tmp/perseus-kit-local` and the page-cache drop in the one `sudo` test.

| Exit | Behaviour | Test |
|---|---|---|
| done | `status=ok`; summary.txt + `FINISHED` marker written | local e2e run |
| failed | `status=failed`, exit code in `status.json`, stderr kept; run continues with the next step | e2e: corrupted model copy fails its stage step |
| skipped | `status=skipped` + reason (single NUMA node, no sudo, not enough RAM, model not on USB) | e2e on this 1-node box |
| timeout | `timeout` exit 124/137 → `status=timeout` | untested (see Known gaps) |
| cancelled (Ctrl-C) | the signal reaches the benchmark itself (`timeout --foreground`, same process group), current step → `interrupted`, summary written, lock released, exit 130 within seconds; rerun repeats that step | e2e: SIGINT to the process group mid-bench; stops < 20 s, no `llama-*` left |
| terminal closed (SIGHUP) | same as Ctrl-C | shares the trap; not separately tested |
| crash / power loss / OOM kill mid-step | `started` marker without status → next run marks it `crashed` and does **not** retry it (so one bad step can't wedge every rerun); `--retry-failed` reruns it. A dead runner left as a zombie doesn't hold the lock | e2e: kill -9 the process group mid-step, with and without an init reaping zombies |
| retried | `--retry-failed` reruns failed/timeout/crashed; ok/skipped never rerun | e2e |
| duplicated (two runs at once) | `mkdir` lock with host+pid; live owner → refuse; stale owner → atomic `mv` takeover, one winner | e2e: second run while first is live |
| paused (`run.sh pause`) | flag file; the runner finishes the current test, then stops cleanly (exit 0, summary written); a stage step waiting on the copier ends `interrupted` and reruns | e2e: pause mid-bench |
| stopped (`run.sh stop`) | SIGINT to the runner's process group (recorded in the lock) = Ctrl-C | e2e |
| time limit (`--for N`) | no new test starts after N minutes (uptime clock, immune to a wrong wall clock); the running one finishes | e2e with `--for 1` |
| background copy | copier in its own session at idle CPU/IO priority (the runner records its pid before any step runs, so a stage step never copies alongside a copier that hasn't started yet), copies in plan order, sha256 → `.ok`/`.bad`/`.nospace`; killed with the runner; a copier orphaned by a crash is killed on the next start and its `.part` removed; each bench records whether a copy was running (`env.txt`) | e2e: copy mode, stop mid-copy, kill -9 mid-copy; e2e-diag (b): copier delayed 3 s at start, the resumed run's first stage step waits for it |
| reordered | steps have fixed ids; resume keys on the id, not position | by construction |
| USB moved to another machine | lock owner host differs → treated as stale; results dir is per host | by construction |
| identity change (sudo vs not) | local cache path fixed (`/var/tmp`), not `$HOME` | by construction |
| wrong CPU build | variant picked from `/proc/cpuinfo` (or forced with `PERSEUS_VARIANT`, validated), then proven with a tiny real benchmark; SIGILL → next older build | QEMU (`build-kit.sh test`): emulator traps each newer extension, each build runs every shipped kernel on its oldest target CPU, a newer build traps. The step-down loop itself is untested |
| perseus kernels wrong on the real CPU | steps 2-00 (SSSE3, multi-column and repacked-IQ3_S kernels), 2-00b (tiled matmul), 2-00d (IQ3_S repack through ggml's graph API) and 2-00f (Flash-Next formats q6_K/q5_K/iq4_nl/q2_0, plus q4_1) run the self-tests on the lab box first; a failure is recorded like any step, so Saluki (2-00..2-00d) or Qwen3.8-27B (2-00f: its q6_K/q5_K/q4_1 tensors) numbers after it are known-suspect | `kernels/test_ssse3_kernels.c`, `kernels/test_fn_kernels.c` (both mutation-tested), llama.cpp's `test-tiled-mulmat`, `kernels/test_repack_mulmat.c` (fails when the repack is off), QEMU Penryn/Nehalem |
| repacked weights use more RAM than the file | IQ3_S tensors grow 31% at load (Saluki +0.39 GB, 5%); `ram_check` keeps a 15% + 1.5 GB margin over the file size | by arithmetic |
| A/B steps (5-08 multi-column off, 6-14 tiled off, 6-19 repack off) | the step exports `PERSEUS_MC=0` / `GGML_CPU_TILED_MM=0` / `PERSEUS_REPACK=0` inside its own subshell, so later steps never inherit it; `env.txt` records the overrides; a malformed `PERSEUS_MC` / `PERSEUS_REPACK` aborts instead of silently defaulting | `PERSEUS_MC=off` aborts; with `PERSEUS_MC=0` the self-test fails (the switch works); MTP output identical with it on and off |
| model not on this USB (diag kit; core kit's standard-tier models) | `status=absent` + reason. Not final: every resume re-evaluates it, so a USB re-assembled with a bigger tier runs it. Only the USB counts (a local copy from an earlier, bigger kit doesn't), so a diag kit never runs model tests. `status` counts them per phase ("need models not on this USB"); summary.txt lists each one plus their total | `kit/test/e2e-diag.sh` (b): diag run paused after 2-05, USB re-assembled with core, resume runs 2-01..2-05. Fails when `absent` is final like `skipped` |
| diag kit (no models) | phases 1-2 run; every model step ends `absent`; FINISHED and DONE (with the absent count); no background copier and nothing in the local dir but `bin/`; the startup line says diagnostics-only, ~10 min. With no tiny model, one trial of the kernel self-test proves the CPU build instead of a tiny benchmark | e2e-diag (a), on a FAT32 image mounted by `udisksctl` |
| USB re-assembled after a FINISHED run | FINISHED is final: the next run starts a new results folder and repeats phases 1-2 (~10 min) before the model tests | by construction (`pick_run_dir`) |
| FAT32 USB | `assemble` refuses it for core/standard (files over 4 GB) and allows diag. udisks mounts vfat with `showexec` (udisks2 2.11.2's builtin `vfat_defaults=uid=$UID,gid=$GID,shortname=mixed,utf8=1,showexec,flush`), so nothing on the stick is executable (exec gives 126, chmod is a no-op); run.sh only executes its local copy of `bin/` | e2e-diag (a) on a FAT32 image mounted by `udisksctl` |
| damaged model copy | sha256 checked after copying to local disk; mismatch → `.bad`, model steps skip | e2e: one flipped byte in the USB copy |
| RAM too small for a model | skipped with the numbers, never started (avoids OOM-killing the box) | untested (see Known gaps) |
| hidden memory use beyond the weights | completion/MTP steps pass `-c` (default is the trained 262k context = ~17 GB KV on Qwen3.8-27B); llama-bench sizes its own context | 27B + MTP head runs in a 12 GB cgroup with `-c 1024`; OOM-killed twice without it |
| USB read-only / full | refuses to start with a message | untested (see Known gaps) |

Copies others hold: the page cache holds model files between steps (deliberate; `distribute-cold` drops it first); `/var/tmp/perseus-kit-local` holds binaries + models until `run.sh clean` (a diag kit puts only `bin/` there). Model copies left there by an earlier kit are not used unless the USB carries that model too.

## Known gaps
- A step whose command itself exits 3 is recorded as `skipped`: `skip()` and such a command share the exit code. `absent` uses a marker file, so it can't be faked that way.
- `assemble` never deletes: assembling a smaller tier onto a stick that holds a bigger tier's models leaves them (it logs which), and run.sh uses them.
- Untested exits: timeout (needs hours of runtime or a test-only knob), RAM-too-small skip, USB read-only/full refusal, SIGHUP. The code paths are short; each records an explicit status if hit.
- The macOS path (sysinfo; membw only if a compiler exists) is untested: no Mac here.
- The wall clock on an offline box may be wrong; durations use `/proc/uptime`, timestamps are only as good as the clock.
- The background copy can overlap the first benchmarks (idle priority, and `env.txt` records it). Phase 2-3 numbers may be a few % low on a box where the copy competes; `env.txt` says which steps overlapped it.
- Throttling or background load during a bench is recorded (loadavg before each bench, process list in sysinfo) but not prevented.
