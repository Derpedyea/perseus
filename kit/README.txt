PERSEUS LAB KIT: what to do in the lab
=======================================

It measures the Mac Pro (CPU, RAM layout, memory speed, its Radeon graphics
card) and how fast the candidate models run on it, and saves everything to
this USB. It needs no
internet and installs nothing. It runs the most important tests first, so
even a short visit gets the key data, and it continues where it stopped.

1. Plug the USB into the Mac Pro and boot Linux.
   (Under macOS it still records the hardware, but the model tests need Linux.)

2. Open a terminal in this folder, e.g.
       cd "/media/$USER/<USB name>/PERSEUS-KIT"

3. Start it. For a 2-hour visit:
       sudo bash run.sh --for 95m
   It stops starting new tests after 95 minutes and finishes the one it is
   on (a few minutes), leaving time to pack up. --for also takes hours:
   --for 8h, --for 1h30m. Overnight, leave it out: it runs every test.
   sudo bash run.sh --help lists everything it can do. A whole run (phases
   1-5) takes about 1 hour 20 minutes, plus the copy of any big model that
   isn't on the Mac's disk from last time (Flint 25.1 GB: ~12 minutes;
   Flash-Next 75.8 GB: ~35 minutes).
   Please do use sudo: phase 4b pins the graphics card's clocks for a few
   tests (and puts them back after each one) and profiles the CPU, and both
   need it. Without sudo those tests are skipped; started without sudo, it
   asks first (Enter = no, so you can start it again with sudo).
   Phases 3, 4, 4b and 5 use the graphics card: the screen may stutter for a
   moment. Phase 4 waits until the big models are copied off the USB (phases
   1-3 run meanwhile), unless they are still on the Mac's disk.

4. While it runs, keys in its window: p pause after this test (p again
   cancels), s status, l the test's last output lines, m CPU/RAM/GPU now,
   ? the keys. Or from a second terminal (same folder, also with sudo):
       sudo bash run.sh status    how far it is
       sudo bash run.sh pause     finish the current test, then stop
       sudo bash run.sh stop      stop right now (same as Ctrl-C in its window)
   Don't use the Mac for anything else meanwhile; it skews the numbers.

5. Wait for "Paused", "Stopped", "Time limit ... reached" or "Done" before
   ejecting the USB. Bring it back; everything is in PERSEUS-KIT/results/
   (summary.txt there has the headline numbers).

Next visit: run the same command again (sudo bash run.sh --for 95m). It
continues from where it stopped. If a test was cut off by a crash or power
cut it is skipped, not retried; add --retry-failed to retry it.

Disk use: models are copied in the background to /var/tmp/perseus-kit-local
on the Mac's disk, because USB 2.0 is too slow to load them from repeatedly:
1.0 GB for the core kit, 102 GB for full, plus ~0.4 GB of programs.
Each copy starts only if the disk has the model's size + 5 GB free; otherwise
that model runs straight from the USB (slow loads). Delete them when you're
completely done:
       sudo bash run.sh clean
(If a run was cut off while the graphics card's clocks were pinned, the next
run or 'clean' puts them back; so does a reboot.)

USB format: must be exFAT or ext4. FAT32 can't hold files over 4 GB.

Faster copy (optional): the USB drive is a normal 2.5" SATA SSD in a case.
If you may open the Mac, the bare drive in a free drive bay copies about 7x
faster than USB 2.0 (the full kit in ~8 minutes instead of ~55). Run the kit
from wherever the drive is mounted; nothing else changes.

DIAGNOSTICS-ONLY KIT (no models)
--------------------------------
A smaller kit that skips the models: it records the machine facts (CPU, RAM
layout and speed, NUMA nodes, graphics card), measures memory bandwidth,
and checks and times our speed-up code on this CPU and graphics card.
Phases 1-3, about 15 minutes; start it
the same way (sudo bash run.sh). The model tests show as 'absent'.
It needs well under 1 GB, so any USB stick works, FAT32 included (the format
most small sticks ship with). If the same stick later gets the models, just
run the same command again; the model tests run then.
From the zip (PERSEUS-KIT-diag.zip): extract it to the top of the drive with
any unzip tool, then: cd "/media/$USER/<drive>/PERSEUS-KIT" && sudo bash run.sh
