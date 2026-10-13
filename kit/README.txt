PERSEUS LAB KIT: what to do in the lab
=======================================

It measures the Mac Pro (CPU, RAM layout, memory speed) and how fast the
candidate models run on it, and saves everything to this USB. It needs no
internet and installs nothing. It runs the most important tests first, so
even a short visit gets the key data, and it continues where it stopped.

1. Plug the USB into the Mac Pro and boot Linux.
   (Under macOS it still records the hardware, but the model tests need Linux.)

2. Open a terminal in this folder, e.g.
       cd "/media/$USER/<USB name>/PERSEUS-KIT"

3. Start it. For a 2-hour visit:
       sudo bash run.sh --for 95
   It stops starting new tests after 95 minutes and finishes the one it is
   on (a few minutes), leaving time to pack up. Phases 1-5 (~80 min) are
   the important data; phase 6 is extras.

4. While it runs, from a second terminal (same folder, also with sudo):
       sudo bash run.sh status    how far it is
       sudo bash run.sh pause     finish the current test, then stop
       sudo bash run.sh stop      stop right now (same as Ctrl-C in its window)
   Don't use the Mac for anything else meanwhile; it skews the numbers.

5. Wait for "Paused", "Stopped", "Time limit ... reached" or "DONE" before
   ejecting the USB. Bring it back; everything is in PERSEUS-KIT/results/
   (summary.txt there has the headline numbers).

Next visit: run the same command again (sudo bash run.sh --for 95). It
continues from where it stopped. If a test was cut off by a crash or power
cut it is skipped, not retried; add --retry-failed to retry it.

Disk use: models are copied in the background to /var/tmp/perseus-kit-local
on the Mac's disk, because USB 2.0 is too slow to load them from repeatedly:
22.1 GB for the core kit, 39.5 GB for standard, plus ~0.3 GB of programs.
Each copy starts only if the disk has the model's size + 5 GB free; otherwise
that model runs straight from the USB (slow loads). Delete them when you're
completely done:
       sudo bash run.sh clean

USB format: must be exFAT or ext4. FAT32 can't hold files over 4 GB.

DIAGNOSTICS-ONLY KIT (no models)
--------------------------------
A smaller kit that skips the models: it records the machine facts (CPU, RAM
layout and speed, NUMA nodes), measures memory bandwidth, and checks and
times our speed-up code on this CPU. Phases 1-2, about 10 minutes; start it
the same way (sudo bash run.sh). The model tests show as 'absent'.
It needs well under 1 GB, so any USB stick works, FAT32 included (the format
most small sticks ship with). If the same stick later gets the models, just
run the same command again; the model tests run then.
