"""Small helpers."""

import os

import torch


def tune_cpu(threads=2):
    """Keep training from crowding out the desktop: run on 6 of the M1 Pro's 8 performance cores (leaving 2 and the
    efficiency cores to the compositor and other apps), at lower priority, with `threads` threads."""
    if os.cpu_count() == 10 and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, range(4, 10))
    try:
        os.nice(5)
    except OSError:
        pass
    torch.set_num_threads(threads)
