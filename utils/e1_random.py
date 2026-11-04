"""Method-independent per-step random inputs for the matched E1 experiment."""

import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def e1_random_stream(seed, phase, step, device=None):
    """Restore caller RNGs on exit; requires in-process (zero-worker) augmentation.

    phase+step keys do not depend on variant or model predictions. This isolates
    selection from dropout/augmentation/evaluation RNG consumption.
    """
    key = f'edgeqgfed-e1-v1:{int(seed)}:{phase}:{int(step)}'.encode('utf-8')
    stream_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2 ** 63)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index] if device is not None and device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(stream_seed)
            np.random.seed(stream_seed % (2 ** 32))
            torch.random.default_generator.manual_seed(stream_seed)
            if devices:
                torch.cuda.manual_seed(stream_seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
