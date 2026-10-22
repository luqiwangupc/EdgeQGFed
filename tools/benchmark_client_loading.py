"""Synthetic CIFAR-like data loading only: no downloads, models, training or trackers."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from datasets.SemiDatasets import _build_augmentations, build_base_transform
from datasets.parallel_clients import ParallelClientBatchLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', nargs='+', type=int, default=[0, 2, 4])
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--clients', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if min(args.clients, args.batch_size, args.rounds, args.cpu_threads) < 1 or min(args.workers) < 0:
        parser.error('Sizes/threads must be positive; workers must be nonnegative.')
    torch.set_num_threads(args.cpu_threads)
    count = args.clients * args.batch_size
    data = np.random.default_rng(7).integers(0, 256, size=(count, 32, 32, 3), dtype=np.uint8)
    targets = np.arange(count) % 10
    weak, strong = _build_augmentations('cifar10')
    entries, groups = [], {'Edge0': [], 'Edge1': []}
    for i in range(args.clients):
        client_id = f'Client{i}'
        groups[f'Edge{i % 2}'].append(client_id)
        entries.append(dict(
            client_id=client_id, indices=np.arange(i * args.batch_size, (i+1) * args.batch_size),
            data=data, targets=targets, data_name='cifar10',
            labeled_mask=torch.arange(args.batch_size) % 10 == 0,
            transform=build_base_transform('cifar10'), weak_transform=weak, strong_transform=strong,
        ))
    for workers in args.workers:
        loader = ParallelClientBatchLoader(entries, args.batch_size, 7, workers=workers)
        try:
            # Startup is reported separately; do not mistake spawn time for steady-state speed.
            durations = []
            for step in range(args.rounds + 1):
                start = time.perf_counter()
                loader.prepare(groups, step)
                for ids in groups.values():
                    loader.get_edge_batch(ids)
                durations.append(time.perf_counter() - start)
            mean = sum(durations[1:]) / args.rounds
            print(json.dumps(dict(
                workers=workers, cpu_threads=args.cpu_threads, worker_threads=1,
                clients=args.clients, batch_size=args.batch_size,
                startup_round_s=durations[0], steady_round_s=mean,
                samples_per_s=count/mean,
                scope='synthetic_data_loading_only_not_training_speed',
            )), flush=True)
        finally:
            loader.close()


if __name__ == '__main__':
    main()
