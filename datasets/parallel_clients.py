"""Bounded, ordered client-batch preparation using spawn workers.

One task is one client's batch, not a whole ES batch. The parent samples indices;
workers only read data and create the existing three views. A keyed random-input
protocol makes zero-worker and multi-worker executions comparable. It does not
reproduce the old interleaved edge_step RNG stream.
"""

import hashlib
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from datasets.SemiDatasets import MutableEdgeBatchSampler, ReusableEdgeBatchDataset
from utils.e1_random import e1_random_stream


INPUT_PROTOCOL = 'client_keyed_v2'
MODEL_BATCH_KEYS = ('img', 'label', 'is_labeled', 'weak_img', 'strong_img')


def initialize_cpu_worker(worker_id, threads=1):
    """Spawn workers own their CPU thread pools and never initialize CUDA."""
    torch.set_num_threads(threads)


class ClientTaskDataset(Dataset):
    def __init__(self, records, seed):
        self.records = records
        self.seed = int(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, task):
        client_id, step, indices = task
        with e1_random_stream(self.seed, f'client_views:{client_id}', step):
            batch = default_collate([self.records[index] for index in indices])
        batch['_client_id'] = client_id
        return batch


class ClientTaskSampler:
    """Only the parent owns this mutable task list; workers receive immutable keys."""
    def __init__(self):
        self.tasks = []

    def __iter__(self):
        return iter(self.tasks)

    def __len__(self):
        return len(self.tasks)


class ParallelClientBatchLoader:
    def __init__(self, client_entries, batch_size, seed, workers=4, prefetch_factor=1,
                 pin_memory=False, labeled_fraction=0., class_sampling_weights=None):
        if workers < 0 or prefetch_factor < 1:
            raise ValueError('client_batch_workers must be >= 0 and prefetch_factor >= 1')
        self.seed = int(seed)
        self.workers = int(workers)
        self.records = ReusableEdgeBatchDataset(client_entries)
        self.batch_sampler = MutableEdgeBatchSampler(
            self.records.client_to_indices, batch_size,
            client_to_labeled_indices_by_class=self.records.client_to_labeled_indices_by_class,
            labeled_fraction=labeled_fraction, class_sampling_weights=class_sampling_weights,
        )
        self.task_sampler = ClientTaskSampler()
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        options = {}
        if workers:
            options.update(
                multiprocessing_context='spawn', persistent_workers=True,
                prefetch_factor=int(prefetch_factor),
                worker_init_fn=partial(initialize_cpu_worker, threads=1),
            )
        self.loader = DataLoader(
            ClientTaskDataset(self.records, seed), batch_size=None, sampler=self.task_sampler,
            num_workers=int(workers), pin_memory=self.pin_memory,
            # Iterator/worker startup must not consume model/dropout RNGs.
            generator=torch.Generator().manual_seed(self.seed), **options,
        )
        self.iterator = None
        self.remaining = []
        self.batch_plan = {}

    def prepare(self, clients_by_edge, step):
        if self.remaining:
            raise RuntimeError('Previous client batches were not fully consumed')
        tasks, plan = [], {}
        for client_ids in clients_by_edge.values():
            for client_id in client_ids:
                if client_id in plan:
                    raise ValueError(f'Duplicate client task: {client_id}')
                with e1_random_stream(self.seed, f'client_batch:{client_id}', step):
                    self.batch_sampler.set_clients([client_id])
                    indices = next(iter(self.batch_sampler), [])
                if not indices:
                    raise ValueError(f'Successful client has no samples: {client_id}')
                tasks.append((client_id, int(step), tuple(indices)))
                plan[client_id] = {
                    'sample_count': len(indices),
                    'record_indices_sha256': hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest(),
                    'batch_stream': f'client_batch:{client_id}',
                    'view_stream': f'client_views:{client_id}',
                }
        self.task_sampler.tasks = tasks
        self.batch_plan = plan
        self.remaining = [task[0] for task in tasks]
        self.iterator = iter(self.loader) if tasks else None

    def get_edge_batch(self, client_ids):
        if not client_ids:
            return None
        if list(client_ids) != self.remaining[:len(client_ids)]:
            raise RuntimeError('Client batches must be consumed in the planned ES/client order')
        parts = []
        for client_id in client_ids:
            batch = next(self.iterator)
            if batch.pop('_client_id') != client_id:
                raise RuntimeError('Worker output order differs from the client plan')
            parts.append(batch)
        del self.remaining[:len(client_ids)]
        if len(parts) == 1:
            return parts[0]
        batch = {key: torch.cat([part[key] for part in parts], dim=0) for key in MODEL_BATCH_KEYS}
        if self.pin_memory:
            # cat creates new pageable storage even if individual client parts were pinned.
            batch = {key: tensor.pin_memory() for key, tensor in batch.items()}
        return batch

    def close(self):
        # DataLoader has no public close API. Shut down its persistent pool on normal
        # completion AND exceptions via train()'s ExitStack, rather than leaving it alive.
        iterator = getattr(self.loader, '_iterator', None)
        if iterator is not None:
            shutdown = getattr(iterator, '_shutdown_workers', None)
            if shutdown is not None:
                shutdown()
            self.loader._iterator = None
        self.iterator = None
        self.remaining = []
