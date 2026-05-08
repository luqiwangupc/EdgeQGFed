import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import numpy as np


class BatchDataloader:
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False, worker_init_fn=None,
                 prefetch_factor=2, persistent_workers=False):
        if num_workers > 0:
            try:
                mp.set_start_method('fork')
            except RuntimeError:
                pass

        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else None,
        )
        self.iterator = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._init_iterator()

    def _init_iterator(self):
        self.iterator = iter(self.dataloader)

    def get_batch(self):
        if self.iterator is None:
            self._init_iterator()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self._init_iterator()
            batch = next(self.iterator)
        return batch

    def close(self):
        if hasattr(self, 'iterator'):
            self.iterator = None
        if hasattr(self, 'dataloader'):
            del self.dataloader

    def __del__(self):
        self.close()


def split_classes(max_length, split_num, edge_id=-1):
    if split_num <= 0:
        raise ValueError(f"split_num must be greater than 0, got {split_num}")
    if edge_id != -1:
        numbers = np.random.permutation(max_length) + edge_id
    else:
        numbers = np.random.permutation(max_length)
    result = []

    if split_num <= max_length:
        base_length = max_length // split_num
        extra_count = max_length % split_num
        start_indices = np.zeros(split_num, dtype=int)
        lengths = np.full(split_num, base_length)
        lengths[:extra_count] += 1
        end_indices = np.cumsum(lengths)
        start_indices[1:] = end_indices[:-1]
        result = [numbers[start:end] for start, end in zip(start_indices, end_indices)]
    else:
        base_length = max(1, 10 // split_num)
        for i in range(split_num):
            shifted_array = np.roll(numbers, -i)[:base_length]
            result.append(shifted_array)

    return result