import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets

from datasets import split_classes


def get_n_classes(name):
    if name == 'cifar10':
        n_classes = 10
    elif name == 'cifar100':
        n_classes = 100
    elif name == 'fashionmnist':
        n_classes = 10
    elif name == 'svhn':
        n_classes = 10
    elif name == 'mnist':
        n_classes = 10
    else:
        raise NotImplementedError(f'Unknown dataset: {name}')
    return n_classes


def get_dataset_by_name(name, train=True, transform=None):
    if name == 'cifar10':
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        data = dataset.data
        targets = dataset.targets
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
        data = dataset.data
        targets = dataset.targets
    elif name == 'fashionmnist':
        dataset = datasets.FashionMNIST(root='./data', train=train, download=True, transform=transform)
        data = dataset.data
        targets = dataset.targets
    elif name == 'svhn':
        dataset = datasets.SVHN(root='./data', split='train' if train else 'test', download=True, transform=transform)
        data = dataset.data
        targets = dataset.labels
    elif name == 'mnist':
        dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        data = dataset.data
        targets = dataset.targets
    else:
        raise NotImplementedError(f'Dataset {name} not implemented')
    return data, targets


def _prepare_image(image, data_name):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if data_name == 'svhn':
        image = image.transpose(1, 2, 0)
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
        image = np.concatenate([image, image, image], axis=-1)
    return image


def build_base_transform(data_name, image_size=224, normalize='imagenet'):
    if normalize == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == 'half':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise ValueError(f'Unknown normalize mode: {normalize}')
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Normalize(mean=mean, std=std),
    ])


def _build_augmentations(data_name='cifar10'):
    if data_name in {'svhn', 'mnist', 'fashionmnist'}:
        weak_transform = transforms.Compose([
            transforms.RandomRotation(degrees=5),
        ])
        strong_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
        return weak_transform, strong_transform

    weak_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    strong_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ])
    return weak_transform, strong_transform


def _dirichlet_partition(targets, num_clients, num_classes, alpha, min_client_size):
    targets = np.asarray(targets)
    min_size = 0
    while min_size < min_client_size:
        client_indices = [[] for _ in range(num_clients)]
        for class_id in range(num_classes):
            class_indices = np.where(targets == class_id)[0]
            np.random.shuffle(class_indices)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            for client_id, subset in enumerate(np.split(class_indices, split_points)):
                client_indices[client_id].extend(subset.tolist())
        min_size = min(len(indices) for indices in client_indices)
    return [np.array(np.random.permutation(indices), dtype=np.int64) for indices in client_indices]


def _iid_partition(num_samples, num_clients):
    shuffled_indices = np.random.permutation(num_samples)
    return [np.array(split, dtype=np.int64) for split in np.array_split(shuffled_indices, num_clients)]


def assign_clients_to_edges(num_clients, num_edges, assignment='round_robin'):
    if assignment not in {'round_robin', 'balanced'}:
        raise ValueError(f'Unknown assignment: {assignment}')
    return [client_id % num_edges for client_id in range(num_clients)]


def _edge_client_groups(num_clients, num_edges, assignment):
    edge_assignments = assign_clients_to_edges(num_clients, num_edges, assignment=assignment)
    edge_to_clients = {edge_id: [] for edge_id in range(num_edges)}
    for client_id, edge_id in enumerate(edge_assignments):
        edge_to_clients[edge_id].append(client_id)
    return edge_assignments, edge_to_clients


def _build_edge_class_preferences(num_edges, num_classes, overlap_size, shift, background_scale=0.1):
    overlap_size = max(1, min(overlap_size, num_classes))
    shift = max(1, shift)
    preferences = np.full((num_edges, num_classes), background_scale, dtype=np.float64)
    for edge_id in range(num_edges):
        start = (edge_id * shift) % num_classes
        selected = [(start + offset) % num_classes for offset in range(overlap_size)]
        preferences[edge_id, selected] = 1.0
    return preferences


def _split_indices_by_class(targets, indices, num_classes):
    indices = np.asarray(indices, dtype=np.int64)
    targets = np.asarray(targets)
    class_to_indices = {class_id: [] for class_id in range(num_classes)}
    for idx in indices:
        class_to_indices[int(targets[idx])].append(int(idx))
    return class_to_indices


def _hierarchical_split_edge_clients(edge_indices, targets, edge_to_clients, num_classes, client_alpha, min_client_size):
    targets = np.asarray(targets)
    edge_client_indices = {client_id: [] for client_ids in edge_to_clients.values() for client_id in client_ids}
    min_size = 0
    while min_size < min_client_size:
        edge_client_indices = {client_id: [] for client_ids in edge_to_clients.values() for client_id in client_ids}
        for edge_id, indices in edge_indices.items():
            client_ids = edge_to_clients[edge_id]
            if not client_ids:
                continue
            class_to_indices = _split_indices_by_class(targets, indices, num_classes)
            for class_id, class_indices in class_to_indices.items():
                if not class_indices:
                    continue
                class_indices = np.array(class_indices, dtype=np.int64)
                np.random.shuffle(class_indices)
                proportions = np.random.dirichlet(np.repeat(client_alpha, len(client_ids)))
                split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
                for client_id, subset in zip(client_ids, np.split(class_indices, split_points)):
                    edge_client_indices[client_id].extend(subset.tolist())
        min_size = min(len(indices) for indices in edge_client_indices.values())
    return [np.array(np.random.permutation(edge_client_indices[client_id]), dtype=np.int64) for client_id in sorted(edge_client_indices)]


def _edge_overlap_dirichlet_partition(targets, num_clients, num_edges, num_classes, edge_alpha, client_alpha, min_client_size, assignment, overlap_size, overlap_shift):
    targets = np.asarray(targets)
    edge_assignments, edge_to_clients = _edge_client_groups(num_clients, num_edges, assignment)
    edge_preferences = _build_edge_class_preferences(num_edges, num_classes, overlap_size, overlap_shift)

    min_size = 0
    client_indices = None
    while min_size < min_client_size:
        edge_indices = {edge_id: [] for edge_id in range(num_edges)}
        for class_id in range(num_classes):
            class_indices = np.where(targets == class_id)[0]
            np.random.shuffle(class_indices)
            alpha_vector = np.maximum(edge_alpha * edge_preferences[:, class_id], 1e-3)
            edge_proportions = np.random.dirichlet(alpha_vector)
            split_points = (np.cumsum(edge_proportions) * len(class_indices)).astype(int)[:-1]
            for edge_id, subset in enumerate(np.split(class_indices, split_points)):
                edge_indices[edge_id].extend(subset.tolist())

        client_indices = _hierarchical_split_edge_clients(
            edge_indices=edge_indices,
            targets=targets,
            edge_to_clients=edge_to_clients,
            num_classes=num_classes,
            client_alpha=client_alpha,
            min_client_size=min_client_size,
        )
        min_size = min(len(indices) for indices in client_indices)

    return client_indices, edge_assignments


class ImgDataset(Dataset):
    def __init__(self, data_name, train=True, transform=None):
        self.transform = transform
        self.data_name = data_name
        self.data, self.targets = get_dataset_by_name(data_name, train, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = _prepare_image(image, self.data_name)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class FedSemiDataset(Dataset):
    def __init__(self, labeled_ratio, train=True, transform=None, data_name='cifar10', accept_classes=None):
        self.transform = transform
        self.data_name = data_name
        self.weak_transform, self.strong_transform = _build_augmentations(data_name)

        self.data, self.targets = get_dataset_by_name(data_name, train, transform)
        n_classes = get_n_classes(data_name)

        if accept_classes is not None:
            assert all(0 <= c < n_classes for c in accept_classes)
            targets = np.array(self.targets)
            mask = np.zeros_like(targets, dtype=bool)
            for c in accept_classes:
                mask = mask | (targets == c)
            self.data = self.data[mask]
            self.targets = np.array(self.targets)[mask]

        self.labeled_radio = torch.zeros(len(self.data), dtype=torch.bool)
        labeled_indices = torch.randperm(len(self.data))[:int(len(self.data) * labeled_ratio)]
        if train:
            self.labeled_radio[labeled_indices] = True
        else:
            self.labeled_radio[:] = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        is_labeled = self.labeled_radio[index]
        img = _prepare_image(img, self.data_name)

        if self.transform is not None:
            img = self.transform(img)
        strong_img = self.strong_transform(img)
        weak_img = self.weak_transform(img)

        return {'img': img, 'label': label, 'is_labeled': is_labeled, 'weak_img': weak_img, 'strong_img': strong_img}


class SampledEdgeBatchDataset(Dataset):
    def __init__(self, client_entries, batch_size_per_client):
        self.records = []
        for client_entry in client_entries:
            indices = client_entry['indices']
            local_size = len(indices)
            if local_size == 0:
                continue
            replace = local_size < batch_size_per_client
            sample_size = batch_size_per_client if replace else min(batch_size_per_client, local_size)
            sampled_local_positions = np.random.choice(local_size, size=sample_size, replace=replace)
            for local_pos in sampled_local_positions:
                self.records.append((client_entry, int(local_pos)))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        client_entry, local_pos = self.records[index]
        real_index = client_entry['indices'][local_pos]
        img = client_entry['data'][real_index]
        label = int(client_entry['targets'][real_index])
        is_labeled = bool(client_entry['labeled_mask'][local_pos])
        img = _prepare_image(img, client_entry['data_name'])
        if client_entry['transform'] is not None:
            img = client_entry['transform'](img)
        weak_img = client_entry['weak_transform'](img)
        strong_img = client_entry['strong_transform'](img)
        return {
            'img': img,
            'label': label,
            'is_labeled': is_labeled,
            'weak_img': weak_img,
            'strong_img': strong_img,
        }


def build_sampled_edge_dataset(client_entries, batch_size_per_client):
    return SampledEdgeBatchDataset(client_entries, batch_size_per_client)


def _build_client_entry(data, targets, indices, labeled_ratio, train, transform, data_name, edge_id, client_id):
    labeled_mask = torch.zeros(len(indices), dtype=torch.bool)
    if train:
        labeled_count = int(len(indices) * labeled_ratio)
        if labeled_count > 0:
            labeled_indices = torch.randperm(len(indices))[:labeled_count]
            labeled_mask[labeled_indices] = True
    else:
        labeled_mask[:] = True

    weak_transform, strong_transform = _build_augmentations(data_name)
    return {
        'client_id': f'Client{client_id}',
        'edge_id': edge_id,
        'data': data,
        'targets': np.array(targets),
        'indices': np.array(indices, dtype=np.int64),
        'labeled_mask': labeled_mask,
        'transform': transform,
        'weak_transform': weak_transform,
        'strong_transform': strong_transform,
        'data_name': data_name,
        'num_samples': len(indices),
    }


def build_client_registries(
    data_name,
    num_clients,
    num_edges,
    labeled_ratio=0.1,
    train=True,
    transform=None,
    distributed='nonIID',
    partition_mode='client_dirichlet',
    partition_alpha=0.5,
    min_client_size=16,
    assignment='round_robin',
    edge_dirichlet_alpha=0.3,
    client_dirichlet_alpha=1.0,
    edge_overlap_size=5,
    edge_overlap_shift=1,
):
    data, targets = get_dataset_by_name(data_name, train=train, transform=None)
    num_classes = get_n_classes(data_name)

    if distributed == 'IID':
        client_indices = _iid_partition(len(data), num_clients)
        edge_assignments = assign_clients_to_edges(num_clients, num_edges, assignment=assignment)
    elif distributed == 'nonIID':
        if partition_mode == 'client_dirichlet':
            client_indices = _dirichlet_partition(targets, num_clients, num_classes, partition_alpha, min_client_size)
            edge_assignments = assign_clients_to_edges(num_clients, num_edges, assignment=assignment)
        elif partition_mode == 'edge_overlap_dirichlet':
            client_indices, edge_assignments = _edge_overlap_dirichlet_partition(
                targets=targets,
                num_clients=num_clients,
                num_edges=num_edges,
                num_classes=num_classes,
                edge_alpha=edge_dirichlet_alpha,
                client_alpha=client_dirichlet_alpha,
                min_client_size=min_client_size,
                assignment=assignment,
                overlap_size=edge_overlap_size,
                overlap_shift=edge_overlap_shift,
            )
        else:
            raise ValueError(f'Unknown partition_mode: {partition_mode}')
    else:
        raise ValueError(f'Unknown distributed setting: {distributed}')

    client_entries = []
    for client_id, indices in enumerate(client_indices):
        client_entries.append(
            _build_client_entry(
                data=data,
                targets=targets,
                indices=indices,
                labeled_ratio=labeled_ratio,
                train=train,
                transform=transform,
                data_name=data_name,
                edge_id=edge_assignments[client_id],
                client_id=client_id,
            )
        )
    return client_entries


def get_FedDataset_list(data_name, class_split_num, class_split_edge_num, class_split_edge_id, labeled_ratio=0.1, train=True, transform=None, distributed='nonIID'):
    if distributed == 'nonIID':
        accept_classes_label = split_classes(get_n_classes(data_name), class_split_num)
        dataset_list = []
        for accept_classes in accept_classes_label:
            dataset = FedSemiDataset(
                labeled_ratio=labeled_ratio,
                train=train,
                transform=transform,
                accept_classes=accept_classes,
                data_name=data_name,
            )
            dataset_list.append(dataset)
    elif distributed == 'IID':
        dataset_list = []
        for _ in range(class_split_num):
            dataset = FedSemiDataset(
                labeled_ratio=labeled_ratio,
                train=train,
                transform=transform,
                accept_classes=None,
                data_name=data_name,
            )
            dataset_list.append(dataset)
    else:
        raise ValueError(f'Unknown distributed: {distributed}')

    return dataset_list
