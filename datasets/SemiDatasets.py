import os
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Sampler
from torchvision import datasets

from datasets import split_classes


_HARBOX_CACHE = {}
_HARBOX_ACTIVITY_LABELS = {
    'Call': 0,
    'Hop': 1,
    'Walk': 2,
    'Wave': 3,
    'typing': 4,
}
_NETWORK_CACHE = {}
_NETWORK_DATASETS = {'network', 'nslkdd', 'unsw_nb15'}
_NSLKDD_ATTACK_GROUPS = {
    'normal': 0,
    'normal.': 0,
    '0': 0,
    'dos': 1,
    'back': 1,
    'land': 1,
    'neptune': 1,
    'pod': 1,
    'smurf': 1,
    'teardrop': 1,
    'apache2': 1,
    'udpstorm': 1,
    'processtable': 1,
    'mailbomb': 1,
    'worm': 1,
    'probe': 2,
    'satan': 2,
    'ipsweep': 2,
    'nmap': 2,
    'portsweep': 2,
    'mscan': 2,
    'saint': 2,
    'r2l': 3,
    'guess_passwd': 3,
    'ftp_write': 3,
    'imap': 3,
    'phf': 3,
    'multihop': 3,
    'warezmaster': 3,
    'warezclient': 3,
    'spy': 3,
    'xlock': 3,
    'xsnoop': 3,
    'snmpguess': 3,
    'snmpgetattack': 3,
    'httptunnel': 3,
    'sendmail': 3,
    'named': 3,
    'u2r': 4,
    'buffer_overflow': 4,
    'loadmodule': 4,
    'rootkit': 4,
    'perl': 4,
    'sqlattack': 4,
    'xterm': 4,
    'ps': 4,
}


def get_n_classes(name):
    if name == 'cifar10':
        n_classes = 10
    elif name == 'cifar100':
        n_classes = 100
    elif name == 'fashionmnist':
        n_classes = 10
    elif name == 'svhn':
        n_classes = 10
    elif name == 'harbox':
        n_classes = 5
    elif name in {'network', 'nslkdd'}:
        n_classes = 5
    elif name == 'unsw_nb15':
        n_classes = 2
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
    elif name == 'harbox':
        data, targets, _ = load_harbox_dataset(train=train)
    elif name in _NETWORK_DATASETS:
        data, targets, _ = load_network_dataset(name=name, train=train)
    elif name == 'mnist':
        dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        data = dataset.data
        targets = dataset.targets
    else:
        raise NotImplementedError(f'Dataset {name} not implemented')
    return data, targets


def _first_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _array_from_keys(container, keys):
    for key in keys:
        if key in container:
            return container[key]
    return None


def _normalize_harbox_labels(labels):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    unique_labels = np.unique(labels)
    if unique_labels.min() == 1 and unique_labels.max() == len(unique_labels):
        labels = labels - 1
    return labels


def _standardize_features(features):
    features = np.asarray(features, dtype=np.float32)
    features = features.reshape(features.shape[0], -1)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-6)


def _normalize_network_labels(labels):
    labels = np.asarray(labels).reshape(-1)
    if labels.dtype.kind in {'U', 'S', 'O'}:
        normalized = []
        for label in labels:
            label_text = str(label).strip().lower()
            if label_text in _NSLKDD_ATTACK_GROUPS:
                normalized.append(_NSLKDD_ATTACK_GROUPS[label_text])
            elif label_text in {'benign', '1'}:
                normalized.append(0)
            else:
                normalized.append(1)
        return np.asarray(normalized, dtype=np.int64)
    labels = labels.astype(np.int64)
    unique_labels = np.unique(labels)
    if unique_labels.min() == 1 and unique_labels.max() <= 5:
        return labels - 1
    if unique_labels.min() >= 0 and unique_labels.max() <= 4:
        return labels
    if len(unique_labels) <= 2:
        return (labels != unique_labels.min()).astype(np.int64)
    return labels


def _split_network_arrays(features, labels, users, train):
    features = _standardize_features(features)
    labels = _normalize_network_labels(labels)
    if users is not None:
        users = np.asarray(users).reshape(-1)
        unique_users = np.array(sorted(np.unique(users).tolist()))
        default_train_users = min(max(1, int(0.8 * len(unique_users))), max(len(unique_users) - 1, 1))
        train_user_count = int(os.environ.get('NETWORK_TRAIN_USERS', default_train_users))
        train_user_count = min(max(train_user_count, 1), max(len(unique_users) - 1, 1))
        train_user_set = set(unique_users[:train_user_count].tolist())
        mask = np.array([user in train_user_set for user in users], dtype=bool)
        if not train:
            mask = ~mask
        return features[mask], labels[mask], users[mask]

    rng = np.random.default_rng(2026)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    split = int(0.8 * len(indices))
    selected = indices[:split] if train else indices[split:]
    return features[selected], labels[selected], None


def _load_network_npz(path, train, split_file=False):
    npz = np.load(path, allow_pickle=True)
    split_key = 'train' if train else 'test'
    split_features = _array_from_keys(npz, [f'{split_key}_x', f'{split_key}_X', f'X_{split_key}', f'x_{split_key}', f'{split_key}_data'])
    split_labels = _array_from_keys(npz, [f'{split_key}_y', f'{split_key}_Y', f'y_{split_key}', f'labels_{split_key}', f'{split_key}_labels'])
    if split_features is not None and split_labels is not None:
        split_users = _array_from_keys(npz, [f'{split_key}_users', f'{split_key}_user_ids', f'users_{split_key}', f'user_ids_{split_key}'])
        return _standardize_features(split_features), _normalize_network_labels(split_labels), split_users

    features = _array_from_keys(npz, ['x', 'X', 'data', 'features', 'samples'])
    labels = _array_from_keys(npz, ['y', 'Y', 'label', 'labels', 'targets'])
    users = _array_from_keys(npz, ['user', 'users', 'user_id', 'user_ids', 'host', 'hosts', 'node_id', 'node_ids'])
    if features is None or labels is None:
        raise ValueError(f'Network npz file {path} must contain feature and label arrays')
    if split_file:
        return _standardize_features(features), _normalize_network_labels(labels), users
    return _split_network_arrays(features, labels, users, train)


def _load_network_from_directory(root, train):
    split_name = 'train' if train else 'test'
    split_npz = _first_existing_path([
        root / f'{split_name}.npz',
        root / f'network_{split_name}.npz',
        root / f'nslkdd_{split_name}.npz',
        root / f'unsw_nb15_{split_name}.npz',
    ])
    if split_npz is not None:
        return _load_network_npz(split_npz, train=train, split_file=True)

    dataset_npz = _first_existing_path([
        root / 'dataset.npz',
        root / 'network.npz',
        root / 'nslkdd.npz',
        root / 'unsw_nb15.npz',
    ])
    if dataset_npz is not None:
        return _load_network_npz(dataset_npz, train=train)

    feature_file = _first_existing_path([root / 'X.npy', root / 'x.npy', root / 'features.npy', root / 'data.npy'])
    label_file = _first_existing_path([root / 'y.npy', root / 'Y.npy', root / 'labels.npy', root / 'targets.npy'])
    user_file = _first_existing_path([root / 'users.npy', root / 'user_ids.npy', root / 'hosts.npy', root / 'host_ids.npy'])
    if feature_file is None or label_file is None:
        raise FileNotFoundError(
            f'Network-flow files are not found under {root}. Expected dataset.npz, train/test npz, '
            'or X.npy plus y.npy. Optional users.npy can be used for user-level partition.'
        )
    features = np.load(feature_file, allow_pickle=True)
    labels = np.load(label_file, allow_pickle=True)
    users = np.load(user_file, allow_pickle=True) if user_file is not None else None
    return _split_network_arrays(features, labels, users, train)


def load_network_dataset(name='network', train=True):
    root_name = 'network' if name == 'network' else name
    root_candidates = []
    if os.environ.get('NETWORK_ROOT'):
        root_candidates.append(Path(os.environ['NETWORK_ROOT']))
    root_candidates.extend([
        Path(f'./data/{root_name}'),
        Path('./data/network'),
        Path('./data/nslkdd'),
        Path('./data/NSL-KDD'),
        Path('./data/unsw_nb15'),
        Path('./data/UNSW-NB15'),
    ])
    errors = []
    for root in root_candidates:
        if not root.exists():
            continue
        cache_key = (name, str(root.resolve()), bool(train))
        if cache_key in _NETWORK_CACHE:
            return _NETWORK_CACHE[cache_key]
        try:
            dataset = _load_network_from_directory(root, train=train)
        except FileNotFoundError as error:
            errors.append(str(error))
            continue
        _NETWORK_CACHE[cache_key] = dataset
        return dataset

    searched_paths = ', '.join(str(path) for path in root_candidates)
    detail = f' Last errors: {" | ".join(errors)}' if errors else ''
    raise FileNotFoundError(
        f'Network-flow data is not available. Searched: {searched_paths}. '
        'Put preprocessed dataset.npz or X.npy/y.npy under data/nslkdd, '
        'or set NETWORK_ROOT to the preprocessed network-flow directory.'
        f'{detail}'
    )


def _split_harbox_arrays(features, labels, users, train):
    labels = _normalize_harbox_labels(labels)
    features = _standardize_features(features)
    if users is not None:
        users = np.asarray(users).reshape(-1)
        unique_users = np.array(sorted(np.unique(users).tolist()))
        default_train_users = min(100, max(1, int(0.8 * len(unique_users))))
        train_user_count = int(os.environ.get('HARBOX_TRAIN_USERS', default_train_users))
        train_user_count = min(max(train_user_count, 1), max(len(unique_users) - 1, 1))
        train_user_set = set(unique_users[:train_user_count].tolist())
        mask = np.array([user in train_user_set for user in users], dtype=bool)
        if not train:
            mask = ~mask
        return features[mask], labels[mask], users[mask]

    rng = np.random.default_rng(2026)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    split = int(0.8 * len(indices))
    selected = indices[:split] if train else indices[split:]
    return features[selected], labels[selected], None


def _load_harbox_npz(path, train, split_file=False):
    npz = np.load(path, allow_pickle=True)
    split_key = 'train' if train else 'test'
    split_features = _array_from_keys(npz, [f'{split_key}_x', f'{split_key}_X', f'X_{split_key}', f'x_{split_key}', f'{split_key}_data'])
    split_labels = _array_from_keys(npz, [f'{split_key}_y', f'{split_key}_Y', f'y_{split_key}', f'labels_{split_key}', f'{split_key}_labels'])
    if split_features is not None and split_labels is not None:
        split_users = _array_from_keys(npz, [f'{split_key}_users', f'{split_key}_user_ids', f'users_{split_key}', f'user_ids_{split_key}'])
        return _standardize_features(split_features), _normalize_harbox_labels(split_labels), split_users

    features = _array_from_keys(npz, ['x', 'X', 'data', 'features', 'samples'])
    labels = _array_from_keys(npz, ['y', 'Y', 'label', 'labels', 'targets'])
    users = _array_from_keys(npz, ['user', 'users', 'user_id', 'user_ids', 'subject', 'subjects', 'node_id', 'node_ids'])
    if features is None or labels is None:
        raise ValueError(f'HARBox npz file {path} must contain feature and label arrays')
    if split_file:
        return _standardize_features(features), _normalize_harbox_labels(labels), users
    return _split_harbox_arrays(features, labels, users, train)


def _load_harbox_from_directory(root, train):
    if root.is_file() and root.suffix.lower() == '.zip':
        return _load_harbox_zip(root, train=train)

    split_name = 'train' if train else 'test'
    split_npz = _first_existing_path([
        root / f'{split_name}.npz',
        root / f'harbox_{split_name}.npz',
        root / f'HARBox_{split_name}.npz',
    ])
    if split_npz is not None:
        return _load_harbox_npz(split_npz, train=train, split_file=True)

    dataset_npz = _first_existing_path([
        root / 'harbox.npz',
        root / 'HARBox.npz',
        root / 'dataset.npz',
        root / 'data.npz',
    ])
    if dataset_npz is not None:
        return _load_harbox_npz(dataset_npz, train=train)

    feature_file = _first_existing_path([root / 'X.npy', root / 'x.npy', root / 'features.npy', root / 'data.npy'])
    label_file = _first_existing_path([root / 'y.npy', root / 'Y.npy', root / 'labels.npy', root / 'targets.npy'])
    user_file = _first_existing_path([root / 'users.npy', root / 'user_ids.npy', root / 'subjects.npy', root / 'subject_ids.npy'])
    if feature_file is None or label_file is None:
        raise FileNotFoundError(
            f'HARBox files are not found under {root}. Expected one of: '
            'data/large_scale_HARBox.zip, data/harbox/harbox.npz, or data/harbox/X.npy plus data/harbox/y.npy. '
            'You can also set HARBOX_ROOT to the official zip file or a preprocessed HARBox directory.'
        )
    features = np.load(feature_file, allow_pickle=True)
    labels = np.load(label_file, allow_pickle=True)
    users = np.load(user_file, allow_pickle=True) if user_file is not None else None
    return _split_harbox_arrays(features, labels, users, train)


def _harbox_windows_from_series(series, window_size=None, stride=None):
    window_size = int(os.environ.get('HARBOX_WINDOW_SIZE', window_size or 100))
    stride = int(os.environ.get('HARBOX_WINDOW_STRIDE', stride or 50))
    if series.shape[0] < window_size:
        return np.empty((0, window_size * 9), dtype=np.float32)
    windows = []
    for start in range(0, series.shape[0] - window_size + 1, stride):
        windows.append(series[start:start + window_size].reshape(-1))
    return np.asarray(windows, dtype=np.float32)


def _load_harbox_zip(zip_path, train):
    split_keyword = '_train.txt' if train else '_test.txt'
    features, labels, users = [], [], []
    if not zipfile.is_zipfile(zip_path):
        with open(zip_path, 'rb') as file:
            file_head = file.read(16)
        raise zipfile.BadZipFile(
            f'{zip_path} is not a valid zip file. First bytes: {file_head!r}. '
            'Please check whether the HARBox download is complete or whether the file is an HTML/download page.'
        )
    with zipfile.ZipFile(zip_path) as archive:
        entries = [
            entry for entry in archive.infolist()
            if not entry.is_dir()
            and entry.filename.endswith(split_keyword)
            and '__MACOSX' not in entry.filename
        ]
        if not entries and not train:
            entries = [
                entry for entry in archive.infolist()
                if not entry.is_dir()
                and entry.filename.endswith('_train.txt')
                and '__MACOSX' not in entry.filename
            ]

        for entry in entries:
            parts = entry.filename.replace('\\', '/').split('/')
            if len(parts) < 3:
                continue
            try:
                user_id = int(parts[-2])
            except ValueError:
                continue
            activity_name = parts[-1].split('_')[0]
            if activity_name not in _HARBOX_ACTIVITY_LABELS:
                continue
            with archive.open(entry) as file:
                series = np.loadtxt(file, dtype=np.float32)
            if series.ndim == 1:
                series = series.reshape(1, -1)
            if series.shape[1] >= 10:
                series = series[:, 1:10]
            elif series.shape[1] != 9:
                raise ValueError(f'Unexpected HARBox sensor dimension in {entry.filename}: {series.shape}')

            window_features = _harbox_windows_from_series(series)
            if window_features.size == 0:
                continue
            features.append(window_features)
            labels.extend([_HARBOX_ACTIVITY_LABELS[activity_name]] * len(window_features))
            users.extend([user_id] * len(window_features))

    if not features:
        raise FileNotFoundError(f'No HARBox {split_keyword} files found in {zip_path}')
    return _split_harbox_arrays(
        np.concatenate(features, axis=0),
        np.asarray(labels, dtype=np.int64),
        np.asarray(users, dtype=np.int64),
        train=train,
    )


def load_harbox_dataset(train=True):
    root_candidates = []
    if os.environ.get('HARBOX_ROOT'):
        root_candidates.append(Path(os.environ['HARBOX_ROOT']))
    root_candidates.extend([
        Path('./data/harbox'),
        Path('./data/HARBox'),
        Path('./data/HARBOX'),
        Path('./data/large_scale_HARBox.zip'),
    ])
    errors = []
    for root in root_candidates:
        if not root.exists():
            continue
        cache_key = (str(root.resolve()), bool(train))
        if cache_key in _HARBOX_CACHE:
            return _HARBOX_CACHE[cache_key]
        try:
            dataset = _load_harbox_from_directory(root, train=train)
        except FileNotFoundError as error:
            errors.append(str(error))
            continue
        _HARBOX_CACHE[cache_key] = dataset
        return dataset

    searched_paths = ', '.join(str(path) for path in root_candidates)
    detail = f' Last errors: {" | ".join(errors)}' if errors else ''
    raise FileNotFoundError(
        f'HARBox data is not available. Searched: {searched_paths}. '
        'Put the official large_scale_HARBox.zip under data/, or set HARBOX_ROOT to that zip file.'
        f'{detail}'
    )


def get_dataset_user_ids(name, train=True):
    if name == 'harbox':
        _, _, users = load_harbox_dataset(train=train)
        return users
    if name in _NETWORK_DATASETS:
        _, _, users = load_network_dataset(name=name, train=train)
        return users
    return None


def _prepare_image(image, data_name):
    if data_name == 'harbox' or data_name in _NETWORK_DATASETS:
        return torch.as_tensor(image, dtype=torch.float32).view(-1)
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if data_name == 'svhn':
        image = image.transpose(1, 2, 0)
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
        image = np.concatenate([image, image, image], axis=-1)
    return image


def build_base_transform(data_name, image_size=224, normalize='imagenet'):
    if data_name == 'harbox' or data_name in _NETWORK_DATASETS:
        return None
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


def _harbox_weak_augment(x):
    return x + 0.005 * torch.randn_like(x)


def _harbox_strong_augment(x):
    augmented = x.view(-1, 9).clone()
    scale = 1.0 + 0.05 * torch.randn(1, augmented.size(1), device=augmented.device, dtype=augmented.dtype)
    augmented = augmented * scale
    augmented = augmented + 0.02 * torch.randn_like(augmented)

    if augmented.size(0) >= 10:
        max_mask = max(1, augmented.size(0) // 10)
        mask_len = int(torch.randint(1, max_mask + 1, (1,), device=augmented.device).item())
        start = int(torch.randint(0, augmented.size(0) - mask_len + 1, (1,), device=augmented.device).item())
        augmented[start:start + mask_len] = 0.0

    return augmented.reshape(-1)


def _network_weak_augment(x):
    return x + 0.002 * torch.randn_like(x)


def _network_strong_augment(x):
    augmented = x + 0.01 * torch.randn_like(x)
    if augmented.numel() > 0:
        keep_mask = torch.rand_like(augmented) > 0.05
        augmented = augmented * keep_mask
    return augmented


def _build_augmentations(data_name='cifar10'):
    if data_name == 'harbox':
        weak_transform = transforms.Lambda(_harbox_weak_augment)
        strong_transform = transforms.Lambda(_harbox_strong_augment)
        return weak_transform, strong_transform

    if data_name in _NETWORK_DATASETS:
        weak_transform = transforms.Lambda(_network_weak_augment)
        strong_transform = transforms.Lambda(_network_strong_augment)
        return weak_transform, strong_transform

    if data_name == 'svhn':
        weak_transform = transforms.Compose([
            transforms.RandomRotation(degrees=5),
        ])
        strong_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
        return weak_transform, strong_transform

    if data_name in {'mnist', 'fashionmnist'}:
        weak_transform = transforms.Compose([
            transforms.RandomRotation(degrees=5),
        ])
        strong_transform = transforms.Compose([
            transforms.RandomRotation(degrees=12),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
        return weak_transform, strong_transform

    if data_name == 'cifar100':
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ])
        strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
        ])
        return weak_transform, strong_transform

    if data_name == 'cifar10':
        weak_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ])
        strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
        ])
        return weak_transform, strong_transform

    weak_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    strong_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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


def _user_partition(user_ids, num_clients):
    if user_ids is None:
        raise ValueError('User ids are required for user partition')
    user_ids = np.asarray(user_ids)
    unique_users = sorted(np.unique(user_ids).tolist())
    if num_clients > len(unique_users):
        raise ValueError(f'HARBox user partition has only {len(unique_users)} users, but num_clients={num_clients}')
    client_indices = [[] for _ in range(num_clients)]
    for group_index, user in enumerate(unique_users):
        client_id = group_index % num_clients
        client_indices[client_id].extend(np.where(user_ids == user)[0].tolist())
    return [np.array(np.random.permutation(indices), dtype=np.int64) for indices in client_indices]


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


class ReusableEdgeBatchDataset(Dataset):
    def __init__(self, client_entries):
        self.records = []
        self.client_to_indices = {}
        self.client_to_labeled_indices_by_class = {}
        for client_entry in client_entries:
            client_id = client_entry['client_id']
            self.client_to_indices[client_id] = []
            self.client_to_labeled_indices_by_class[client_id] = {}
            for local_pos in range(len(client_entry['indices'])):
                record_index = len(self.records)
                self.client_to_indices[client_id].append(record_index)
                if bool(client_entry['labeled_mask'][local_pos]):
                    real_index = client_entry['indices'][local_pos]
                    class_id = int(client_entry['targets'][real_index])
                    self.client_to_labeled_indices_by_class[client_id].setdefault(class_id, []).append(record_index)
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


class MutableEdgeBatchSampler(Sampler):
    def __init__(
        self,
        client_to_indices,
        batch_size_per_client,
        client_to_labeled_indices_by_class=None,
        labeled_fraction=0.0,
        class_sampling_weights=None,
    ):
        self.client_to_indices = client_to_indices
        self.batch_size_per_client = int(batch_size_per_client)
        self.client_to_labeled_indices_by_class = client_to_labeled_indices_by_class or {}
        self.labeled_fraction = float(labeled_fraction)
        self.class_sampling_weights = None
        if class_sampling_weights is not None:
            self.class_sampling_weights = np.asarray(list(class_sampling_weights), dtype=np.float64)
        self.client_ids = []

    def set_clients(self, client_ids):
        self.client_ids = [
            client_id for client_id in client_ids
            if client_id in self.client_to_indices and self.client_to_indices[client_id]
        ]

    def __iter__(self):
        batch_indices = []
        for client_id in self.client_ids:
            record_indices = self.client_to_indices[client_id]
            labeled_indices = self._sample_labeled_indices(client_id)
            remaining_size = max(self.batch_size_per_client - len(labeled_indices), 0)
            replace = len(record_indices) < remaining_size
            sampled_positions = np.random.choice(
                len(record_indices),
                size=remaining_size,
                replace=replace,
            ) if remaining_size > 0 else []
            batch_indices.extend(labeled_indices)
            batch_indices.extend(record_indices[int(pos)] for pos in sampled_positions)
        if batch_indices:
            yield batch_indices

    def __len__(self):
        return 1 if self.client_ids else 0

    def _sample_labeled_indices(self, client_id):
        if self.labeled_fraction <= 0:
            return []
        labeled_by_class = self.client_to_labeled_indices_by_class.get(client_id, {})
        available_classes = [class_id for class_id, indices in labeled_by_class.items() if indices]
        if not available_classes:
            return []

        sample_size = int(round(self.batch_size_per_client * self.labeled_fraction))
        sample_size = min(max(sample_size, 0), self.batch_size_per_client)
        if sample_size == 0:
            return []

        probs = None
        if self.class_sampling_weights is not None:
            weights = np.array([
                self.class_sampling_weights[class_id] if class_id < len(self.class_sampling_weights) else 1.0
                for class_id in available_classes
            ], dtype=np.float64)
            if weights.sum() > 0:
                probs = weights / weights.sum()

        sampled = []
        sampled_classes = np.random.choice(available_classes, size=sample_size, replace=True, p=probs)
        for class_id in sampled_classes:
            class_indices = labeled_by_class[int(class_id)]
            sampled.append(int(np.random.choice(class_indices)))
        return sampled


def build_reusable_edge_batch_dataset(client_entries):
    return ReusableEdgeBatchDataset(client_entries)


def _select_labeled_positions(targets, indices, labeled_count, mode):
    if labeled_count <= 0:
        return torch.empty(0, dtype=torch.long)
    if str(mode).lower() != 'stratified':
        return torch.randperm(len(indices))[:labeled_count]

    local_labels = np.asarray(targets)[indices]
    selected = []
    class_ids, class_counts = np.unique(local_labels, return_counts=True)
    for class_id in class_ids[np.argsort(class_counts)]:
        positions = np.where(local_labels == class_id)[0]
        if len(positions) == 0:
            continue
        selected.append(int(np.random.choice(positions)))
        if len(selected) >= labeled_count:
            break

    if len(selected) < labeled_count:
        remaining = np.setdiff1d(np.arange(len(indices)), np.array(selected, dtype=np.int64), assume_unique=False)
        if len(remaining) > 0:
            fill_count = min(labeled_count - len(selected), len(remaining))
            selected.extend(np.random.choice(remaining, size=fill_count, replace=False).astype(np.int64).tolist())
    return torch.tensor(selected[:labeled_count], dtype=torch.long)


def _build_client_entry(
    data,
    targets,
    indices,
    labeled_ratio,
    train,
    transform,
    data_name,
    edge_id,
    client_id,
    label_sampling='random',
    min_labeled_per_client=0,
):
    labeled_mask = torch.zeros(len(indices), dtype=torch.bool)
    if train:
        labeled_count = int(len(indices) * labeled_ratio)
        if labeled_ratio > 0:
            labeled_count = max(labeled_count, int(min_labeled_per_client))
        labeled_count = min(labeled_count, len(indices))
        if labeled_count > 0:
            labeled_indices = _select_labeled_positions(targets, indices, labeled_count, label_sampling)
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
    label_sampling='random',
    min_labeled_per_client=0,
):
    data, targets = get_dataset_by_name(data_name, train=train, transform=None)
    num_classes = get_n_classes(data_name)

    if data_name in {'harbox', 'network', 'nslkdd', 'unsw_nb15'} and partition_mode == 'user':
        user_ids = get_dataset_user_ids(data_name, train=train)
        if user_ids is None:
            client_indices = _iid_partition(len(data), num_clients)
        else:
            client_indices = _user_partition(user_ids, num_clients)
        edge_assignments = assign_clients_to_edges(num_clients, num_edges, assignment=assignment)
    elif distributed == 'IID':
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
                label_sampling=label_sampling,
                min_labeled_per_client=min_labeled_per_client,
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
