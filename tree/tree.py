import random
import math
from typing import Dict, List, Union

import torch
import torch.optim as optim
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader

from datasets.SemiDatasets import (
    MutableEdgeBatchSampler,
    build_base_transform,
    build_client_registries,
    build_reusable_edge_batch_dataset,
    make_train_val_indices,
)
from models.GetModel import get_model
from models.classifier import ENCODER_FEATURE_DIMS


def resolve_aggregation_mode(config):
    mode = str(OmegaConf.select(config, 'models.aggregation_mode', default='auto')).lower()
    if mode == 'auto':
        return 'state_dict' if str(config.datasets.name).lower() == 'harbox' else 'parameters'
    if mode not in {'parameters', 'state_dict'}:
        raise ValueError(f"Unknown models.aggregation_mode: {mode}")
    return mode


def aggregation_tensors(model, mode):
    if mode == 'state_dict':
        return list(model.state_dict().values())
    if mode == 'parameters':
        return list(model.parameters())
    raise ValueError(f'Unknown aggregation mode: {mode}')


def clone_aggregation_tensors(model, mode):
    return [tensor.detach().clone() for tensor in aggregation_tensors(model, mode)]


class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.level = 0
        self.model = None
        self.optimizer = None
        self.aggregation_mode = 'parameters'

    def add_child(self, child_node):
        if self.level >= 1:
            raise ValueError('Tree only keeps cloud and edge levels in the scalable setting')
        child_node.level = self.level + 1
        self.children.append(child_node)
        return child_node

    def run_model(self, *args, **kwargs):
        if self.model is None:
            raise ValueError('Model is None, call self.init_model() first')
        if self.level == 0:
            with torch.no_grad():
                return self.model(*args, **kwargs)
        return self.model(*args, **kwargs)

    def init_model(self, config):
        print(f'Initialize model for node {self.name}')
        self.model = get_model(self.level, config)
        self.aggregation_mode = resolve_aggregation_mode(config)

    def set_optimizer(self, optimizer):
        assert self.level == 1, 'Only edge nodes can own optimizers'
        if self.optimizer is None:
            self.optimizer = optimizer
        else:
            raise ValueError('Optimizer is already initialized')

    def optimizer_step(self, loss):
        if self.optimizer is None:
            raise ValueError('Optimizer is None')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_parameters(self):
        assert self.level == 1, 'Only edge nodes upload parameters'
        if self.model is None:
            raise ValueError('Model is None')
        return clone_aggregation_tensors(self.model, self.aggregation_mode)

    def set_parameters(self, parameters):
        assert self.level == 1, 'Only edge nodes receive personalized parameters'
        if self.model is None:
            raise ValueError('Model is None')
        with torch.no_grad():
            for tensor, new_tensor in zip(aggregation_tensors(self.model, self.aggregation_mode), parameters):
                tensor.copy_(new_tensor.to(device=tensor.device, dtype=tensor.dtype))


class Tree:
    def __init__(self, root_value):
        self.root = TreeNode(root_value)
        self.client_registry: Dict[str, Dict] = {}
        self.edge_to_clients: Dict[str, List[str]] = {}
        self.client_network_profile: Dict[str, Dict] = {}
        self.edge_network_profile: Dict[str, Dict] = {}
        self.edge_batch_loaders: Dict[str, DataLoader] = {}
        self.edge_batch_samplers: Dict[str, MutableEdgeBatchSampler] = {}
        self.config = None

    def find_node(self, name, node=None):
        if node is None:
            node = self.root
        if node.name == name:
            return node
        for child in node.children:
            result = self.find_node(name, child)
            if result:
                return result
        return None

    def add_node(self, parent_value, child_value):
        parent = self.find_node(parent_value)
        if parent is None:
            raise ValueError(f'Parent node {parent_value} is not found')
        new_node = TreeNode(child_value)
        return parent.add_child(new_node)

    def init_node_model(self, config, node=None):
        if node is None:
            node = self.root
        if node.model is None:
            node.init_model(config)
        for child in node.children:
            self.init_node_model(config, child)

    def move_to_device(self, device, node=None):
        if node is None:
            node = self.root
        if node.model is None:
            raise ValueError('Initialize models before moving them to a device')
        node.model.to(device)
        for child in node.children:
            self.move_to_device(device, child)

    def init_client_registry(self, config):
        print('Initialize logical clients'.center(80, '*'))
        self.config = config
        transform = build_base_transform(
            config.datasets.name,
            image_size=int(config.network.client_image_size),
            normalize=OmegaConf.select(config, 'datasets.normalize', default='imagenet'),
        )
        topology = config.topology
        val_ratio = float(OmegaConf.select(config, 'datasets.val_ratio', default=0.0))
        seed = int(OmegaConf.select(config, 'datasets.seed', default=0))
        train_indices = None
        if val_ratio > 0:
            train_indices, val_indices = make_train_val_indices(config.datasets.name, val_ratio=val_ratio, seed=seed)
            print(f'Train/val split: {len(train_indices)} train samples, {len(val_indices)} validation samples'.center(80, '*'))
        client_entries = build_client_registries(
            data_name=config.datasets.name,
            num_clients=topology.num_clients,
            num_edges=topology.num_edges,
            labeled_ratio=config.datasets.labeled_ratio,
            train=True,
            transform=transform,
            distributed=config.datasets.distributed,
            partition_mode=config.datasets.partition_mode,
            partition_alpha=config.datasets.partition_alpha,
            min_client_size=config.datasets.min_client_size,
            assignment=topology.assignment,
            edge_dirichlet_alpha=config.datasets.edge_dirichlet_alpha,
            client_dirichlet_alpha=config.datasets.client_dirichlet_alpha,
            edge_overlap_size=config.datasets.edge_overlap_size,
            edge_overlap_shift=config.datasets.edge_overlap_shift,
            label_sampling=OmegaConf.select(config, 'datasets.label_sampling', default='random'),
            min_labeled_per_client=OmegaConf.select(config, 'datasets.min_labeled_per_client', default=0),
            include_indices=train_indices,
        )
        self.client_registry = {}
        self.edge_to_clients = {f'Edge{edge_id}': [] for edge_id in range(topology.num_edges)}
        for entry in client_entries:
            client_id = entry['client_id']
            edge_name = f"Edge{entry['edge_id']}"
            entry['edge_name'] = edge_name
            self.client_registry[client_id] = entry
            self.edge_to_clients[edge_name].append(client_id)
        self.init_network_profiles(config)
        self.init_edge_batch_loaders(config)
        print(f"Registered {len(self.client_registry)} clients across {len(self.edge_to_clients)} edges".center(80, '*'))

    def init_edge_batch_loaders(self, config):
        num_workers = int(config.datasets.edge_num_workers)
        prefetch_factor = int(config.datasets.edge_prefetch_factor) if num_workers > 0 else None
        persistent_workers = num_workers > 0
        labeled_fraction = float(OmegaConf.select(config, 'datasets.balanced_labeled_batch_fraction', default=0.0))
        class_sampling_weights = OmegaConf.select(config, 'datasets.class_sampling_weights', default=None)
        self.edge_batch_loaders = {}
        self.edge_batch_samplers = {}
        for edge_name, client_ids in self.edge_to_clients.items():
            client_entries = [self.client_registry[client_id] for client_id in client_ids]
            dataset = build_reusable_edge_batch_dataset(client_entries)
            sampler = MutableEdgeBatchSampler(
                dataset.client_to_indices,
                config.datasets.batch_size,
                client_to_labeled_indices_by_class=dataset.client_to_labeled_indices_by_class,
                labeled_fraction=labeled_fraction,
                class_sampling_weights=class_sampling_weights,
            )
            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=config.datasets.pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
            )
            self.edge_batch_samplers[edge_name] = sampler
            self.edge_batch_loaders[edge_name] = dataloader

    @staticmethod
    def _edge_index(edge_name):
        return int(edge_name.replace('Edge', ''))

    @staticmethod
    def _sequence_value(values, edge_name, default):
        if values is None:
            return default
        if isinstance(values, DictConfig):
            return float(values.get(edge_name, default))
        edge_index = Tree._edge_index(edge_name)
        if edge_index < len(values):
            return float(values[edge_index])
        return default

    @staticmethod
    def _sample_distribution(distribution, default):
        if distribution is None:
            return float(default)
        dist_type = str(OmegaConf.select(distribution, 'type', default='uniform')).lower()
        if dist_type == 'uniform':
            low = float(OmegaConf.select(distribution, 'min_mbps', default=default))
            high = float(OmegaConf.select(distribution, 'max_mbps', default=default))
            return random.uniform(min(low, high), max(low, high))
        if dist_type == 'lognormal':
            median = max(float(OmegaConf.select(distribution, 'median_mbps', default=default)), 1e-6)
            sigma = max(float(OmegaConf.select(distribution, 'sigma', default=0.35)), 1e-6)
            low = float(OmegaConf.select(distribution, 'min_mbps', default=1.0))
            high = float(OmegaConf.select(distribution, 'max_mbps', default=default * 3))
            sampled = random.lognormvariate(math.log(median), sigma)
            return min(max(sampled, low), high)
        return float(default)

    @staticmethod
    def _clamp_probability(value):
        return min(max(float(value), 0.0), 1.0)

    def init_network_profiles(self, config):
        bandwidth_mode = str(OmegaConf.select(config, 'network.bandwidth.mode', default='homogeneous')).lower()
        default_client_bw = float(config.network.client_uplink_bandwidth_mbps)
        default_edge_bw = float(config.network.edge_uplink_bandwidth_mbps)
        default_downlink_bw = float(config.network.cloud_downlink_bandwidth_mbps)
        client_by_edge = OmegaConf.select(config, 'network.bandwidth.client_uplink_by_edge_mbps')
        edge_by_edge = OmegaConf.select(config, 'network.bandwidth.edge_uplink_by_edge_mbps')
        downlink_by_edge = OmegaConf.select(config, 'network.bandwidth.cloud_downlink_by_edge_mbps')
        client_dist = OmegaConf.select(config, 'network.bandwidth.client_distribution')
        edge_dist = OmegaConf.select(config, 'network.bandwidth.edge_distribution')
        downlink_dist = OmegaConf.select(config, 'network.bandwidth.cloud_downlink_distribution')
        drop_by_edge = OmegaConf.select(config, 'network.mobility.edge_drop_rates')
        mobility_mode = str(OmegaConf.select(config, 'network.mobility.mode', default='static')).lower()

        self.edge_network_profile = {}
        for edge_name in self.edge_to_clients:
            if bandwidth_mode == 'edge_profile':
                edge_uplink = self._sequence_value(edge_by_edge, edge_name, default_edge_bw)
                cloud_downlink = self._sequence_value(downlink_by_edge, edge_name, default_downlink_bw)
            elif bandwidth_mode == 'sampled_distribution':
                edge_uplink = self._sample_distribution(edge_dist, default_edge_bw)
                cloud_downlink = self._sample_distribution(downlink_dist, default_downlink_bw)
            else:
                edge_uplink = default_edge_bw
                cloud_downlink = default_downlink_bw
            if mobility_mode in {'edge_profile', 'mobility'}:
                drop_rate = self._sequence_value(drop_by_edge, edge_name, config.topology.client_drop_rate)
            else:
                drop_rate = config.topology.client_drop_rate
            self.edge_network_profile[edge_name] = {
                'edge_uplink_mbps': max(edge_uplink, 1e-6),
                'cloud_downlink_mbps': max(cloud_downlink, 1e-6),
                'drop_rate': self._clamp_probability(drop_rate),
            }

        self.client_network_profile = {}
        for client_id, entry in self.client_registry.items():
            edge_name = entry['edge_name']
            if bandwidth_mode == 'edge_profile':
                client_uplink = self._sequence_value(client_by_edge, edge_name, default_client_bw)
            elif bandwidth_mode == 'sampled_distribution':
                client_uplink = self._sample_distribution(client_dist, default_client_bw)
            else:
                client_uplink = default_client_bw
            labeled_ratio = float(entry['labeled_mask'].float().mean().item()) if len(entry['labeled_mask']) > 0 else 0.0
            self.client_network_profile[client_id] = {
                'client_uplink_mbps': max(client_uplink, 1e-6),
                'drop_rate': self.edge_network_profile[edge_name]['drop_rate'],
                'labeled_ratio': labeled_ratio,
                'num_samples': int(entry['num_samples']),
            }

    def get_edge_batch(self, client_ids):
        if not client_ids:
            return None
        edge_name = self.client_registry[client_ids[0]]['edge_name']
        sampler = self.edge_batch_samplers.get(edge_name)
        dataloader = self.edge_batch_loaders.get(edge_name)
        if sampler is None or dataloader is None:
            raise ValueError(f'Edge batch loader for {edge_name} is not initialized')
        sampler.set_clients(client_ids)
        if len(sampler) == 0:
            return None
        return next(iter(dataloader))

    def _estimate_client_upload_mb(self):
        views = int(self.config.network.client_view_count)
        bytes_per_value = int(self.config.network.bytes_per_value)
        batch_size = int(self.config.datasets.batch_size)
        encoder_name = str(OmegaConf.select(self.config, 'models.encoder_name', default=''))
        feature_dim = ENCODER_FEATURE_DIMS.get(encoder_name, OmegaConf.select(self.config, 'network.client_feature_dim'))
        if feature_dim is not None:
            total_bytes = batch_size * int(feature_dim) * bytes_per_value * views
        else:
            channels = int(self.config.network.client_input_channels)
            img_size = int(self.config.network.client_image_size)
            total_bytes = batch_size * channels * img_size * img_size * bytes_per_value * views
        return total_bytes / (1024 ** 2)

    def _estimate_edge_sync_mb(self):
        for edge in self.root.children:
            if edge.model is not None:
                total_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in aggregation_tensors(edge.model, edge.aggregation_mode)
                )
                return total_bytes / (1024 ** 2)
        return 0.0

    def get_client_uplink_bandwidth(self, client_id):
        return self.client_network_profile.get(client_id, {}).get(
            'client_uplink_mbps',
            float(self.config.network.client_uplink_bandwidth_mbps),
        )

    def get_edge_uplink_bandwidth(self, edge_name):
        return self.edge_network_profile.get(edge_name, {}).get(
            'edge_uplink_mbps',
            float(self.config.network.edge_uplink_bandwidth_mbps),
        )

    def get_cloud_downlink_bandwidth(self, edge_name):
        return self.edge_network_profile.get(edge_name, {}).get(
            'cloud_downlink_mbps',
            float(self.config.network.cloud_downlink_bandwidth_mbps),
        )

    def get_client_drop_probability(self, client_id, config):
        profile = self.client_network_profile.get(client_id, {})
        base_drop = profile.get('drop_rate', float(config.topology.client_drop_rate))
        mobility_mode = str(OmegaConf.select(config, 'network.mobility.mode', default='static')).lower()
        if mobility_mode == 'mobility':
            jitter = float(OmegaConf.select(config, 'network.mobility.drop_jitter', default=0.0))
            handoff_probability = float(OmegaConf.select(config, 'network.mobility.handoff_probability', default=0.0))
            handoff_boost = float(OmegaConf.select(config, 'network.mobility.handoff_drop_boost', default=0.0))
            base_drop += random.uniform(-jitter, jitter)
            if random.random() < handoff_probability:
                base_drop += handoff_boost
        return self._clamp_probability(base_drop)

    def _selected_edges(self, sampled):
        return [edge_name for edge_name, client_ids in sampled.items() if client_ids]

    def _estimated_round_comm_mb(self, sampled, config, include_model_sync=None):
        selected_edges = self._selected_edges(sampled)
        client_upload_mb = sum(len(client_ids) for client_ids in sampled.values()) * self._estimate_client_upload_mb()
        if include_model_sync is None:
            include_model_sync = bool(OmegaConf.select(config, 'network.budget.include_model_sync', default=True))
        if not include_model_sync:
            return client_upload_mb
        sync_mb = self._estimate_edge_sync_mb()
        edge_upload_mb = len(selected_edges) * sync_mb
        cloud_downlink_mb = len(selected_edges) * sync_mb
        return client_upload_mb + edge_upload_mb + cloud_downlink_mb

    def _client_utility(self, client_id, config):
        profile = self.client_network_profile[client_id]
        max_samples = max(entry['num_samples'] for entry in self.client_registry.values())
        sample_score = profile['num_samples'] / max(max_samples, 1)
        bandwidth_score = profile['client_uplink_mbps'] / max(float(config.network.client_uplink_bandwidth_mbps), 1e-6)
        availability_score = 1.0 - profile['drop_rate']
        label_weight = float(OmegaConf.select(config, 'network.resource_sampling.label_ratio_weight', default=0.35))
        data_weight = float(OmegaConf.select(config, 'network.resource_sampling.data_size_weight', default=0.25))
        bandwidth_weight = float(OmegaConf.select(config, 'network.resource_sampling.bandwidth_weight', default=0.25))
        availability_weight = float(OmegaConf.select(config, 'network.resource_sampling.availability_weight', default=0.15))
        latency_cost_weight = float(OmegaConf.select(config, 'network.resource_sampling.latency_cost_weight', default=1.0))
        quality = (
            label_weight * profile['labeled_ratio']
            + data_weight * sample_score
            + bandwidth_weight * bandwidth_score
            + availability_weight * availability_score
        )
        upload_latency = (self._estimate_client_upload_mb() * 8.0) / max(profile['client_uplink_mbps'], 1e-6)
        return quality / max(upload_latency ** latency_cost_weight, 1e-8)

    def _apply_network_constraints(self, sampled, config, include_model_sync=None):
        max_parallel = int(config.network.max_parallel_uploads_per_edge)
        if max_parallel > 0:
            for edge_name in sampled:
                if len(sampled[edge_name]) > max_parallel:
                    sampled[edge_name] = sorted(
                        sampled[edge_name],
                        key=lambda client_id: self._client_utility(client_id, config),
                        reverse=True,
                    )[:max_parallel]

        budget_mb = float(config.network.round_comm_budget_mb)
        if budget_mb > 0:
            while self._estimated_round_comm_mb(sampled, config, include_model_sync=include_model_sync) > budget_mb:
                removable = [
                    (self._client_utility(client_id, config), edge_name, client_id)
                    for edge_name, client_ids in sampled.items()
                    for client_id in client_ids
                ]
                if len(removable) <= 1:
                    break
                _, edge_name, client_id = min(removable, key=lambda item: item[0])
                sampled[edge_name].remove(client_id)
        return {edge_name: client_ids for edge_name, client_ids in sampled.items() if client_ids}

    def _sample_resource_aware_clients(self, config, include_model_sync=None):
        total_clients = min(config.topology.clients_per_round, len(self.client_registry))
        edge_names = list(self.edge_to_clients.keys())
        edges_per_round = min(config.topology.edges_per_round, len(edge_names), total_clients)
        exploration_rate = float(OmegaConf.select(config, 'network.resource_sampling.exploration_rate', default=0.05))

        edge_scores = []
        for edge_name in edge_names:
            client_ids = self.edge_to_clients[edge_name]
            if not client_ids:
                continue
            mean_utility = sum(self._client_utility(client_id, config) for client_id in client_ids) / len(client_ids)
            edge_scores.append((mean_utility, edge_name))
        edge_scores.sort(reverse=True)
        selected_edges = [edge_name for _, edge_name in edge_scores[:edges_per_round]]

        sampled = {edge_name: [] for edge_name in edge_names}
        candidates = []
        for edge_name in selected_edges:
            for client_id in self.edge_to_clients[edge_name]:
                score = self._client_utility(client_id, config)
                if random.random() < exploration_rate:
                    score *= random.uniform(0.5, 1.5)
                candidates.append((score, edge_name, client_id))
        candidates.sort(reverse=True)

        per_edge_limit = max(1, int(config.network.max_parallel_uploads_per_edge))
        for _, edge_name, client_id in candidates:
            if sum(len(client_ids) for client_ids in sampled.values()) >= total_clients:
                break
            if len(sampled[edge_name]) >= per_edge_limit:
                continue
            sampled[edge_name].append(client_id)
        return self._apply_network_constraints(sampled, config, include_model_sync=include_model_sync)

    def sample_clients(self, config, include_model_sync=None):
        total_clients = min(config.topology.clients_per_round, len(self.client_registry))
        edge_names = list(self.edge_to_clients.keys())
        if not edge_names or total_clients == 0:
            return {}

        sample_mode = config.topology.client_sample_mode.lower()
        sampled = {edge_name: [] for edge_name in edge_names}
        if sample_mode == 'resource_aware':
            sampled = self._sample_resource_aware_clients(config, include_model_sync=include_model_sync)
        elif sample_mode in {'balanced', 'budget_aware'}:
            edges_per_round = min(config.topology.edges_per_round, len(edge_names), total_clients)
            selected_edges = edge_names if edges_per_round == len(edge_names) else random.sample(edge_names, edges_per_round)
            base_quota = total_clients // len(selected_edges)
            remainder = total_clients % len(selected_edges)

            for edge_index, edge_name in enumerate(selected_edges):
                population = self.edge_to_clients[edge_name]
                quota = base_quota + (1 if edge_index < remainder else 0)
                quota = min(quota, len(population))
                if quota > 0:
                    sampled[edge_name] = random.sample(population, quota)

            selected_total = sum(len(client_ids) for client_ids in sampled.values())
            remaining = max(0, total_clients - selected_total)
            if remaining > 0:
                remaining_pool = []
                for edge_name in selected_edges:
                    chosen = set(sampled[edge_name])
                    remaining_pool.extend([client_id for client_id in self.edge_to_clients[edge_name] if client_id not in chosen])
                if remaining_pool:
                    extra_clients = random.sample(remaining_pool, min(remaining, len(remaining_pool)))
                    for client_id in extra_clients:
                        edge_name = self.client_registry[client_id]['edge_name']
                        sampled[edge_name].append(client_id)
            if sample_mode == 'budget_aware':
                sampled = self._apply_network_constraints(sampled, config, include_model_sync=include_model_sync)
        elif sample_mode == 'uniform':
            all_client_ids = list(self.client_registry.keys())
            selected_clients = random.sample(all_client_ids, total_clients)
            for client_id in selected_clients:
                edge_name = self.client_registry[client_id]['edge_name']
                sampled[edge_name].append(client_id)
            sampled = self._apply_network_constraints(sampled, config, include_model_sync=include_model_sync)
        else:
            raise ValueError(f'Unknown client_sample_mode: {config.topology.client_sample_mode}')

        return {edge_name: client_ids for edge_name, client_ids in sampled.items() if client_ids}

    def init_edge_optimizer(self, config):
        print('Initialize edge optimizers'.center(80, '*'))
        for edge in self.root.children:
            if edge.model is None:
                raise ValueError('Initialize models before optimizers')
            if config.train.optimizer == 'AdamW':
                optimizer = optim.AdamW(edge.model.parameters(), lr=config.train.learning_rate)
                edge.set_optimizer(optimizer)
            else:
                raise ValueError(f"Unsupported optimizer: {config.train.optimizer}")
        print('Edge optimizers are ready'.center(80, '*'))


def create_tree(config: Union[DictConfig, ListConfig]) -> Tree:
    cloud_tree = Tree('Cloud')
    for edge_id in range(int(config.topology.num_edges)):
        cloud_tree.add_node('Cloud', f'Edge{edge_id}')
    print('Initialize node models'.center(80, '*'))
    cloud_tree.init_node_model(config)
    cloud_tree.init_client_registry(config)
    cloud_tree.init_edge_optimizer(config)
    print('Training starts'.center(80, '*'))
    return cloud_tree


if __name__ == '__main__':
    config = OmegaConf.load('config/formated_config.yaml')
    tree = create_tree(config)
    print({edge: len(clients) for edge, clients in tree.edge_to_clients.items()})
