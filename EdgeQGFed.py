import datetime
import argparse
import os
import random
import time
from collections import OrderedDict
from contextlib import ExitStack, nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.SemiDatasets import FedSemiDataset, build_base_transform, get_n_classes, make_train_val_indices
from models.encoder import get_encoder
from models.graph_aggregator import HierarchicalGraphAggregator
from tree.tree import clone_aggregation_tensors, create_tree
from utils.evaluate import evaluate
from utils.losses import get_loss_function
from utils.warmup import get_warm_up_value
from utils.e1_logging import E1RunLogger
from utils.e1_random import e1_random_stream
from datasets.parallel_clients import initialize_cpu_worker
from utils.communication import communication_unit_bytes, strict_budget_enabled
import swanlab


def laplace_noise(shape, scale, device):
    uniform_noise = torch.rand(shape, device=device) - 0.5
    raw_noise = -scale * torch.log(1 - 2 * uniform_noise.abs() + 1e-6)
    raw_noise = raw_noise * uniform_noise.sign()
    min_val = raw_noise.min()
    max_val = raw_noise.max()
    return (raw_noise - min_val) / (max_val - min_val + 1e-6)


def scalar_zero(device):
    return torch.tensor(0.0, device=device)


def to_number(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cpu_threads(config):
    threads = OmegaConf.select(config, 'train.cpu_threads')
    if threads is None and OmegaConf.select(config, 'datasets.client_batch_workers') is not None:
        threads = 2
    if threads is not None:
        threads = int(threads)
        if threads < 1:
            raise ValueError('train.cpu_threads must be >= 1')
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        OmegaConf.update(config, 'train.cpu_threads', threads, force_add=True)


def compute_consistency_loss(consistency_criterion, student_probs, teacher_probs, criterion_name):
    if criterion_name.lower() == 'kl':
        return consistency_criterion(torch.log(student_probs + 1e-8), teacher_probs.detach())
    return consistency_criterion(student_probs, teacher_probs.detach())


def compute_soft_pseudo_weights(edge_confidence, cloud_confidence, agreement_mask, unlabeled_mask, config):
    scale = max(float(config.train.pseudo.weight_temperature), 1e-6)
    edge_score = torch.sigmoid((edge_confidence - config.train.pseudo.edge_threshold) / scale)
    cloud_score = torch.sigmoid((cloud_confidence - config.train.pseudo.cloud_threshold) / scale)
    return unlabeled_mask.float() * agreement_mask.float() * edge_score * cloud_score


def compute_weighted_cross_entropy(logits, targets, weights):
    per_sample_loss = F.cross_entropy(logits, targets, reduction='none')
    return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def compute_labeled_prototypes(features, labels, is_labeled, num_classes):
    if features.dim() > 2:
        features = features.flatten(start_dim=1)
    prototypes = torch.zeros(num_classes, features.size(1), device=features.device)
    counts = torch.zeros(num_classes, device=features.device)
    if is_labeled.sum() == 0:
        return prototypes, counts

    labeled_features = features[is_labeled]
    labeled_labels = labels[is_labeled]
    for class_id in range(num_classes):
        class_mask = labeled_labels == class_id
        if class_mask.any():
            prototypes[class_id] = labeled_features[class_mask].mean(dim=0)
            counts[class_id] = class_mask.sum()
    return prototypes, counts


def average_edge_parameters(edge_summaries, weighting='uniform'):
    """Standalone average helper; explicit weighting is required for E1 N/quality averages."""
    if not edge_summaries:
        return None
    if weighting not in {'uniform', 'sample', 'sample_quality'}:
        raise ValueError(f'Unknown source weighting: {weighting}')
    weights = None
    if weighting != 'uniform':
        weights = torch.tensor([s['batch_size'] for s in edge_summaries], dtype=torch.float32,
                               device=edge_summaries[0]['parameters'][0].device)
        if weighting == 'sample_quality':
            weights *= torch.tensor([1.0 + s['pseudo_quality'] for s in edge_summaries],
                                    dtype=torch.float32, device=weights.device)
        weights = HierarchicalGraphAggregator._normalize_weights(weights)
    avg_parameters = []
    for param_index in range(len(edge_summaries[0]['parameters'])):
        params = [summary['parameters'][param_index] for summary in edge_summaries]
        if params[0].is_floating_point() or params[0].is_complex():
            if weights is None:
                avg_parameters.append(torch.stack(params).mean(dim=0))
            else:
                avg_parameters.append(sum((weight * param for weight, param in zip(weights, params)),
                                          torch.zeros_like(params[0])))
        else:
            avg_parameters.append(params[0].clone())
    return avg_parameters


def sync_edge_models_from_cloud(tree):
    aggregation_mode = tree.root.aggregation_mode
    cloud_parameters = clone_aggregation_tensors(tree.root.model.model, aggregation_mode)
    for edge in tree.root.children:
        edge.set_parameters(cloud_parameters)


def tensor_megabytes(tensor):
    return tensor.numel() * tensor.element_size() / (1024 ** 2)


def client_payload_megabytes(batch, encoded_views, config):
    if strict_budget_enabled(config):
        # E4 matched tensor envelope, decimal MB. Excludes protocol/header/retry bytes.
        tensors = list(encoded_views) + [batch['label'], batch['is_labeled']]
        return sum(t.numel() * t.element_size() for t in tensors) / 1_000_000
    if OmegaConf.select(config, 'network.client_feature_dim') is not None:
        return sum(tensor_megabytes(view) for view in encoded_views)
    return (
        tensor_megabytes(batch['img'])
        + tensor_megabytes(batch['weak_img'])
        + tensor_megabytes(batch['strong_img'])
    )


def parameter_list_megabytes(parameters, config=None):
    return sum(param.numel() * param.element_size() for param in parameters) / communication_unit_bytes(config)


def edge_summary_megabytes(summary, config=None):
    total_bytes = 0
    for param in summary['parameters']:
        total_bytes += param.numel() * param.element_size()
    total_bytes += summary['prototype'].numel() * summary['prototype'].element_size()
    total_bytes += summary['prototype_counts'].numel() * summary['prototype_counts'].element_size()
    total_bytes += 3 * 4
    if strict_budget_enabled(config):
        total_bytes += 8  # int64 N
    return total_bytes / communication_unit_bytes(config)


def network_latency_seconds(payload_mb, bandwidth_mbps, parallel_factor=1):
    if bandwidth_mbps <= 0:
        return 0.0
    effective_bandwidth = bandwidth_mbps * max(parallel_factor, 1)
    return (payload_mb * 8.0) / effective_bandwidth


def parallel_upload_makespan_seconds(payloads_mb, bandwidths_mbps, max_parallel):
    if not payloads_mb:
        return 0.0
    upload_times = [
        network_latency_seconds(payload_mb, bandwidth_mbps, 1)
        for payload_mb, bandwidth_mbps in zip(payloads_mb, bandwidths_mbps)
    ]
    slot_count = min(max(1, int(max_parallel)), len(upload_times))
    slots = [0.0 for _ in range(slot_count)]
    for upload_time in sorted(upload_times, reverse=True):
        slot_index = min(range(slot_count), key=lambda idx: slots[idx])
        slots[slot_index] += upload_time
    return max(slots)


def _use_nslkdd_metrics(config):
    return str(OmegaConf.select(config, 'datasets.name', default='')).lower() == 'nslkdd'


def build_class_weights_from_labeled_clients(tree, num_classes, config, device):
    mode = str(OmegaConf.select(config, 'train.class_balance.mode', default='none')).lower()
    if mode in {'none', 'off', 'false'}:
        return None

    counts = torch.zeros(num_classes, dtype=torch.float32)
    use_labeled_only = bool(OmegaConf.select(config, 'train.class_balance.labeled_only', default=True))
    for entry in tree.client_registry.values():
        indices = torch.as_tensor(entry['indices'], dtype=torch.long)
        if use_labeled_only:
            mask = entry['labeled_mask'].bool()
            indices = indices[mask]
        if indices.numel() == 0:
            continue
        labels = torch.as_tensor(entry['targets'][indices.numpy()], dtype=torch.long)
        counts += torch.bincount(labels, minlength=num_classes).float()[:num_classes]

    if counts.sum().item() <= 0:
        return None

    smoothing = float(OmegaConf.select(config, 'train.class_balance.smoothing', default=1.0))
    exponent = float(OmegaConf.select(config, 'train.class_balance.exponent', default=0.5))
    min_weight = float(OmegaConf.select(config, 'train.class_balance.min_weight', default=0.25))
    max_weight = float(OmegaConf.select(config, 'train.class_balance.max_weight', default=8.0))
    weights = (counts.sum() / max(num_classes, 1)) / (counts + smoothing)
    weights = weights.clamp_min(1e-6).pow(exponent)
    weights = weights / weights.mean().clamp_min(1e-6)
    class_multipliers = OmegaConf.select(config, 'train.class_balance.class_multipliers', default=None)
    if class_multipliers is not None:
        multipliers = torch.ones(num_classes, dtype=torch.float32)
        for class_id, multiplier in enumerate(class_multipliers):
            if class_id < num_classes:
                multipliers[class_id] = float(multiplier)
        weights = weights * multipliers
        weights = weights / weights.mean().clamp_min(1e-6)
    weights = weights.clamp(min=min_weight, max=max_weight)
    print(f'Class counts for weighted loss: {[int(value) for value in counts.tolist()]}')
    print(f'Class weights for weighted loss: {[round(float(value), 4) for value in weights.tolist()]}')
    return weights.to(device)


def evaluate_edges(encoded_model, tree, val_loader, criterion, device, extra_metrics=False):
    edge_metrics = []
    for edge in tree.root.children:
        metrics = evaluate(
            encoded_model=encoded_model,
            model=edge.model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            extra_metrics=extra_metrics,
        )
        edge_metrics.append(metrics)

    if not edge_metrics:
        return {'edge_avg_loss': 0.0, 'edge_avg_acc': 0.0}

    edge_avg_loss = sum(metric['val_loss'] for metric in edge_metrics) / len(edge_metrics)
    edge_avg_acc = sum(metric['val_accuracy'] for metric in edge_metrics) / len(edge_metrics)
    result = {'edge_avg_loss': edge_avg_loss, 'edge_avg_acc': edge_avg_acc}
    metric_keys = set().union(*(metric.keys() for metric in edge_metrics))
    for key in metric_keys:
        if key.startswith('val_') and key not in {'val_loss', 'val_accuracy'}:
            result[f'edge_avg_{key[4:]}'] = sum(metric.get(key, 0.0) for metric in edge_metrics) / len(edge_metrics)
    return result


def compact_metric_dict(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if value is not None and isinstance(value, (int, float))
    }


def _has_unlabeled_data(config):
    return float(OmegaConf.select(config, 'datasets.labeled_ratio', default=1.0)) < 1.0


def _is_dynamic_network_experiment(config):
    mobility_mode = str(OmegaConf.select(config, 'network.mobility.mode', default='static')).lower()
    bandwidth_mode = str(OmegaConf.select(config, 'network.bandwidth.mode', default='homogeneous')).lower()
    drop_rate = OmegaConf.select(config, 'topology.client_drop_rate', default=0.0)
    if OmegaConf.is_list(drop_rate) or isinstance(drop_rate, (list, tuple)):
        has_drop = any(float(value) > 0 for value in drop_rate)
    else:
        has_drop = float(drop_rate) > 0
    return (
        has_drop
        or mobility_mode != 'static'
        or bandwidth_mode == 'sampled_distribution'
    )


def build_train_log_metrics(metrics, config):
    log_metrics = {
        'cumulative_comm_mb': metrics.get('cumulative_comm_mb'),
        'cumulative_estimated_latency_s': metrics.get('cumulative_estimated_latency_s'),
        'round_wall_clock_s': metrics.get('round_wall_clock_s'),
        'terminal_edge_upload_latency_s': metrics.get('terminal_edge_upload_latency_s'),
        'edge_compute_latency_s': metrics.get('edge_compute_latency_s'),
        'edge_cloud_upload_latency_s': metrics.get('edge_cloud_upload_latency_s'),
        'cloud_aggregation_latency_s': metrics.get('cloud_aggregation_latency_s'),
        'cloud_edge_downlink_latency_s': metrics.get('cloud_edge_downlink_latency_s'),
        'estimated_network_latency_s': metrics.get('estimated_network_latency_s'),
        'formal_round_latency_s': metrics.get('formal_round_latency_s'),
        'amortized_edge_cloud_upload_latency_s': metrics.get('amortized_edge_cloud_upload_latency_s'),
        'amortized_cloud_aggregation_latency_s': metrics.get('amortized_cloud_aggregation_latency_s'),
        'amortized_cloud_edge_downlink_latency_s': metrics.get('amortized_cloud_edge_downlink_latency_s'),
        'amortized_formal_round_latency_s': metrics.get('amortized_formal_round_latency_s'),
        'cloud_sync_active': metrics.get('cloud_sync_active'),
        'edge_total_loss': metrics.get('total_loss'),
        'edge_class_loss': metrics.get('avg_class_loss'),
        'active_edges': metrics.get('active_edges'),
        'selected_clients': metrics.get('selected_clients'),
        'active_clients': metrics.get('sampled_clients'),
    }

    if _has_unlabeled_data(config):
        log_metrics['edge_consistency_loss'] = metrics.get('avg_consis_loss')
        log_metrics['edge_weighted_consistency_loss'] = metrics.get('avg_weighted_consis_loss')
        log_metrics['avg_consistency_ratio'] = metrics.get('avg_consistency_ratio')

    if _has_unlabeled_data(config) and bool(OmegaConf.select(config, 'train.pseudo.use', default=False)):
        log_metrics.update({
            'edge_pseudo_loss': metrics.get('avg_pseudo_loss'),
            'edge_weighted_pseudo_loss': metrics.get('avg_weighted_pseudo_loss'),
            'avg_pseudo_ratio': metrics.get('avg_pseudo_ratio'),
            'avg_pseudo_weight': metrics.get('avg_pseudo_weight'),
            'avg_agreement_ratio': metrics.get('avg_agreement_ratio'),
            'avg_cloud_confidence': metrics.get('avg_cloud_confidence'),
            'avg_edge_confidence': metrics.get('avg_edge_confidence'),
        })

    if 'graph_attention_mean' in metrics:
        log_metrics.update({
            'graph_attention_mean': metrics.get('graph_attention_mean'),
            'graph_attention_diag': metrics.get('graph_attention_diag'),
            'graph_reliability_mean': metrics.get('graph_reliability_mean'),
            'graph_label_ratio_mean': metrics.get('graph_label_ratio_mean'),
            'graph_confidence_mean': metrics.get('graph_confidence_mean'),
        })

    if _is_dynamic_network_experiment(config):
        log_metrics.update({
            'total_comm_mb': metrics.get('total_comm_mb'),
            'samples_per_estimated_second': metrics.get('samples_per_estimated_second'),
            'budget_used_ratio': metrics.get('budget_used_ratio'),
            'client_drop_ratio': metrics.get('client_drop_ratio'),
            'selected_clients': metrics.get('selected_clients'),
            'active_clients': metrics.get('sampled_clients'),
        })

    return compact_metric_dict(log_metrics)


def print_best_metrics(best_metrics):
    print('\nBest Checkpoint Metrics:')
    print(f"Best Step: {best_metrics.get('step', 'N/A')}")
    print(f"Best Cloud Accuracy: {best_metrics.get('cloud_accuracy', 0.0):.2f}%")
    print(f"Best Cloud Loss: {best_metrics.get('cloud_loss', 0.0):.4f}")
    print(f"Best Edge Avg Accuracy: {best_metrics.get('edge_avg_accuracy', 0.0):.2f}%")
    print(f"Best Edge Avg Loss: {best_metrics.get('edge_avg_loss', 0.0):.4f}")
    if best_metrics.get('checkpoint_path') is not None:
        print(f"Best Checkpoint Path: {best_metrics['checkpoint_path']}")


def _state_dict_to_cpu(state_dict):
    return {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in state_dict.items()
    }


def save_tree_checkpoint(tree, path, step, val_metrics, edge_metrics):
    checkpoint = {
        'step': step,
        'cloud_model': _state_dict_to_cpu(tree.root.model.model.state_dict()),
        'edge_models': {
            edge.name: _state_dict_to_cpu(edge.model.state_dict())
            for edge in tree.root.children
        },
        'val_metrics': val_metrics,
        'edge_val_metrics': edge_metrics,
    }
    torch.save(checkpoint, path)


def load_tree_checkpoint(tree, path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'cloud_model' in checkpoint:
        tree.root.model.model.load_state_dict(checkpoint['cloud_model'])
        edge_states = checkpoint.get('edge_models', {})
        for edge in tree.root.children:
            if edge.name in edge_states:
                edge.model.load_state_dict(edge_states[edge.name])
        return checkpoint

    tree.root.model.model.load_state_dict(checkpoint)
    sync_edge_models_from_cloud(tree)
    return {'cloud_model': checkpoint, 'edge_models': {}}


def print_test_metrics(title, val_metrics, edge_metrics):
    print(f'\n{title}:')
    print(f"Cloud Loss: {val_metrics['val_loss']:.4f}")
    print(f"Cloud Accuracy: {val_metrics['val_accuracy']:.2f}%")
    print(f"Edge Avg Loss: {edge_metrics['edge_avg_loss']:.4f}")
    print(f"Edge Avg Accuracy: {edge_metrics['edge_avg_acc']:.2f}%")


def build_eval_log_metrics(val_metrics, edge_metrics, info_metrics, config):
    class_names = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    log_metrics = {
        'cloud_accuracy': val_metrics.get('val_accuracy'),
        'cloud_loss': val_metrics.get('val_loss'),
        'edge_avg_accuracy': edge_metrics.get('edge_avg_acc'),
        'edge_avg_loss': edge_metrics.get('edge_avg_loss'),
        'cumulative_comm_mb': info_metrics.get('cumulative_comm_mb'),
        'cumulative_estimated_latency_s': info_metrics.get('cumulative_estimated_latency_s'),
        'time_to_target_accuracy_s': info_metrics.get('time_to_target_accuracy_s'),
        'wall_time_to_target_accuracy_s': info_metrics.get('wall_time_to_target_accuracy_s'),
    }
    if _use_nslkdd_metrics(config):
        log_metrics.update({
            'cloud_macro_f1': val_metrics.get('val_macro_f1'),
            'cloud_weighted_f1': val_metrics.get('val_weighted_f1'),
            'cloud_macro_recall': val_metrics.get('val_macro_recall'),
            'edge_avg_macro_f1': edge_metrics.get('edge_avg_macro_f1'),
            'edge_avg_weighted_f1': edge_metrics.get('edge_avg_weighted_f1'),
            'edge_avg_macro_recall': edge_metrics.get('edge_avg_macro_recall'),
        })
        for class_id, class_name in enumerate(class_names):
            recall_key = f'val_recall_class_{class_id}'
            edge_recall_key = f'edge_avg_recall_class_{class_id}'
            if recall_key in val_metrics:
                log_metrics[f'cloud_recall_{class_name}'] = val_metrics.get(recall_key)
            if edge_recall_key in edge_metrics:
                log_metrics[f'edge_avg_recall_{class_name}'] = edge_metrics.get(edge_recall_key)
    return compact_metric_dict(log_metrics)


def evaluate_and_log(
    encoder_model,
    tree,
    valloader,
    classification_criterion,
    device,
    current_steps,
    cumulative_comm_mb,
    cumulative_estimated_latency_s,
    first_target_time,
    first_target_wall_time,
    train_start_wall,
    target_accuracy,
    best_metrics,
    config,
    e1_logger=None,
):
    val_metrics = evaluate(
        encoded_model=encoder_model,
        model=tree.root.model,
        val_loader=valloader,
        criterion=classification_criterion,
        device=device,
        extra_metrics=_use_nslkdd_metrics(config),
    )
    edge_val_metrics = evaluate_edges(
        encoded_model=encoder_model,
        tree=tree,
        val_loader=valloader,
        criterion=classification_criterion,
        device=device,
        extra_metrics=_use_nslkdd_metrics(config),
    )
    if first_target_time is None and val_metrics['val_accuracy'] >= target_accuracy:
        first_target_time = cumulative_estimated_latency_s
        first_target_wall_time = time.perf_counter() - train_start_wall
    info_metrics = {
        'cumulative_comm_mb': cumulative_comm_mb,
        'cumulative_estimated_latency_s': cumulative_estimated_latency_s,
        'time_to_target_accuracy_s': first_target_time if first_target_time is not None else -1.0,
        'wall_time_to_target_accuracy_s': first_target_wall_time if first_target_wall_time is not None else -1.0,
    }
    if e1_logger is not None:
        e1_logger.record_evaluation(current_steps, val_metrics, edge_val_metrics, info_metrics)
    print('\nValidation Metrics:')
    print(f"Cloud Loss: {val_metrics['val_loss']:.4f}")
    print(f"Cloud Accuracy: {val_metrics['val_accuracy']:.2f}%")
    if _use_nslkdd_metrics(config) and 'val_macro_f1' in val_metrics:
        print(f"Cloud Macro-F1: {val_metrics['val_macro_f1']:.4f}")
        print(f"Cloud Macro Recall: {val_metrics['val_macro_recall']:.4f}")
    print(f"Edge Avg Accuracy: {edge_val_metrics['edge_avg_acc']:.2f}%")
    if first_target_time is not None:
        print(f"Time to Target Accuracy (s): {first_target_time:.3f}")
        print(f"Wall Time to Target Accuracy (s): {first_target_wall_time:.3f}")
    wandb.log(build_eval_log_metrics(val_metrics, edge_val_metrics, info_metrics, config), step=current_steps)

    best_accuracy = float(best_metrics.get('cloud_accuracy', 0.0))
    if val_metrics['val_accuracy'] > best_accuracy:
        ckpt_dir = os.path.join(config.train.ckpt_save_path, config.datasets.name)
        ckpt_path = os.path.join(ckpt_dir, config.train.ckpt_save_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        save_tree_checkpoint(tree, ckpt_path, current_steps, val_metrics, edge_val_metrics)
        best_metrics.update({
            'cloud_accuracy': val_metrics['val_accuracy'],
            'cloud_loss': val_metrics['val_loss'],
            'edge_avg_accuracy': edge_val_metrics['edge_avg_acc'],
            'edge_avg_loss': edge_val_metrics['edge_avg_loss'],
            'step': current_steps,
            'checkpoint_path': ckpt_path,
        })
        print(f"Best checkpoint updated at step {current_steps}: {ckpt_path}")

    return best_metrics, first_target_time, first_target_wall_time


def edge_run_loop(
    tree,
    sampled_clients,
    encoder_model,
    config,
    current_steps,
    consistency_weight,
    classification_criterion,
    consistency_criterion,
    device,
    update=False,
):
    metrics = OrderedDict()
    total_loss = 0.0
    class_loss = []
    consis_loss = []
    pseudo_loss_values = []
    pseudo_ratio_values = []
    pseudo_weight_values = []
    agreement_ratio_values = []
    edge_confidence_values = []
    cloud_confidence_values = []
    unlabeled_ratio_values = []
    consistency_ratio_values = []
    edge_summaries = []
    sampled_client_count = 0
    num_classes = get_n_classes(config.datasets.name)
    cloud_node = tree.root
    client_upload_mb_by_edge = {}
    client_count_by_edge = {}
    edge_uplink_mb_by_edge = {}
    active_client_ids_by_edge = {}
    selected_client_count_by_edge = {}
    dropped_client_count_by_edge = {}
    explicit_losses = OmegaConf.select(config, 'train.losses') is not None
    supervised_enabled = bool(OmegaConf.select(config, 'train.losses.supervised', default=True))
    consistency_enabled = bool(OmegaConf.select(config, 'train.losses.consistency', default=True))
    pseudo_enabled = bool(config.train.pseudo.use) and bool(
        OmegaConf.select(config, 'train.losses.pseudo', default=True)
    )

    if tree.client_batch_loader is not None:
        tree.prepare_client_batches(sampled_clients, current_steps)

    for edge in cloud_node.children:
        client_ids = sampled_clients.get(edge.name, [])
        if not client_ids:
            continue

        selected_client_count_by_edge[edge.name] = len(client_ids)
        active_client_ids = tree.active_round_clients.get(edge.name, []) if tree.client_batch_loader is not None else [
            client_id for client_id in client_ids
            if random.random() >= tree.get_client_drop_probability(client_id, config)
        ]
        dropped_client_count_by_edge[edge.name] = len(client_ids) - len(active_client_ids)
        if not active_client_ids:
            continue

        edge.model.train()
        sampled_client_count += len(active_client_ids)
        client_count_by_edge[edge.name] = len(active_client_ids)
        active_client_ids_by_edge[edge.name] = active_client_ids
        batch = tree.get_edge_batch(active_client_ids)
        if batch is None:
            continue

        client_inputs = batch['img'].to(device, non_blocking=True)
        client_labels = batch['label'].to(device, non_blocking=True)
        client_is_labeled = batch['is_labeled'].to(device, non_blocking=True).bool()
        client_weak_input = batch['weak_img'].to(device, non_blocking=True)
        client_strong_input = batch['strong_img'].to(device, non_blocking=True)

        if random.random() < config.models.attack_rate:
            client_inputs = torch.randn_like(client_inputs, device=device)
            client_labels = torch.randint_like(client_labels, high=num_classes, device=device)
            client_is_labeled = torch.randint(0, 2, size=client_is_labeled.size(), device=device).bool()
            client_weak_input = torch.randn_like(client_weak_input, device=device)
            client_strong_input = torch.randn_like(client_strong_input, device=device)

        client_output = encoder_model(client_inputs)
        if config.train.differential.use:
            max_value = torch.max(client_output)
            min_value = torch.min(client_output)
            scale = (max_value - min_value) / config.train.differential.epsilon
            client_output = client_output + laplace_noise(client_output.shape, scale, device)
            client_output = torch.clamp(client_output, min=min_value, max=max_value)

        client_weak_output = encoder_model(client_weak_input)
        if config.train.differential.use:
            max_value = torch.max(client_weak_output)
            min_value = torch.min(client_weak_output)
            scale = (max_value - min_value) / config.train.differential.epsilon
            client_weak_output = client_weak_output + laplace_noise(client_weak_output.shape, scale, device)
            client_weak_output = torch.clamp(client_weak_output, min=min_value, max=max_value)

        client_strong_output = encoder_model(client_strong_input)
        if config.train.differential.use:
            max_value = torch.max(client_strong_output)
            min_value = torch.min(client_strong_output)
            scale = (max_value - min_value) / config.train.differential.epsilon
            client_strong_output = client_strong_output + laplace_noise(client_strong_output.shape, scale, device)
            client_strong_output = torch.clamp(client_strong_output, min=min_value, max=max_value)

        client_upload_mb_by_edge[edge.name] = client_payload_megabytes(
            batch,
            [client_output, client_weak_output, client_strong_output],
            config,
        )

        encoded_inputs = client_output
        all_client_labels = client_labels
        all_client_is_labeled = client_is_labeled
        all_client_weak = client_weak_output
        all_client_strong = client_strong_output

        edge_classification_output = edge.run_model(encoded_inputs)
        if supervised_enabled and all_client_is_labeled.any():
            classification_loss = classification_criterion(
                edge_classification_output[all_client_is_labeled],
                all_client_labels[all_client_is_labeled],
            )
        else:
            classification_loss = scalar_zero(device)

        unlabeled_mask = ~all_client_is_labeled
        pseudo_start_step = int(OmegaConf.select(config, 'train.pseudo.start_step', default=0))
        pseudo_active = pseudo_enabled and current_steps >= pseudo_start_step
        auxiliary_observed = not explicit_losses or consistency_enabled or pseudo_active
        if auxiliary_observed:
            edge_consistency_output = edge.run_model(all_client_strong)
            cloud_consistency_output = cloud_node.run_model(all_client_weak)
            edge_probs = F.softmax(edge_consistency_output / config.train.pseudo.temperature, dim=1)
            cloud_probs = F.softmax(cloud_consistency_output / config.train.pseudo.temperature, dim=1)
            edge_confidence, edge_prediction = edge_probs.max(dim=1)
            cloud_confidence, cloud_prediction = cloud_probs.max(dim=1)
            agreement_mask = edge_prediction.eq(cloud_prediction)
            pseudo_weights = compute_soft_pseudo_weights(edge_confidence, cloud_confidence, agreement_mask, unlabeled_mask, config)
            effective_mask = pseudo_weights > config.train.pseudo.min_weight
        else:
            pseudo_weights = torch.zeros(len(unlabeled_mask), device=device)
            edge_confidence = cloud_confidence = pseudo_weights
            agreement_mask = effective_mask = torch.zeros_like(unlabeled_mask)
        if pseudo_active and effective_mask.any():
            pseudo_loss = compute_weighted_cross_entropy(
                edge_consistency_output[effective_mask],
                cloud_prediction[effective_mask],
                pseudo_weights[effective_mask],
            )
        else:
            pseudo_loss = scalar_zero(device)

        consistency_mask = unlabeled_mask if consistency_enabled else torch.zeros_like(unlabeled_mask)
        if bool(OmegaConf.select(config, 'train.consistency_confidence_gate', default=False)):
            consistency_mask = effective_mask if consistency_enabled else consistency_mask
        if consistency_mask.any():
            consistency_loss = compute_consistency_loss(
                consistency_criterion,
                edge_probs[consistency_mask],
                cloud_probs[consistency_mask],
                config.train.consis_fn,
            )
        else:
            consistency_loss = scalar_zero(device)

        loss = classification_loss + consistency_weight * consistency_loss
        if pseudo_active:
            loss = loss + config.train.pseudo.weight * pseudo_loss
        if loss.requires_grad:
            edge.optimizer_step(loss)

        unlabeled_count = max(int(unlabeled_mask.sum().item()), 1)
        pseudo_ratio = effective_mask[unlabeled_mask].float().mean().item() if unlabeled_mask.any() else 0.0
        mean_pseudo_weight = pseudo_weights[unlabeled_mask].mean().item() if unlabeled_mask.any() else 0.0
        agreement_ratio = agreement_mask[unlabeled_mask].float().mean().item() if unlabeled_mask.any() else 0.0
        mean_edge_confidence = edge_confidence[unlabeled_mask].mean().item() if unlabeled_mask.any() else 0.0
        mean_cloud_confidence = cloud_confidence[unlabeled_mask].mean().item() if unlabeled_mask.any() else 0.0
        consistency_ratio = consistency_mask[unlabeled_mask].float().mean().item() if unlabeled_mask.any() else 0.0
        mean_confidence = mean_cloud_confidence
        labeled_ratio = all_client_is_labeled.float().mean().item()
        unlabeled_ratio = unlabeled_mask.float().mean().item()
        prototypes, prototype_counts = compute_labeled_prototypes(
            encoded_inputs.detach(),
            all_client_labels,
            all_client_is_labeled,
            num_classes,
        )

        total_loss += loss.item()
        class_loss.append(to_number(classification_loss))
        consis_loss.append(to_number(consistency_loss))
        pseudo_loss_values.append(to_number(pseudo_loss))
        pseudo_ratio_values.append(pseudo_ratio)
        pseudo_weight_values.append(mean_pseudo_weight)
        agreement_ratio_values.append(agreement_ratio)
        edge_confidence_values.append(mean_edge_confidence)
        cloud_confidence_values.append(mean_cloud_confidence)
        unlabeled_ratio_values.append(unlabeled_ratio)
        consistency_ratio_values.append(consistency_ratio)

        if update:
            summary = {
                'name': edge.name,
                'parameters': edge.get_parameters(),
                'prototype': prototypes.detach().clone(),
                'prototype_counts': prototype_counts.detach().clone(),
                'pseudo_quality': mean_pseudo_weight,
                'labeled_ratio': labeled_ratio,
                'mean_confidence': mean_confidence,
                'batch_size': int(encoded_inputs.size(0)),
                'quality_observed': auxiliary_observed,
            }
            edge_summaries.append(summary)
            edge_uplink_mb_by_edge[edge.name] = edge_summary_megabytes(summary, config)

    active_edges = max(len(class_loss), 1)
    metrics['total_loss'] = total_loss / active_edges
    metrics['avg_class_loss'] = sum(class_loss) / active_edges if class_loss else 0.0
    metrics['avg_consis_loss'] = sum(consis_loss) / active_edges if consis_loss else 0.0
    metrics['avg_pseudo_loss'] = sum(pseudo_loss_values) / active_edges if pseudo_loss_values else 0.0
    metrics['avg_weighted_consis_loss'] = consistency_weight * metrics['avg_consis_loss']
    metrics['avg_weighted_pseudo_loss'] = config.train.pseudo.weight * metrics['avg_pseudo_loss'] if config.train.pseudo.use else 0.0
    metrics['avg_pseudo_ratio'] = sum(pseudo_ratio_values) / active_edges if pseudo_ratio_values else 0.0
    metrics['avg_pseudo_weight'] = sum(pseudo_weight_values) / active_edges if pseudo_weight_values else 0.0
    metrics['avg_agreement_ratio'] = sum(agreement_ratio_values) / active_edges if agreement_ratio_values else 0.0
    metrics['avg_edge_confidence'] = sum(edge_confidence_values) / active_edges if edge_confidence_values else 0.0
    metrics['avg_cloud_confidence'] = sum(cloud_confidence_values) / active_edges if cloud_confidence_values else 0.0
    metrics['avg_unlabeled_ratio'] = sum(unlabeled_ratio_values) / active_edges if unlabeled_ratio_values else 0.0
    metrics['avg_consistency_ratio'] = sum(consistency_ratio_values) / active_edges if consistency_ratio_values else 0.0
    metrics['sampled_clients'] = sampled_client_count
    selected_client_count = sum(selected_client_count_by_edge.values())
    dropped_client_count = sum(dropped_client_count_by_edge.values())
    metrics['selected_clients'] = selected_client_count
    metrics['dropped_clients'] = dropped_client_count
    metrics['client_drop_ratio'] = dropped_client_count / max(selected_client_count, 1)
    metrics['active_edges'] = len(class_loss)
    metrics['classification_loss'] = class_loss
    metrics['consistency_loss'] = consis_loss
    metrics['pseudo_loss'] = pseudo_loss_values

    network_state = {
        'client_upload_mb_by_edge': client_upload_mb_by_edge,
        'edge_uplink_mb_by_edge': edge_uplink_mb_by_edge,
        'client_count_by_edge': client_count_by_edge,
        'active_client_ids_by_edge': active_client_ids_by_edge,
        'selected_client_count_by_edge': selected_client_count_by_edge,
        'dropped_client_count_by_edge': dropped_client_count_by_edge,
    }
    if tree.client_batch_loader is not None:
        network_state['client_batch_plan'] = tree.client_batch_loader.batch_plan
    if strict_budget_enabled(config):
        network_state['selection'] = tree.selection_state
    return edge_summaries if update else None, metrics, network_state


def train(config):
    with ExitStack() as resources:
        return _train(config, resources)


def _train(config, resources):
    if str(config.datasets.name).lower() == 'harbox':
        train_users = OmegaConf.select(config, 'datasets.harbox_train_users')
        if train_users is not None:
            if int(train_users) < 1:
                raise ValueError('datasets.harbox_train_users must be positive')
            os.environ['HARBOX_TRAIN_USERS'] = str(int(train_users))
    configure_cpu_threads(config)
    seed = int(OmegaConf.select(config, 'datasets.seed', default=0))
    set_random_seed(seed)
    paired_inputs = bool(OmegaConf.select(config, 'experiment.paired_inputs', default=False))
    client_workers = OmegaConf.select(config, 'datasets.client_batch_workers')
    if strict_budget_enabled(config):
        if client_workers is None or not paired_inputs:
            raise ValueError('Strict E4 requires client_batch_workers and paired_inputs')
        if int(config.network.client_view_count) != 3 or int(config.network.bytes_per_value) != 4:
            raise ValueError('Strict E4 requires three float32 encoded views')
    if paired_inputs and client_workers is None and int(config.datasets.edge_num_workers) != 0:
        raise ValueError('Legacy paired_inputs requires edge_num_workers=0; use --client-workers for parallel client batches')
    if client_workers is not None and int(client_workers) < 0:
        raise ValueError('datasets.client_batch_workers must be >= 0')
    # Validate modes before constructing encoders/datasets (which may otherwise download).
    graph_aggregator = HierarchicalGraphAggregator(config)
    e1_logger = E1RunLogger(config, graph_aggregator) if bool(
        OmegaConf.select(config, 'experiment.e1_logging', default=False)
    ) else None
    device = torch.device(f"cuda:{config.train.device}" if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(config.train.device)

    transform = build_base_transform(
        config.datasets.name,
        image_size=int(config.network.client_image_size),
        normalize=OmegaConf.select(config, 'datasets.normalize', default='imagenet'),
    )

    val_ratio = float(OmegaConf.select(config, 'datasets.val_ratio', default=0.0))
    split_seed = int(OmegaConf.select(config, 'datasets.seed', default=0))
    if val_ratio > 0:
        _, val_indices = make_train_val_indices(config.datasets.name, val_ratio=val_ratio, seed=split_seed)
        valset = FedSemiDataset(
            labeled_ratio=1,
            train=True,
            transform=transform,
            data_name=config.datasets.name,
            indices=val_indices,
        )
    else:
        print('Warning: datasets.val_ratio <= 0, using the official test split for validation/model selection.')
        valset = FedSemiDataset(labeled_ratio=1, train=False, transform=transform, data_name=config.datasets.name)
    eval_options = {}
    if client_workers is not None and int(config.datasets.eval_num_workers) > 0:
        # Avoid forking an initialized CUDA / SDK / CPU-thread-pool process.
        eval_options = dict(multiprocessing_context='spawn', persistent_workers=True,
                            worker_init_fn=initialize_cpu_worker)
    valloader = DataLoader(
        valset,
        batch_size=config.datasets.batch_size,
        shuffle=False,
        num_workers=config.datasets.eval_num_workers,
        pin_memory=config.datasets.pin_memory,
        **eval_options,
    )
    testset = FedSemiDataset(labeled_ratio=1, train=False, transform=transform, data_name=config.datasets.name)
    testloader = DataLoader(
        testset,
        batch_size=config.datasets.batch_size,
        shuffle=False,
        num_workers=config.datasets.eval_num_workers,
        pin_memory=config.datasets.pin_memory,
        **eval_options,
    )
    encoder_model = get_encoder(config).to(device)

    tree = create_tree(config)
    resources.callback(tree.close_batch_loaders)
    if client_workers is not None:
        # Release evaluation pools as well, including on an exception.
        resources.callback(close_persistent_loader, valloader)
        resources.callback(close_persistent_loader, testloader)
        print(f'Input protocol: client_keyed_v2; client workers={client_workers}; '
              f'CPU threads={torch.get_num_threads()}; prefetch/client worker='
              f'{OmegaConf.select(config, "datasets.client_batch_prefetch_factor", default=1)}')
    tree.move_to_device(device)
    sync_edge_models_from_cloud(tree)
    if e1_logger is not None:
        e1_logger.record_inputs(tree, config, encoder_model)
        print(f'E1 records: {e1_logger.directory}')

    num_classes = get_n_classes(config.datasets.name)
    class_weights = build_class_weights_from_labeled_clients(tree, num_classes, config, device)
    classification_criterion, consistency_criterion = get_loss_function(
        config.train.class_fn,
        config.train.consis_fn,
        num_class=num_classes,
        class_weights=class_weights,
        focal_gamma=OmegaConf.select(config, 'train.class_balance.focal_gamma', default=2.0),
    )
    classification_criterion = classification_criterion.to(device)
    consistency_criterion = consistency_criterion.to(device)

    current_steps = 0
    best_metrics = {'cloud_accuracy': 0.0}
    round_start_wall = time.perf_counter()
    first_target_time = None
    first_target_wall_time = None
    train_start_wall = time.perf_counter()
    cumulative_estimated_latency_s = 0.0
    cumulative_comm_mb = 0.0
    final_train_metrics = {}
    last_edge_cloud_upload_latency_s = 0.0
    last_cloud_aggregation_latency_s = 0.0
    last_cloud_edge_downlink_latency_s = 0.0
    best_metrics, first_target_time, first_target_wall_time = evaluate_and_log(
        encoder_model=encoder_model,
        tree=tree,
        valloader=valloader,
        classification_criterion=classification_criterion,
        device=device,
        current_steps=0,
        cumulative_comm_mb=cumulative_comm_mb,
        cumulative_estimated_latency_s=cumulative_estimated_latency_s,
        first_target_time=first_target_time,
        first_target_wall_time=first_target_wall_time,
        train_start_wall=train_start_wall,
        target_accuracy=config.network.target_accuracy,
        best_metrics=best_metrics,
        config=config,
        e1_logger=e1_logger,
    )
    while current_steps < config.train.total_steps:
        round_start_wall = time.perf_counter()
        consistency_weight = get_warm_up_value(
            current_epoch=current_steps,
            warm_up_epochs=config.train.warm_up_steps,
            initial_weight=config.train.initial_weight,
            final_weight=config.train.final_weight,
            mode=config.train.warm_mode,
        )
        sync_this_round = current_steps % config.train.ema_update_step == 0
        include_sync_in_budget = bool(OmegaConf.select(config, 'network.budget.include_model_sync', default=True))
        with e1_random_stream(seed, 'selection', current_steps) if paired_inputs else nullcontext():
            sampled_clients = tree.sample_clients(config, include_model_sync=include_sync_in_budget and sync_this_round,
                                                 step=current_steps)
        with e1_random_stream(seed, 'edge_step', current_steps, device) if paired_inputs else nullcontext():
            edge_summaries, train_metrics, network_state = edge_run_loop(
                tree=tree,
                sampled_clients=sampled_clients,
                encoder_model=encoder_model,
                config=config,
                current_steps=current_steps,
                consistency_weight=consistency_weight,
                classification_criterion=classification_criterion,
                consistency_criterion=consistency_criterion,
                device=device,
                update=sync_this_round,
            )
        if e1_logger is not None:
            e1_logger.record_participation(current_steps, sampled_clients, network_state)

        train_metrics['consistency_weight'] = consistency_weight
        client_upload_mb = sum(network_state['client_upload_mb_by_edge'].values())
        edge_upload_mb = sum(network_state['edge_uplink_mb_by_edge'].values())
        cloud_downlink_mb = 0.0
        cloud_downlink_mb_by_edge = {}
        teacher_downlink_mb = 0.0
        if edge_summaries is not None:
            aggregation_result = graph_aggregator.aggregate(edge_summaries)
            # Capture decay BEFORE AEMA.update_decay(), not the value for the next update.
            ema_decay_used = tree.root.model.decay
            if aggregation_result['global_parameters'] is not None:
                global_parameters = aggregation_result['global_parameters']
                tree.root.model.update_by_parameters(global_parameters)
                for edge in tree.root.children:
                    personalized_parameters = aggregation_result['personalized_parameters'].get(edge.name)
                    edge_parameters = personalized_parameters if personalized_parameters is not None else global_parameters
                    edge.set_parameters(edge_parameters)
                    edge_downlink_mb = parameter_list_megabytes(edge_parameters, config)
                    if strict_budget_enabled(config):
                        # Common model+teacher dispatch envelope for mechanism controls,
                        # including supervised-only controls; simulated, not physical sockets.
                        teacher_size = tree._estimate_edge_sync_mb()
                        teacher_downlink_mb += teacher_size
                        edge_downlink_mb += teacher_size
                    cloud_downlink_mb_by_edge[edge.name] = edge_downlink_mb
                    cloud_downlink_mb += edge_downlink_mb
                train_metrics.update(aggregation_result['metrics'])
            if e1_logger is not None:
                e1_logger.record_aggregation(
                    current_steps, aggregation_result, edge_summaries,
                    tree.root.model.algorithm, ema_decay_used,
                    list(cloud_downlink_mb_by_edge),
                )
            train_metrics['ema_decay'] = tree.root.model.decay

        client_latency_candidates = []
        for edge_name, payload_mb in network_state['client_upload_mb_by_edge'].items():
            count = network_state['client_count_by_edge'].get(edge_name, 1)
            client_ids = network_state['active_client_ids_by_edge'].get(edge_name, [])
            payload_per_client_mb = payload_mb / max(count, 1)
            payloads = [payload_per_client_mb for _ in client_ids]
            bandwidths = [tree.get_client_uplink_bandwidth(client_id) for client_id in client_ids]
            client_latency_candidates.append(
                parallel_upload_makespan_seconds(
                    payloads,
                    bandwidths,
                    config.network.max_parallel_uploads_per_edge,
                )
            )
        terminal_edge_upload_latency_s = max(client_latency_candidates) if client_latency_candidates else 0.0

        edge_uplink_latency_candidates = [
            network_latency_seconds(payload_mb, tree.get_edge_uplink_bandwidth(edge_name), 1)
            for edge_name, payload_mb in network_state['edge_uplink_mb_by_edge'].items()
        ]
        edge_cloud_upload_latency_s = max(edge_uplink_latency_candidates) if edge_uplink_latency_candidates else 0.0

        downlink_latency_candidates = [
            network_latency_seconds(payload_mb, tree.get_cloud_downlink_bandwidth(edge_name), 1)
            for edge_name, payload_mb in cloud_downlink_mb_by_edge.items()
        ]
        cloud_edge_downlink_latency_s = max(downlink_latency_candidates) if downlink_latency_candidates else 0.0
        edge_compute_latency_s = float(OmegaConf.select(config, 'network.latency.edge_compute_s', default=0.0))
        cloud_aggregation_latency_s = float(OmegaConf.select(config, 'network.latency.cloud_aggregation_s', default=0.0))
        round_wall_clock_s = time.perf_counter() - round_start_wall
        estimated_network_latency_s = terminal_edge_upload_latency_s + edge_cloud_upload_latency_s + cloud_edge_downlink_latency_s
        formal_round_latency_s = (
            terminal_edge_upload_latency_s
            + edge_compute_latency_s
            + edge_cloud_upload_latency_s
            + cloud_aggregation_latency_s
            + cloud_edge_downlink_latency_s
        )
        sync_interval = max(int(config.train.ema_update_step), 1)
        cloud_sync_active = 1.0 if edge_summaries is not None else 0.0
        if edge_summaries is not None:
            last_edge_cloud_upload_latency_s = edge_cloud_upload_latency_s
            last_cloud_aggregation_latency_s = cloud_aggregation_latency_s
            last_cloud_edge_downlink_latency_s = cloud_edge_downlink_latency_s
        amortized_edge_cloud_upload_latency_s = last_edge_cloud_upload_latency_s / sync_interval
        amortized_cloud_aggregation_latency_s = last_cloud_aggregation_latency_s / sync_interval
        amortized_cloud_edge_downlink_latency_s = last_cloud_edge_downlink_latency_s / sync_interval
        amortized_formal_round_latency_s = (
            terminal_edge_upload_latency_s
            + edge_compute_latency_s
            + amortized_edge_cloud_upload_latency_s
            + amortized_cloud_aggregation_latency_s
            + amortized_cloud_edge_downlink_latency_s
        )
        cumulative_estimated_latency_s += formal_round_latency_s
        total_comm_mb = client_upload_mb + edge_upload_mb + cloud_downlink_mb
        cumulative_comm_mb += total_comm_mb
        if strict_budget_enabled(config):
            budget = float(config.network.round_comm_budget_mb)
            if total_comm_mb > budget + 1e-9:
                raise RuntimeError(f'Strict payload exceeds budget: {total_comm_mb} > {budget} MB')
            if e1_logger is not None:
                e1_logger.record_communication(current_steps, client_upload_mb, edge_upload_mb,
                                               cloud_downlink_mb-teacher_downlink_mb,
                                               teacher_downlink_mb, cumulative_comm_mb, budget,
                                               tree.selection_state)
        processed_samples = train_metrics['sampled_clients'] * config.datasets.batch_size
        train_metrics.update({
            'client_upload_mb': client_upload_mb,
            'edge_upload_mb': edge_upload_mb,
            'cloud_downlink_mb': cloud_downlink_mb,
            'total_comm_mb': total_comm_mb,
            'cumulative_comm_mb': cumulative_comm_mb,
            'round_comm_budget_mb': float(config.network.round_comm_budget_mb),
            'budget_used_ratio': total_comm_mb / max(float(config.network.round_comm_budget_mb), 1e-8),
            'terminal_edge_upload_latency_s': terminal_edge_upload_latency_s,
            'client_uplink_latency_s': terminal_edge_upload_latency_s,
            'edge_compute_latency_s': edge_compute_latency_s,
            'edge_cloud_upload_latency_s': edge_cloud_upload_latency_s,
            'edge_uplink_latency_s': edge_cloud_upload_latency_s,
            'cloud_aggregation_latency_s': cloud_aggregation_latency_s,
            'cloud_edge_downlink_latency_s': cloud_edge_downlink_latency_s,
            'cloud_downlink_latency_s': cloud_edge_downlink_latency_s,
            'estimated_network_latency_s': estimated_network_latency_s,
            'formal_round_latency_s': formal_round_latency_s,
            'amortized_edge_cloud_upload_latency_s': amortized_edge_cloud_upload_latency_s,
            'amortized_cloud_aggregation_latency_s': amortized_cloud_aggregation_latency_s,
            'amortized_cloud_edge_downlink_latency_s': amortized_cloud_edge_downlink_latency_s,
            'amortized_formal_round_latency_s': amortized_formal_round_latency_s,
            'cloud_sync_active': cloud_sync_active,
            'cumulative_estimated_latency_s': cumulative_estimated_latency_s,
            'round_wall_clock_s': round_wall_clock_s,
            'throughput_samples_per_s': processed_samples / max(round_wall_clock_s, 1e-8),
            'samples_per_estimated_second': processed_samples / max(formal_round_latency_s, 1e-8),
        })
        final_train_metrics = train_metrics

        wandb.log(build_train_log_metrics(train_metrics, config), step=current_steps)

        if (current_steps + 1) % config.train.log_step == 0:
            print(f"\nStep [{current_steps + 1}/{config.train.total_steps}]")
            print('Training Metrics:')
            print(f"Total Loss: {train_metrics['total_loss']:.4f}")
            print(f"Classification Loss: {train_metrics['avg_class_loss']:.4f}")
            print(f"Consistency Loss: {train_metrics['avg_consis_loss']:.4f}")
            print(f"Pseudo Loss: {train_metrics['avg_pseudo_loss']:.4f}")
            print(f"Pseudo Ratio: {train_metrics['avg_pseudo_ratio']:.4f}")
            print(f"Sampled Clients: {train_metrics['sampled_clients']}")
            print(f"Client Drop Ratio: {train_metrics['client_drop_ratio']:.4f}")
            print(f"Total Comm (MB): {train_metrics['total_comm_mb']:.2f}")
            print(f"Round Wall Time (s): {train_metrics['round_wall_clock_s']:.3f}")
            print(f"Terminal-Edge Upload Latency (s): {train_metrics['terminal_edge_upload_latency_s']:.3f}")
            print(f"Edge Compute Latency (s): {train_metrics['edge_compute_latency_s']:.3f}")
            print(f"Edge-Cloud Upload Latency (s): {train_metrics['edge_cloud_upload_latency_s']:.3f}")
            print(f"Cloud Aggregation Latency (s): {train_metrics['cloud_aggregation_latency_s']:.3f}")
            print(f"Cloud-Edge Downlink Latency (s): {train_metrics['cloud_edge_downlink_latency_s']:.3f}")
            print(f"Estimated Network Latency (s): {train_metrics['estimated_network_latency_s']:.3f}")
            print(f"Formal Round Latency (s): {train_metrics['formal_round_latency_s']:.3f}")
            print(f"Amortized Edge-Cloud Upload Latency (s): {train_metrics['amortized_edge_cloud_upload_latency_s']:.3f}")
            print(f"Amortized Cloud Aggregation Latency (s): {train_metrics['amortized_cloud_aggregation_latency_s']:.3f}")
            print(f"Amortized Cloud-Edge Downlink Latency (s): {train_metrics['amortized_cloud_edge_downlink_latency_s']:.3f}")
            print(f"Amortized Formal Round Latency (s): {train_metrics['amortized_formal_round_latency_s']:.3f}")

        if (current_steps + 1) % config.train.evaluate_step == 0:
            best_metrics, first_target_time, first_target_wall_time = evaluate_and_log(
                encoder_model=encoder_model,
                tree=tree,
                valloader=valloader,
                classification_criterion=classification_criterion,
                device=device,
                current_steps=current_steps + 1,
                cumulative_comm_mb=cumulative_comm_mb,
                cumulative_estimated_latency_s=cumulative_estimated_latency_s,
                first_target_time=first_target_time,
                first_target_wall_time=first_target_wall_time,
                train_start_wall=train_start_wall,
                target_accuracy=config.network.target_accuracy,
                best_metrics=best_metrics,
                config=config,
                e1_logger=e1_logger,
            )

        current_steps += 1

    if e1_logger is not None:
        # Preserve the fixed horizon before the existing best-validation reload below.
        if e1_logger.last_evaluation['step'] != current_steps:
            best_metrics, first_target_time, first_target_wall_time = evaluate_and_log(
                encoder_model, tree, valloader, classification_criterion, device, current_steps,
                cumulative_comm_mb, cumulative_estimated_latency_s, first_target_time,
                first_target_wall_time, train_start_wall, config.network.target_accuracy,
                best_metrics, config, e1_logger=e1_logger,
            )
        last_evaluation = e1_logger.last_evaluation
        save_tree_checkpoint(tree, str(e1_logger.directory / 'final.pth'), current_steps,
                             last_evaluation['cloud'], last_evaluation['shared_split_edge_mean'])
        final_test = evaluate(encoder_model, tree.root.model, testloader,
                              classification_criterion, device, extra_metrics=_use_nslkdd_metrics(config))
        final_edges = evaluate_edges(encoder_model, tree, testloader,
                                    classification_criterion, device, extra_metrics=_use_nslkdd_metrics(config))
        e1_logger.record_evaluation(current_steps, final_test, final_edges,
                                    last_evaluation['costs'], split='fixed_horizon_test')
        print_test_metrics('Fixed-Horizon Test Metrics (E1)', final_test, final_edges)

    print_best_metrics(best_metrics)
    if best_metrics.get('checkpoint_path') is not None:
        load_tree_checkpoint(tree, best_metrics['checkpoint_path'], device)
        best_test_metrics = evaluate(
            encoded_model=encoder_model,
            model=tree.root.model,
            val_loader=testloader,
            criterion=classification_criterion,
            device=device,
            extra_metrics=_use_nslkdd_metrics(config),
        )
        best_edge_test_metrics = evaluate_edges(
            encoded_model=encoder_model,
            tree=tree,
            val_loader=testloader,
            criterion=classification_criterion,
            device=device,
            extra_metrics=_use_nslkdd_metrics(config),
        )
        print_test_metrics('Final Test Metrics at Best Validation Checkpoint', best_test_metrics, best_edge_test_metrics)


def close_persistent_loader(loader):
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        shutdown = getattr(iterator, '_shutdown_workers', None)
        if shutdown is not None:
            shutdown()
        loader._iterator = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/formated_config.yaml')
    parser.add_argument('--client-workers', type=int, default=None,
                        help='Opt into client_keyed_v2: 0=serial reference, >0=spawn client-batch workers.')
    parser.add_argument('--cpu-threads', type=int, default=None,
                        help='Main-process PyTorch CPU threads (client mode defaults to 2).')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.client_workers is not None:
        if args.client_workers < 0:
            parser.error('--client-workers must be >= 0')
        OmegaConf.update(config, 'datasets.client_batch_workers', args.client_workers, force_add=True)
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error('--cpu-threads must be >= 1')
        OmegaConf.update(config, 'train.cpu_threads', args.cpu_threads, force_add=True)
    # Apply the cap before SDK startup and any model/data tensor operations.
    configure_cpu_threads(config)
    os.makedirs('./logs', exist_ok=True)
    swanlab.sync_wandb()
    wandb.init(
        project='EdgeQGFed',
        dir='logs',
        name=f"EdgeQGFed-{config.models.encoder_name}-{config.datasets.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}",
        config=OmegaConf.to_container(config),
        job_type='train',
    )
    train(config)
