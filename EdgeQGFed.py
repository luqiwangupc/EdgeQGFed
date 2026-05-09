import datetime
import argparse
import os
import random
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.SemiDatasets import FedSemiDataset, build_base_transform, get_n_classes
from models.encoder import get_encoder
from models.graph_aggregator import HierarchicalGraphAggregator
from tree.tree import create_tree
from utils.evaluate import evaluate
from utils.losses import get_loss_function
from utils.warmup import get_warm_up_value
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


def average_edge_parameters(edge_summaries):
    if not edge_summaries:
        return None
    avg_parameters = []
    for param_index in range(len(edge_summaries[0]['parameters'])):
        params = [summary['parameters'][param_index] for summary in edge_summaries]
        if params[0].is_floating_point() or params[0].is_complex():
            avg_parameters.append(torch.stack(params).mean(dim=0))
        else:
            avg_parameters.append(params[0].clone())
    return avg_parameters


def sync_edge_models_from_cloud(tree):
    cloud_parameters = [tensor.detach().clone() for tensor in tree.root.model.model.state_dict().values()]
    for edge in tree.root.children:
        edge.set_parameters(cloud_parameters)


def tensor_megabytes(tensor):
    return tensor.numel() * tensor.element_size() / (1024 ** 2)


def parameter_list_megabytes(parameters):
    return sum(param.numel() * param.element_size() for param in parameters) / (1024 ** 2)


def edge_summary_megabytes(summary):
    total_bytes = 0
    for param in summary['parameters']:
        total_bytes += param.numel() * param.element_size()
    total_bytes += summary['prototype'].numel() * summary['prototype'].element_size()
    total_bytes += summary['prototype_counts'].numel() * summary['prototype_counts'].element_size()
    total_bytes += 3 * 4
    return total_bytes / (1024 ** 2)


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


def evaluate_edges(encoded_model, tree, val_loader, criterion, device):
    edge_metrics = []
    for edge in tree.root.children:
        metrics = evaluate(
            encoded_model=encoded_model,
            model=edge.model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )
        edge_metrics.append(metrics)

    if not edge_metrics:
        return {'edge_avg_loss': 0.0, 'edge_avg_acc': 0.0}

    edge_avg_loss = sum(metric['val_loss'] for metric in edge_metrics) / len(edge_metrics)
    edge_avg_acc = sum(metric['val_accuracy'] for metric in edge_metrics) / len(edge_metrics)
    return {'edge_avg_loss': edge_avg_loss, 'edge_avg_acc': edge_avg_acc}


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
        'edge_total_loss': metrics.get('total_loss'),
        'edge_class_loss': metrics.get('avg_class_loss'),
    }

    if _has_unlabeled_data(config):
        log_metrics['edge_consistency_loss'] = metrics.get('avg_consis_loss')

    if _has_unlabeled_data(config) and bool(OmegaConf.select(config, 'train.pseudo.use', default=False)):
        log_metrics.update({
            'edge_pseudo_loss': metrics.get('avg_pseudo_loss'),
            'avg_pseudo_ratio': metrics.get('avg_pseudo_ratio'),
            'avg_pseudo_weight': metrics.get('avg_pseudo_weight'),
            'avg_agreement_ratio': metrics.get('avg_agreement_ratio'),
            'avg_cloud_confidence': metrics.get('avg_cloud_confidence'),
            'avg_edge_confidence': metrics.get('avg_edge_confidence'),
        })

    if bool(OmegaConf.select(config, 'models.graph.use', default=False)):
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
            'formal_round_latency_s': metrics.get('formal_round_latency_s'),
            'samples_per_estimated_second': metrics.get('samples_per_estimated_second'),
            'budget_used_ratio': metrics.get('budget_used_ratio'),
            'client_drop_ratio': metrics.get('client_drop_ratio'),
            'selected_clients': metrics.get('selected_clients'),
            'active_clients': metrics.get('sampled_clients'),
        })

    return compact_metric_dict(log_metrics)


def build_summary_metrics(config, best_accuracy, cumulative_estimated_latency_s, cumulative_comm_mb, final_metrics=None):
    final_metrics = final_metrics or {}
    return compact_metric_dict({
        'best_cloud_accuracy': best_accuracy,
        'total_estimated_latency_s': cumulative_estimated_latency_s,
        'total_comm_mb': cumulative_comm_mb,
        'final_total_comm_mb': final_metrics.get('total_comm_mb'),
        'final_formal_round_latency_s': final_metrics.get('formal_round_latency_s'),
        'final_samples_per_estimated_second': final_metrics.get('samples_per_estimated_second'),
        'final_budget_used_ratio': final_metrics.get('budget_used_ratio'),
        'final_selected_clients': final_metrics.get('selected_clients'),
        'final_active_clients': final_metrics.get('sampled_clients'),
        'final_client_drop_ratio': final_metrics.get('client_drop_ratio'),
        'round_comm_budget_mb': float(config.network.round_comm_budget_mb),
        'target_accuracy': float(config.network.target_accuracy),
        'labeled_ratio': float(config.datasets.labeled_ratio),
        'partition_alpha': float(config.datasets.partition_alpha),
        'num_clients': int(config.topology.num_clients),
        'num_edges': int(config.topology.num_edges),
        'clients_per_round': int(config.topology.clients_per_round),
        'edges_per_round': int(config.topology.edges_per_round),
        'client_drop_rate': float(config.topology.client_drop_rate),
        'max_parallel_uploads_per_edge': int(config.network.max_parallel_uploads_per_edge),
    })


def build_eval_log_metrics(val_metrics, edge_metrics, info_metrics):
    return compact_metric_dict({
        'cloud_accuracy': val_metrics.get('val_accuracy'),
        'cloud_loss': val_metrics.get('val_loss'),
        'edge_avg_accuracy': edge_metrics.get('edge_avg_acc'),
        'edge_avg_loss': edge_metrics.get('edge_avg_loss'),
        'cumulative_comm_mb': info_metrics.get('cumulative_comm_mb'),
        'cumulative_estimated_latency_s': info_metrics.get('cumulative_estimated_latency_s'),
        'time_to_target_accuracy_s': info_metrics.get('time_to_target_accuracy_s'),
        'wall_time_to_target_accuracy_s': info_metrics.get('wall_time_to_target_accuracy_s'),
    })


def edge_run_loop(
    tree,
    sampled_clients,
    encoder_model,
    config,
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

    for edge in cloud_node.children:
        client_ids = sampled_clients.get(edge.name, [])
        if not client_ids:
            continue

        selected_client_count_by_edge[edge.name] = len(client_ids)
        active_client_ids = [
            client_id
            for client_id in client_ids
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

        client_inputs = batch['img'].to(device)
        client_labels = batch['label'].to(device)
        client_is_labeled = batch['is_labeled'].to(device).bool()
        client_weak_input = batch['weak_img'].to(device)
        client_strong_input = batch['strong_img'].to(device)
        client_upload_mb_by_edge[edge.name] = tensor_megabytes(batch['img']) + tensor_megabytes(batch['weak_img']) + tensor_megabytes(batch['strong_img'])

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

        encoded_inputs = client_output
        all_client_labels = client_labels
        all_client_is_labeled = client_is_labeled
        all_client_weak = client_weak_output
        all_client_strong = client_strong_output

        edge_classification_output = edge.run_model(encoded_inputs)
        if all_client_is_labeled.any():
            classification_loss = classification_criterion(
                edge_classification_output[all_client_is_labeled],
                all_client_labels[all_client_is_labeled],
            )
        else:
            classification_loss = scalar_zero(device)

        edge_consistency_output = edge.run_model(all_client_strong)
        cloud_consistency_output = cloud_node.run_model(all_client_weak)
        edge_probs = F.softmax(edge_consistency_output / config.train.pseudo.temperature, dim=1)
        cloud_probs = F.softmax(cloud_consistency_output / config.train.pseudo.temperature, dim=1)

        unlabeled_mask = ~all_client_is_labeled
        edge_confidence, edge_prediction = edge_probs.max(dim=1)
        cloud_confidence, cloud_prediction = cloud_probs.max(dim=1)
        agreement_mask = edge_prediction.eq(cloud_prediction)
        pseudo_weights = compute_soft_pseudo_weights(edge_confidence, cloud_confidence, agreement_mask, unlabeled_mask, config)
        effective_mask = pseudo_weights > config.train.pseudo.min_weight

        if config.train.pseudo.use and effective_mask.any():
            pseudo_loss = compute_weighted_cross_entropy(
                edge_consistency_output[effective_mask],
                cloud_prediction[effective_mask],
                pseudo_weights[effective_mask],
            )
        else:
            pseudo_loss = scalar_zero(device)

        if unlabeled_mask.any():
            consistency_loss = compute_consistency_loss(
                consistency_criterion,
                edge_probs[unlabeled_mask],
                cloud_probs[unlabeled_mask],
                config.train.consis_fn,
            )
        else:
            consistency_loss = scalar_zero(device)

        loss = classification_loss + consistency_weight * consistency_loss
        if config.train.pseudo.use:
            loss = loss + config.train.pseudo.weight * pseudo_loss
        edge.optimizer_step(loss)

        unlabeled_count = max(int(unlabeled_mask.sum().item()), 1)
        pseudo_ratio = effective_mask[unlabeled_mask].float().mean().item() if unlabeled_mask.any() else 0.0
        mean_pseudo_weight = pseudo_weights[unlabeled_mask].mean().item() if unlabeled_mask.any() else 0.0
        agreement_ratio = agreement_mask[unlabeled_mask].float().mean().item() if unlabeled_mask.any() else 0.0
        mean_edge_confidence = edge_confidence[unlabeled_mask].mean().item() if unlabeled_mask.any() else 0.0
        mean_cloud_confidence = cloud_confidence[unlabeled_mask].mean().item() if unlabeled_mask.any() else 0.0
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
            }
            edge_summaries.append(summary)
            edge_uplink_mb_by_edge[edge.name] = edge_summary_megabytes(summary)

    active_edges = max(len(class_loss), 1)
    metrics['total_loss'] = total_loss / active_edges
    metrics['avg_class_loss'] = sum(class_loss) / active_edges if class_loss else 0.0
    metrics['avg_consis_loss'] = sum(consis_loss) / active_edges if consis_loss else 0.0
    metrics['avg_pseudo_loss'] = sum(pseudo_loss_values) / active_edges if pseudo_loss_values else 0.0
    metrics['avg_pseudo_ratio'] = sum(pseudo_ratio_values) / active_edges if pseudo_ratio_values else 0.0
    metrics['avg_pseudo_weight'] = sum(pseudo_weight_values) / active_edges if pseudo_weight_values else 0.0
    metrics['avg_agreement_ratio'] = sum(agreement_ratio_values) / active_edges if agreement_ratio_values else 0.0
    metrics['avg_edge_confidence'] = sum(edge_confidence_values) / active_edges if edge_confidence_values else 0.0
    metrics['avg_cloud_confidence'] = sum(cloud_confidence_values) / active_edges if cloud_confidence_values else 0.0
    metrics['avg_unlabeled_ratio'] = sum(unlabeled_ratio_values) / active_edges if unlabeled_ratio_values else 0.0
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
    return edge_summaries if update else None, metrics, network_state


def train(config):
    device = torch.device(f"cuda:{config.train.device}" if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(config.train.device)

    transform = build_base_transform(
        config.datasets.name,
        image_size=int(config.network.client_image_size),
        normalize=OmegaConf.select(config, 'datasets.normalize', default='imagenet'),
    )

    valset = FedSemiDataset(labeled_ratio=1, train=False, transform=transform, data_name=config.datasets.name)
    valloader = DataLoader(
        valset,
        batch_size=config.datasets.batch_size,
        shuffle=False,
        num_workers=config.datasets.eval_num_workers,
        pin_memory=config.datasets.pin_memory,
    )
    encoder_model = get_encoder(config).to(device)
    graph_aggregator = HierarchicalGraphAggregator(config)

    classification_criterion, consistency_criterion = get_loss_function(
        config.train.class_fn,
        config.train.consis_fn,
        num_class=get_n_classes(config.datasets.name),
    )

    tree = create_tree(config)
    tree.move_to_device(device)
    sync_edge_models_from_cloud(tree)

    current_steps = 0
    best_accuracy = 0
    round_start_wall = time.perf_counter()
    first_target_time = None
    first_target_wall_time = None
    train_start_wall = time.perf_counter()
    cumulative_estimated_latency_s = 0.0
    cumulative_comm_mb = 0.0
    final_train_metrics = {}
    while current_steps < config.train.total_steps:
        round_start_wall = time.perf_counter()
        consistency_weight = get_warm_up_value(
            current_epoch=current_steps,
            warm_up_epochs=config.train.warm_up_steps,
            initial_weight=config.train.initial_weight,
            final_weight=config.train.final_weight,
            mode=config.train.warm_mode,
        )
        sampled_clients = tree.sample_clients(config)
        edge_summaries, train_metrics, network_state = edge_run_loop(
            tree=tree,
            sampled_clients=sampled_clients,
            encoder_model=encoder_model,
            config=config,
            consistency_weight=consistency_weight,
            classification_criterion=classification_criterion,
            consistency_criterion=consistency_criterion,
            device=device,
            update=current_steps % config.train.ema_update_step == 0,
        )

        train_metrics['consistency_weight'] = consistency_weight
        client_upload_mb = sum(network_state['client_upload_mb_by_edge'].values())
        edge_upload_mb = sum(network_state['edge_uplink_mb_by_edge'].values())
        cloud_downlink_mb = 0.0
        cloud_downlink_mb_by_edge = {}
        if edge_summaries is not None:
            if config.models.graph.use:
                aggregation_result = graph_aggregator.aggregate(edge_summaries)
                if aggregation_result['global_parameters'] is not None:
                    tree.root.model.update_by_parameters(aggregation_result['global_parameters'])
                    for edge in tree.root.children:
                        personalized_parameters = aggregation_result['personalized_parameters'].get(edge.name)
                        if personalized_parameters is not None:
                            edge.set_parameters(personalized_parameters)
                            edge_downlink_mb = parameter_list_megabytes(personalized_parameters)
                            cloud_downlink_mb_by_edge[edge.name] = edge_downlink_mb
                            cloud_downlink_mb += edge_downlink_mb
                    train_metrics.update(aggregation_result['metrics'])
            else:
                avg_parameters = average_edge_parameters(edge_summaries)
                if avg_parameters is not None:
                    tree.root.model.update_by_parameters(avg_parameters)
                    avg_parameters_mb = parameter_list_megabytes(avg_parameters)
                    for edge in tree.root.children:
                        edge.set_parameters(avg_parameters)
                        cloud_downlink_mb_by_edge[edge.name] = avg_parameters_mb
                    cloud_downlink_mb = len(tree.root.children) * avg_parameters_mb
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
        cumulative_estimated_latency_s += formal_round_latency_s
        total_comm_mb = client_upload_mb + edge_upload_mb + cloud_downlink_mb
        cumulative_comm_mb += total_comm_mb
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
            print(f"Estimated Network Latency (s): {train_metrics['estimated_network_latency_s']:.3f}")
            print(f"Formal Round Latency (s): {train_metrics['formal_round_latency_s']:.3f}")

        if (current_steps + 1) % config.train.evaluate_step == 0:
            val_metrics = evaluate(
                encoded_model=encoder_model,
                model=tree.root.model,
                val_loader=valloader,
                criterion=classification_criterion,
                device=device,
            )
            edge_val_metrics = evaluate_edges(
                encoded_model=encoder_model,
                tree=tree,
                val_loader=valloader,
                criterion=classification_criterion,
                device=device,
            )
            if first_target_time is None and val_metrics['val_accuracy'] >= config.network.target_accuracy:
                first_target_time = cumulative_estimated_latency_s
                first_target_wall_time = time.perf_counter() - train_start_wall
            info_metrics = {
                'cumulative_comm_mb': cumulative_comm_mb,
                'cumulative_estimated_latency_s': cumulative_estimated_latency_s,
                'time_to_target_accuracy_s': first_target_time if first_target_time is not None else -1.0,
                'wall_time_to_target_accuracy_s': first_target_wall_time if first_target_wall_time is not None else -1.0,
            }
            print('\nValidation Metrics:')
            print(f"Cloud Loss: {val_metrics['val_loss']:.4f}")
            print(f"Cloud Accuracy: {val_metrics['val_accuracy']:.2f}%")
            print(f"Edge Avg Accuracy: {edge_val_metrics['edge_avg_acc']:.2f}%")
            if first_target_time is not None:
                print(f"Time to Target Accuracy (s): {first_target_time:.3f}")
                print(f"Wall Time to Target Accuracy (s): {first_target_wall_time:.3f}")
            wandb.log(build_eval_log_metrics(val_metrics, edge_val_metrics, info_metrics), step=current_steps)

            if val_metrics['val_accuracy'] > best_accuracy:
                best_accuracy = val_metrics['val_accuracy']
                os.makedirs(os.path.join(config.train.ckpt_save_path, config.datasets.name), exist_ok=True)
                tree.root.model.save(os.path.join(config.train.ckpt_save_path, config.datasets.name, config.train.ckpt_save_name))

        current_steps += 1

    if wandb.run is not None:
        wandb.run.summary.update(
            build_summary_metrics(
                config=config,
                best_accuracy=best_accuracy,
                cumulative_estimated_latency_s=cumulative_estimated_latency_s,
                cumulative_comm_mb=cumulative_comm_mb,
                final_metrics=final_train_metrics,
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/formated_config.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    os.makedirs('./logs', exist_ok=True)
    swanlab.sync_wandb()
    wandb.init(
        project='EdgeQGFed',
        dir='logs',
        name=f"EdgeQGFed-aema3000-semi-{config.datasets.name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}",
        config=OmegaConf.to_container(config),
        job_type='train',
    )
    train(config)
