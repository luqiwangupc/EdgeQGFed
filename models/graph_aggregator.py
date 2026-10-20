from typing import Dict, List

import math

import torch
import torch.nn.functional as F


class HierarchicalGraphAggregator:
    GRAPH_MODES = {'model_graph', 'mp_graph', 'mpq_graph'}
    AVERAGE_MODES = {'uniform_average', 'sample_average', 'quality_average'}

    def __init__(self, config):
        graph_config = config.models.graph
        # No explicit mode: preserve old graph.use behaviour (including ES-equal average).
        mode = graph_config.get('mode')
        self.mode = str(mode).lower() if mode is not None else (
            'mpq_graph' if graph_config.use else 'uniform_average'
        )
        if self.mode not in self.GRAPH_MODES | self.AVERAGE_MODES:
            raise ValueError(f'Unknown aggregation mode: {self.mode}')
        default_teacher = 'sample_quality' if self.mode in self.GRAPH_MODES else 'direct'
        self.teacher_weight_mode = str(graph_config.get('teacher_weight_mode', default_teacher)).lower()
        allowed_teacher = {'sample', 'sample_quality'} if self.mode in self.GRAPH_MODES else {'direct'}
        if self.teacher_weight_mode not in allowed_teacher:
            raise ValueError(f'{self.mode} requires teacher_weight_mode in {sorted(allowed_teacher)}')
        self.temperature = float(graph_config.temperature)
        self.prototype_weight = float(graph_config.prototype_weight)
        self.reliability_weight = float(graph_config.reliability_weight)
        self.label_ratio_weight = float(graph_config.label_ratio_weight)
        self.confidence_weight = float(graph_config.confidence_weight)
        self.self_bias = float(graph_config.self_bias)
        self.min_attention = float(graph_config.min_attention)
        settings = (self.temperature, self.prototype_weight, self.reliability_weight,
                    self.label_ratio_weight, self.confidence_weight, self.self_bias,
                    self.min_attention)
        if not all(math.isfinite(value) for value in settings):
            raise ValueError('Aggregation settings must be finite')
        if self.temperature <= 0 or not 0 <= self.min_attention <= 1:
            raise ValueError('temperature must be positive and min_attention must be in [0, 1]')

    @staticmethod
    def _normalize_weights(weights):
        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('Source weights must be finite, nonnegative and have positive total')
        return weights / weights.sum()

    @staticmethod
    def _parameter_signature(parameters: List[torch.Tensor]) -> torch.Tensor:
        signature = []
        for param in parameters:
            tensor = param.detach().float().view(-1)
            if tensor.numel() == 0:
                continue
            signature.extend([
                tensor.mean(),
                tensor.std(unbiased=False),
                tensor.norm() / (tensor.numel() ** 0.5),
            ])
        return torch.stack(signature)

    @staticmethod
    def _safe_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = x.norm(p=2)
        y_norm = y.norm(p=2)
        if x_norm.item() == 0 or y_norm.item() == 0:
            return torch.tensor(0.0, device=x.device)
        return torch.clamp(torch.dot(x / x_norm, y / y_norm), min=-1.0, max=1.0)

    @staticmethod
    def _prototype_signature(summary: Dict) -> torch.Tensor:
        prototype = summary["prototype"].detach().float()
        counts = summary["prototype_counts"].detach().float().unsqueeze(1)
        weighted_prototype = prototype * counts
        prototype_vector = weighted_prototype.view(-1)
        if prototype_vector.numel() == 0 or prototype_vector.abs().sum().item() == 0:
            return torch.zeros_like(prototype_vector)
        return F.normalize(prototype_vector, dim=0)

    @torch.no_grad()
    def aggregate(self, edge_summaries: List[Dict]) -> Dict[str, object]:
        if not edge_summaries:
            return {
                "mode": self.mode,
                "teacher_weight_mode": self.teacher_weight_mode,
                "active_edge_ids": [],
                "personalized_parameters": {},
                "global_parameters": None,
                "attention_matrix": None,
                "teacher_weights": None,
                "source_weights": None,
                "metrics": {},
            }

        names = [summary['name'] for summary in edge_summaries]
        if len(set(names)) != len(names):
            raise ValueError('Edge summary names must be unique')
        device = edge_summaries[0]["parameters"][0].device
        edge_num = len(edge_summaries)
        reliability = torch.tensor(
            [summary["pseudo_quality"] for summary in edge_summaries],
            dtype=torch.float32,
            device=device,
        )
        label_ratio = torch.tensor(
            [summary["labeled_ratio"] for summary in edge_summaries],
            dtype=torch.float32,
            device=device,
        )
        confidence = torch.tensor(
            [summary["mean_confidence"] for summary in edge_summaries],
            dtype=torch.float32,
            device=device,
        )
        batch_weight = torch.tensor(
            [summary["batch_size"] for summary in edge_summaries],
            dtype=torch.float32,
            device=device,
        )
        if not torch.isfinite(reliability).all() or ((reliability < 0) | (reliability > 1)).any():
            raise ValueError('pseudo_quality must be finite and in [0, 1]')
        sample_weights = self._normalize_weights(batch_weight)

        if self.mode in self.AVERAGE_MODES:
            if self.mode == 'uniform_average':
                source_weights = torch.ones_like(batch_weight) / edge_num
            elif self.mode == 'sample_average':
                source_weights = sample_weights
            else:
                source_weights = self._normalize_weights(batch_weight * (1.0 + reliability))
            # Each recipient receives the same direct source mixture. No second N/rho stage.
            attention_matrix = source_weights.unsqueeze(0).repeat(edge_num, 1)
            global_weights = torch.ones_like(batch_weight) / edge_num
        else:
            parameter_signatures = [self._parameter_signature(s['parameters']) for s in edge_summaries]
            prototype_signatures = [self._prototype_signature(s).to(device) for s in edge_summaries]
            attention_matrix = self._attention(
                parameter_signatures, prototype_signatures, reliability, label_ratio, confidence,
            )
            global_weights = sample_weights
            if self.teacher_weight_mode == 'sample_quality':
                global_weights = self._normalize_weights(sample_weights * (1.0 + reliability))
            source_weights = global_weights @ attention_matrix

        personalized_parameters = {}
        for target_index, summary in enumerate(edge_summaries):
            if self.mode in self.AVERAGE_MODES and target_index > 0:
                personalized_parameters[summary['name']] = [
                    tensor.clone() for tensor in personalized_parameters[names[0]]
                ]
                continue
            aggregated_parameters = []
            for param_index in range(len(summary["parameters"])):
                if not (summary["parameters"][param_index].is_floating_point() or summary["parameters"][param_index].is_complex()):
                    aggregated_parameters.append(summary["parameters"][param_index].clone())
                    continue
                mixed_param = torch.zeros_like(summary["parameters"][param_index])
                if self.mode == 'uniform_average':
                    # Match the original stack(...).mean(...) arithmetic for legacy runs.
                    mixed_param = torch.stack([s['parameters'][param_index] for s in edge_summaries]).mean(dim=0)
                else:
                    for source_index, source_summary in enumerate(edge_summaries):
                        mixed_param = mixed_param + attention_matrix[target_index, source_index] * source_summary["parameters"][param_index]
                aggregated_parameters.append(mixed_param.clone())
            personalized_parameters[summary["name"]] = aggregated_parameters

        if self.mode in self.AVERAGE_MODES:
            global_parameters = [tensor.clone() for tensor in personalized_parameters[names[0]]]
            # Integer buffers: all average recipients use first active source; graph
            # recipients retain local values and the cloud uses first active source.
            for parameters in personalized_parameters.values():
                for index, tensor in enumerate(global_parameters):
                    if not (tensor.is_floating_point() or tensor.is_complex()):
                        parameters[index] = tensor.clone()
        else:
            global_parameters = []
            for param_index in range(len(edge_summaries[0]["parameters"])):
                if not (edge_summaries[0]["parameters"][param_index].is_floating_point() or edge_summaries[0]["parameters"][param_index].is_complex()):
                    global_parameters.append(edge_summaries[0]["parameters"][param_index].clone())
                    continue
                global_param = torch.zeros_like(edge_summaries[0]["parameters"][param_index])
                for edge_index, summary in enumerate(edge_summaries):
                    global_param = global_param + global_weights[edge_index] * personalized_parameters[summary["name"]][param_index]
                global_parameters.append(global_param.clone())

        metrics = {
            "graph_attention_mean": attention_matrix.mean().item(),
            "graph_attention_diag": attention_matrix.diag().mean().item(),
            "graph_reliability_mean": reliability.mean().item(),
            "graph_label_ratio_mean": label_ratio.mean().item(),
            "graph_confidence_mean": confidence.mean().item(),
        }
        return {
            "mode": self.mode,
            "teacher_weight_mode": self.teacher_weight_mode,
            "active_edge_ids": names,
            "personalized_parameters": personalized_parameters,
            "global_parameters": global_parameters,
            "attention_matrix": attention_matrix.detach().cpu(),
            "teacher_weights": global_weights.detach().cpu(),
            "source_weights": source_weights.detach().cpu(),
            "metrics": metrics,
        }

    def _attention(self, parameter_signatures, prototype_signatures, reliability, label_ratio, confidence):
        edge_num = len(parameter_signatures)
        device = reliability.device

        attention_scores = torch.zeros(edge_num, edge_num, device=device)
        for i in range(edge_num):
            for j in range(edge_num):
                parameter_similarity = self._safe_cosine(parameter_signatures[i], parameter_signatures[j])
                prototype_similarity = self._safe_cosine(prototype_signatures[i], prototype_signatures[j])
                reliability_similarity = 1.0 - torch.abs(reliability[i] - reliability[j])
                label_ratio_similarity = 1.0 - torch.abs(label_ratio[i] - label_ratio[j])
                confidence_similarity = 1.0 - torch.abs(confidence[i] - confidence[j])

                score = parameter_similarity
                if self.mode != 'model_graph':
                    score = score + self.prototype_weight * prototype_similarity
                if self.mode == 'mpq_graph':
                    score = score + self.reliability_weight * reliability_similarity
                    score = score + self.label_ratio_weight * label_ratio_similarity
                    score = score + self.confidence_weight * confidence_similarity
                if i == j:
                    score = score + self.self_bias
                attention_scores[i, j] = score

        attention_matrix = F.softmax(attention_scores / self.temperature, dim=1)
        attention_matrix = torch.clamp(attention_matrix, min=self.min_attention)
        attention_matrix = attention_matrix / attention_matrix.sum(dim=1, keepdim=True)

        return attention_matrix
