from typing import Dict, List

import torch
import torch.nn.functional as F


class HierarchicalGraphAggregator:
    def __init__(self, config):
        graph_config = config.models.graph
        self.temperature = float(graph_config.temperature)
        self.prototype_weight = float(graph_config.prototype_weight)
        self.reliability_weight = float(graph_config.reliability_weight)
        self.label_ratio_weight = float(graph_config.label_ratio_weight)
        self.confidence_weight = float(graph_config.confidence_weight)
        self.self_bias = float(graph_config.self_bias)
        self.min_attention = float(graph_config.min_attention)

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

    def aggregate(self, edge_summaries: List[Dict]) -> Dict[str, object]:
        if not edge_summaries:
            return {
                "personalized_parameters": {},
                "global_parameters": None,
                "attention_matrix": None,
                "metrics": {},
            }

        device = edge_summaries[0]["parameters"][0].device
        edge_num = len(edge_summaries)
        parameter_signatures = [
            self._parameter_signature(summary["parameters"]).to(device)
            for summary in edge_summaries
        ]
        prototype_signatures = [self._prototype_signature(summary).to(device) for summary in edge_summaries]

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

        attention_scores = torch.zeros(edge_num, edge_num, device=device)
        for i in range(edge_num):
            for j in range(edge_num):
                parameter_similarity = self._safe_cosine(parameter_signatures[i], parameter_signatures[j])
                prototype_similarity = self._safe_cosine(prototype_signatures[i], prototype_signatures[j])
                reliability_similarity = 1.0 - torch.abs(reliability[i] - reliability[j])
                label_ratio_similarity = 1.0 - torch.abs(label_ratio[i] - label_ratio[j])
                confidence_similarity = 1.0 - torch.abs(confidence[i] - confidence[j])

                score = parameter_similarity
                score = score + self.prototype_weight * prototype_similarity
                score = score + self.reliability_weight * reliability_similarity
                score = score + self.label_ratio_weight * label_ratio_similarity
                score = score + self.confidence_weight * confidence_similarity
                if i == j:
                    score = score + self.self_bias
                attention_scores[i, j] = score

        attention_matrix = F.softmax(attention_scores / self.temperature, dim=1)
        attention_matrix = torch.clamp(attention_matrix, min=self.min_attention)
        attention_matrix = attention_matrix / attention_matrix.sum(dim=1, keepdim=True)

        personalized_parameters = {}
        for target_index, summary in enumerate(edge_summaries):
            aggregated_parameters = []
            for param_index in range(len(summary["parameters"])):
                mixed_param = torch.zeros_like(summary["parameters"][param_index])
                for source_index, source_summary in enumerate(edge_summaries):
                    mixed_param = mixed_param + attention_matrix[target_index, source_index] * source_summary["parameters"][param_index]
                aggregated_parameters.append(mixed_param.clone())
            personalized_parameters[summary["name"]] = aggregated_parameters

        global_parameters = []
        global_weights = (batch_weight / batch_weight.sum()) * (reliability + 1.0)
        global_weights = global_weights / global_weights.sum()
        for param_index in range(len(edge_summaries[0]["parameters"])):
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
            "personalized_parameters": personalized_parameters,
            "global_parameters": global_parameters,
            "attention_matrix": attention_matrix.detach().cpu(),
            "metrics": metrics,
        }