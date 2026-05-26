import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer('weight', weight if weight is not None else None)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1.0 - pt) ** self.gamma * ce_loss).mean()


def get_loss_function(class_cri: str, consis_cri: str, num_class: int = None, class_weights=None, focal_gamma=2.0):
    class_cri = class_cri.lower()
    consis_cri = consis_cri.lower()

    if class_cri == "ce":
        classification_criterion = nn.CrossEntropyLoss()
    elif class_cri == "weighted_ce":
        classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif class_cri in {"focal", "weighted_focal"}:
        classification_criterion = FocalLoss(weight=class_weights, gamma=focal_gamma)
    elif class_cri == "encrypt":
        raise NotImplementedError("Encrypted classification loss is currently disabled.")
    else:
        raise ValueError(f"Unknown classification criterion: {class_cri}")

    if consis_cri == "mse":
        consistency_criterion = nn.MSELoss()
    elif consis_cri == "l1":
        consistency_criterion = nn.L1Loss()
    elif consis_cri == "kl":
        consistency_criterion = nn.KLDivLoss(reduction="batchmean")
    else:
        raise ValueError(f"Unknown consistency criterion: {consis_cri}")

    return classification_criterion, consistency_criterion
