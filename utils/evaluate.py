import torch
from tqdm import tqdm


def _classification_metrics(labels, predictions):
    if not labels:
        return {}
    labels = torch.tensor(labels, dtype=torch.long)
    predictions = torch.tensor(predictions, dtype=torch.long)
    num_classes = int(max(labels.max().item(), predictions.max().item())) + 1
    recalls = []
    f1_scores = []
    supports = []
    metrics = {}
    for class_id in range(num_classes):
        true_positive = ((predictions == class_id) & (labels == class_id)).sum().item()
        false_positive = ((predictions == class_id) & (labels != class_id)).sum().item()
        false_negative = ((predictions != class_id) & (labels == class_id)).sum().item()
        support = (labels == class_id).sum().item()
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
        metrics[f'val_recall_class_{class_id}'] = recall

    total_support = max(sum(supports), 1)
    metrics['val_macro_f1'] = sum(f1_scores) / max(num_classes, 1)
    metrics['val_weighted_f1'] = sum(f1 * support for f1, support in zip(f1_scores, supports)) / total_support
    metrics['val_macro_recall'] = sum(recalls) / max(num_classes, 1)
    return metrics


def evaluate(encoded_model, model, val_loader, criterion, device, process=False, extra_metrics=False):
    encoded_model.to(device)
    model.to(device)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(val_loader) if process else val_loader:
            inputs, labels = batch['img'], batch['label']
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            encoded_inputs = encoded_model(inputs)

            outputs = model(encoded_inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if extra_metrics:
                all_labels.extend(labels.detach().cpu().tolist())
                all_predictions.extend(predicted.detach().cpu().tolist())

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)

    metrics = {'val_loss': avg_loss, 'val_accuracy': accuracy}
    if extra_metrics:
        metrics.update(_classification_metrics(all_labels, all_predictions))
    return metrics
