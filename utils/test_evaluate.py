import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def test_evaluate(encoded_model, model, val_loader, device, process=False):
    encoded_model.to(device)
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(val_loader) if process else val_loader:
            inputs, labels = batch['img'], batch['label']
            inputs, labels = inputs.to(device), labels.to(device)

            encoded_inputs = encoded_model(inputs)
            outputs = model(encoded_inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect all labels and predictions for metrics calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Convert collected data to NumPy arrays for scikit-learn compatibility
    all_labels_np = np.array(all_labels)
    all_predictions_np = np.array(all_predictions)

    # Calculate F1 Score
    f1 = f1_score(all_labels_np, all_predictions_np, average='macro')

    # Binarize labels for AUC calculation
    y_test_binarized = label_binarize(all_labels_np, classes=np.arange(10))
    y_pred_binarized = label_binarize(all_predictions_np, classes=np.arange(10))

    # Calculate AUC-ROC
    all_auc = roc_auc_score(y_test_binarized, y_pred_binarized, average="macro")

    accuracy = 100 * correct / total

    return {'val_f1': f1, 'val_accuracy': accuracy, 'val_auc': all_auc}
