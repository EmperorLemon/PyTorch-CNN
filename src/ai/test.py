from .model import Model

from torch import max, no_grad
from sklearn.metrics import confusion_matrix, mean_absolute_error

import numpy as np

def evaluate_model(test_data, model: Model):
    all_predictions = []
    all_labels = []

    model.eval()
    with no_grad():
        for images, labels in test_data:
            images = images.to(model.device)
            outputs = model(images)
            _, predicted = max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Mean Absolute Error
    mae = mean_absolute_error(all_labels, all_predictions)
    print(f"Mean Absolute Error: {mae:.4f}")