from .model import Model

from torch import max, no_grad, autocast
from sklearn.metrics import confusion_matrix, mean_absolute_error

from collections import Counter
from tqdm import tqdm

import numpy as np

def evaluate_model(dataset, test_loader, model):
    model.eval()
    with no_grad(), autocast(device_type=model.device.type, enabled=True):
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            outputs = model(images.to(model.device))

            outputs = model(images)
            _, predicted = outputs.max(1)

            print(predicted)

    # conf_matrix = confusion_matrix(all_labels, all_predictions)
    # print("\nConfusion Matrix:")
    # print(conf_matrix)
    # print(f"Shape of confusion matrix: {conf_matrix.shape}")

    # # Mean Absolute Error
    # mae = mean_absolute_error(all_labels, all_predictions)
    # print(f"\nMean Absolute Error: {mae:.4f}")