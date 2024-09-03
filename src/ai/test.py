from .model import Model

from torch import max, no_grad
from sklearn.metrics import confusion_matrix, mean_absolute_error

from collections import Counter

import numpy as np

def verify_data_and_model(dataset, test_loader, model: Model):
    # Check test set size and class distribution
    test_data = test_loader.dataset
    print(f"Number of test samples: {len(test_data)}")
    
    all_labels = [label for _, label in test_data]
    print(f"Class distribution in test set: {Counter(all_labels)}")
    
    # Verify model output
    model.eval()
    with no_grad():
        images, _ = next(iter(test_loader))
        outputs = model(images.to(model.device))
    
    # Check class mapping
    print(f"Class names: {dataset.class_names}")
    print(f"Number of classes: {len(dataset.class_names)}")

def evaluate_model(dataset, test_loader, model):
    all_predictions = []
    all_labels = []
    
    model.eval()
    with no_grad():
        for images, labels in test_loader:
            images = images.to(model.device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(f"Shape of confusion matrix: {conf_matrix.shape}")

    print("\nClass Distribution in Test Set:")
    for i, count in enumerate(np.bincount(all_labels, minlength=len(dataset.class_names))):
        print(f"  {dataset.class_names[i]}: {count}")

    # Mean Absolute Error
    mae = mean_absolute_error(all_labels, all_predictions)
    print(f"\nMean Absolute Error: {mae:.4f}")