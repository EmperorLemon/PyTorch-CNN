from .model import Model

from torch import max, no_grad
from sklearn.metrics import confusion_matrix, mean_absolute_error, classification_report

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
        print(f"Model output shape: {outputs.shape}")
        print(f"Model output example:\n{outputs[0]}")
    
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

    # Get unique classes from both predictions and labels
    unique_classes = np.unique(np.concatenate((all_predictions, all_labels)))
    
    print("\nClassification Report:")
    try:
        print(classification_report(all_labels, all_predictions, 
                                    target_names=[dataset.class_names[i] for i in unique_classes]))
    except ValueError as e:
        print(f"Could not generate classification report: {e}")
        print("Generating a simplified report:")
        for class_idx in unique_classes:
            class_name = dataset.class_names[class_idx]
            true_positives = np.sum((all_labels == class_idx) & (all_predictions == class_idx))
            actual_total = np.sum(all_labels == class_idx)
            predicted_total = np.sum(all_predictions == class_idx)
            
            precision = true_positives / predicted_total if predicted_total > 0 else 0
            recall = true_positives / actual_total if actual_total > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{class_name}:")
            print(f"  Precision: {precision:.2f}")
            print(f"  Recall: {recall:.2f}")
            print(f"  F1-score: {f1:.2f}")
            print(f"  Support: {actual_total}")
            print()

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(f"Shape of confusion matrix: {conf_matrix.shape}")

    print("\nClass Distribution in Test Set:")
    for i, count in enumerate(np.bincount(all_labels, minlength=len(dataset.class_names))):
        print(f"  {dataset.class_names[i]}: {count}")

    print("\nUnique predicted classes:", unique_classes)
    print("Unique true classes:", np.unique(all_labels))

    # Mean Absolute Error
    mae = mean_absolute_error(all_labels, all_predictions)
    print(f"\nMean Absolute Error: {mae:.4f}")