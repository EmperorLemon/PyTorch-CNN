from torch import no_grad, autocast
from sklearn.metrics import confusion_matrix, mean_absolute_error
from tqdm import tqdm
import numpy as np

def evaluate_model(test_loader, model):
    all_predictions = []
    all_labels = []
    
    model.eval()    
    with no_grad(), autocast(device_type=model.device.type, enabled=True):
        # Evaluation on the entire test set
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            outputs = model(images.to(model.device))
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute metrics
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)

    # Print results
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(f"Shape of confusion matrix: {conf_matrix.shape}")

    print(f"\nMean Absolute Error: {mae:.4f}")