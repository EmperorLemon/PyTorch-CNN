import matplotlib.pyplot as plt
plt.switch_backend('agg')

from torch import no_grad, autocast
from sklearn.metrics import confusion_matrix, mean_absolute_error
from tqdm import tqdm

from tensorboardX import SummaryWriter

import numpy as np
import seaborn as sns

def evaluate_model(model, test_loader, writer: SummaryWriter):
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
    
    # Log to TensorBoardX
    writer.add_scalar("Test/MAE", mae)

    # Create and log confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", square=True, cbar=True, ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    writer.add_figure("Confusion Matrix", fig)
    plt.close(fig)