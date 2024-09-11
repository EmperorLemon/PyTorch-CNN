from .model import Model

from torch import no_grad, autocast
from sklearn.metrics import confusion_matrix, mean_absolute_error

from tqdm import tqdm

from globals import OUTPUT_DIR

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import random

def evaluate_model(dataset, test_loader, model: Model):
    all_predictions = []
    all_labels = []

    num_samples = 10
    
    model.eval()
    # Evaluation on the entire test set
    with no_grad(), autocast(device_type=model.device.type, enabled=True):
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

    _, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Convert mean and std to tensors
    mean = torch.tensor(dataset.mean).view(3, 1, 1)
    std = torch.tensor(dataset.std).view(3, 1, 1)
    
    # Convert data_loader to iterator
    data_iter = iter(test_loader)
    
    samples_visualized = 0
    while samples_visualized < num_samples:
        try:
            # Get a batch of data
            images, labels = next(data_iter)
        except StopIteration:
            # If we've gone through all batches, start over
            data_iter = iter(test_loader)
            continue
        
        # Randomly select images from this batch
        for _ in range(min(num_samples - samples_visualized, len(images))):
            idx = random.randint(0, len(images) - 1)
            img = images[idx].unsqueeze(0).to(model.device)
            label = labels[idx].item()
            
            with torch.no_grad():
                output = model(img)
            
            pred = output.argmax(dim=1).item()

            class_names = dataset.class_names
            
            # Denormalize the image
            denormalized_image = img.cpu() * std + mean
            
            # Ensure the image is in the correct range [0, 1]
            denormalized_image = torch.clamp(denormalized_image, 0, 1)
            
            # Convert to numpy for plotting
            img_np = denormalized_image.squeeze().permute(1, 2, 0).numpy()

            axes[samples_visualized].imshow(img_np)
            axes[samples_visualized].set_title(f"True: {class_names[label]}\nPred: {class_names[pred]}")
            axes[samples_visualized].axis("off")
            
            samples_visualized += 1
            if samples_visualized == num_samples:
                break
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "results.png"))
    
    print("\nClassification results visualization saved as 'results.png'")