from .model import Model
from torch import no_grad

import matplotlib.pyplot as plt
import random

import os

def visualize_results(dataset, model: Model, output_dir: str, num_samples: int=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        idx = random.randint(0, len(dataset)-1)
        img, label = dataset[idx]
        img = img.unsqueeze(0).to(model.device)
        
        with no_grad():
            output = model(img)
        
        pred = output.argmax(dim=1).item()

        axes[i].imshow(img.cpu().squeeze().numpy(), cmap="gray")
        axes[i].set_title(f"True: {label}, Pred: {pred}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join(output_dir, "results.png")))
    
    print("\nClassification results visualization saved as 'results.png'")