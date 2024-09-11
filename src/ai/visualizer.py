from .model import Model
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import numpy as np

def visualize_results(test_loader, model: Model, writer, output_dir: str):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    img_grid = make_grid(images)

    # Normalize grid from [-1, 1] to [0, 1]
    img_grid = img_grid / 2 + 0.5

    writer.add_image(output_dir, img_grid)