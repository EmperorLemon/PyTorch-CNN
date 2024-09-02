from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from typing import Optional, Callable

class ImageDataset(Dataset):
    def __init__(self, root_dir : str):
        