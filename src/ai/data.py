from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from typing import Optional, Callable

class ImageDataset(Dataset):
    def __init__(self, data_path : str, transform: Optional[Callable]=None):
        pass