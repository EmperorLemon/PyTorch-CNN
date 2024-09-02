from torch.utils.data import Dataset

from torchvision import transforms, datasets

import os

class ImageDataset(Dataset):
    def __init__(self, root_dir : str):
        self.train_dir = os.path.join(root_dir, "train")
        self.valid_dir = os.path.join(root_dir, "valid")
        self.test_dir = os.path.join(root_dir, "test")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        print(self.train_dir)
        print(self.valid_dir)
        print(self.test_dir)

        #self.train_dataset = datasets.ImageFolder()