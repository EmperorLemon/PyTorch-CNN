from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

from collections import Counter

import os
import logging

class ImageDataset(Dataset):
    def __init__(self, 
                 root_dir : str, 
                 img_size: int = 224, 
                 mean: list = [0.485, 0.456, 0.406],
                 std: list = [0.229, 0.224, 0.225],
                 use_augmentation: bool = True):
        
        self.train_dir = os.path.join(root_dir, "train")
        self.valid_dir = os.path.join(root_dir, "valid")
        self.test_dir = os.path.join(root_dir, "test")

        self.img_size = img_size
        self.mean = mean
        self.std = std

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Pad(0),
            transforms.CenterCrop((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        for dir_path in [self.train_dir, self.valid_dir, self.test_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.transform)
        self.valid_dataset = datasets.ImageFolder(self.valid_dir, transform=self.transform)
        self.test_dataset = datasets.ImageFolder(self.test_dir, transform=self.transform)

        self.class_names = self.train_dataset.classes

        logging.info(f"Number of classes: {len(self.class_names)}")
        logging.info(f"Class names: {self.class_names}")

    def get_dataloaders(self, batch_size=32, num_workers=4):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.diagnose_dataset(train_loader=train_loader)
        self.print_dataset_info(train_loader=train_loader, val_loader=val_loader)

        return train_loader, val_loader, test_loader
    
    def diagnose_dataset(self, train_loader):
        # Check class distribution
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.numpy())
        
        # Count occurrences of each class
        label_counts = Counter(train_labels)
        
        print("Class distribution in training set:")
        for class_idx, class_name in enumerate(self.class_names):
            count = label_counts.get(class_idx, 0)  # Get count, default to 0 if class not present
            print(f"Class {class_idx} ({class_name}): {count}")

        # If you want to create a list of tuples with (class_index, class_name, count):
        class_info = [(idx, name, label_counts.get(idx, 0)) for idx, name in enumerate(self.class_names)]

        return class_info

    def print_dataset_info(self, train_loader, val_loader):
        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset)
        batch_size = train_loader.batch_size
        train_batches = len(train_loader)
        val_batches = len(val_loader)

        print(f"\nTraining samples: {train_samples}")
        print(f"Validation samples: {val_samples}")
        print(f"Batch size: {batch_size}")
        print(f"Number of training batches: {train_batches}")
        print(f"Number of validation batches: {val_batches}")