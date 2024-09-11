from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import datasets, transforms

from collections import Counter

import os
import numpy as np

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

        # Training transforms
        self.train_transform = transforms.Compose([
            # Resize the image to a fixed size
            transforms.Resize((self.img_size, self.img_size)),
            # Randomly flip the image horizontally
            transforms.RandomHorizontalFlip(p=0.5),  
            # Randomly rotate the image by up to 10 degrees
            transforms.RandomRotation(10),  
            # Randomly change the brightness, contrast, and saturation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
            # Convert the image to a PyTorch tensor
            transforms.ToTensor(),
            # Normalize the image (using ImageNet stats)
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Validation transforms
        self.valid_transform = transforms.Compose([
            # Resize the image to a fixed size
            transforms.Resize((self.img_size, self.img_size)),
            # Convert the image to a PyTorch tensor
            transforms.ToTensor(),
            # Normalize the image (using ImageNet stats)
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.test_transform = self.valid_transform

        for dir_path in [self.train_dir, self.valid_dir, self.test_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")

        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        self.valid_dataset = datasets.ImageFolder(self.valid_dir, transform=self.valid_transform)
        self.test_dataset = datasets.ImageFolder(self.test_dir, transform=self.test_transform)

        self.class_names = self.train_dataset.classes
        self.diagnose_dataset(train_dataset=self.train_dataset, val_dataset=self.valid_dataset, test_dataset=self.test_dataset)

    def get_dataloaders(self, batch_size=32, num_workers=4):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return train_loader, val_loader, test_loader
    
    def get_labels(self, dataset, max_samples=10000):
        if hasattr(dataset, 'targets'):
            return dataset.targets
        elif isinstance(dataset, Subset):
            return [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            # Sample a subset of the data if it's too large
            indices = np.random.choice(len(dataset), min(max_samples, len(dataset)), replace=False)
            return [dataset[i][1] for i in indices]
    
    def diagnose_dataset(self, train_dataset, val_dataset, test_dataset):
        datasets = {
            'Training': train_dataset,
            'Validation': val_dataset,
            'Test': test_dataset
        }

        all_info = {}

        for split_name, dataset in datasets.items():
            labels = self.get_labels(dataset)
            label_counts = Counter(labels)
            
            print(f"\nClass distribution in {split_name} set:")
            class_info = []
            for class_idx, class_name in enumerate(self.class_names):
                count = label_counts.get(class_idx, 0)
                percentage = (count / len(labels)) * 100 if len(labels) > 0 else 0
                print(f"Class {class_idx} ({class_name}): {count} ({percentage:.2f}%)")
                class_info.append((class_idx, class_name, count, percentage))
            
            all_info[split_name] = {
                'samples': len(dataset),
                'class_info': class_info
            }

        self.print_dataset_summary(all_info)

        return all_info

    def print_dataset_summary(self, all_info):
        print("\nDataset Summary:")
        for split_name, info in all_info.items():
            print(f"\n{split_name} set:")
            print(f"  Total samples: {info['samples']}")