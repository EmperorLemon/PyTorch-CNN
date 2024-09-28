from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from collections import Counter
from sklearn.model_selection import train_test_split

import torch

class ImageDataset(Dataset):
    def __init__(self, 
                 root_dir : str, 
                 img_size: int = 224, 
                 mean: list = [0.485, 0.456, 0.406],
                 std: list = [0.229, 0.224, 0.225],
                 val_split: float = 0.2,
                 test_split: float = 0.1,
                 random_seed: int = 42):
        
        self.root_dir = root_dir

        self.img_size = img_size
        self.mean = mean
        self.std = std

        self.val_split = val_split
        self.test_split = test_split

        self.random_seed = random_seed
        
        self._compose_transforms()

        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)

        self.full_dataset = datasets.ImageFolder(self.root_dir)
        self.train_indices, self.val_indices, self.test_indices = self._stratified_split()

        self.train_dataset = Subset(self.full_dataset, self.train_indices)
        self.train_dataset.dataset.transform = self.train_transform

        self.val_dataset = Subset(self.full_dataset, self.val_indices)
        self.val_dataset.dataset.transform = self.eval_transform

        self.test_dataset = Subset(self.full_dataset, self.test_indices)
        self.test_dataset.dataset.transform = self.eval_transform

        self.class_names = self.full_dataset.classes

        self.diagnose_dataset()
        
    def _compose_transforms(self):
        # Training transforms
        self.train_transform = transforms.Compose([
            # Resize the image to a fixed size
            transforms.Resize((self.img_size, self.img_size)),
            # Randomly flip the image horizontally
            transforms.RandomHorizontalFlip(),  
            # Randomly rotate the image by up to 20 degrees
            transforms.RandomRotation(20),  
            # Randomly change the brightness, contrast, and saturation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
            # Convert the image to a PyTorch tensor
            transforms.ToTensor(),
            # Normalize the image (using ImageNet stats)
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Validation and Test transforms (no augmentation)
        self.eval_transform = transforms.Compose([
            # Resize the image to a fixed size
            transforms.Resize((self.img_size, self.img_size)),
            # Convert the image to a PyTorch tensor
            transforms.ToTensor(),
            # Normalize the image (using ImageNet stats)
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def _stratified_split(self):
        labels = [label for _, label in self.full_dataset.samples]
        
        # First, split off the test set
        train_val_indices, test_indices = train_test_split(
            range(len(self.full_dataset)),
            test_size=self.test_split,
            stratify=labels,
            random_state=self.random_seed
        )
        
        # Then split the remaining data into train and validation sets
        train_val_labels = [labels[i] for i in train_val_indices]
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=self.val_split / (1 - self.test_split),
            stratify=train_val_labels,
            random_state=self.random_seed
        )
        
        return train_indices, val_indices, test_indices

    def get_dataloaders(self, batch_size=32, num_workers=4):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return train_loader, val_loader, test_loader
    
    def diagnose_dataset(self):
        train_labels = [self.full_dataset.targets[i] for i in self.train_indices]
        val_labels = [self.full_dataset.targets[i] for i in self.val_indices]
        test_labels = [self.full_dataset.targets[i] for i in self.test_indices]

        print("\nDataset Summary:\n")
        print(f"Total samples: {len(self.full_dataset)}")
        print(f"Training samples: {len(self.train_dataset)} ({len(self.train_dataset) / len(self.full_dataset) * 100.0:.2f}%)")
        print(f"Validation samples: {len(self.val_dataset)} ({len(self.val_dataset) / len(self.full_dataset) * 100.0:.2f}%)")
        print(f"Test samples: {len(self.test_dataset)} ({len(self.test_dataset) / len(self.full_dataset) * 100.0:.2f}%)")

        print("\nClass distribution:")
        for split_name, labels in [("Training", train_labels), ("Validation", val_labels), ("Test", test_labels)]:
            label_counts = Counter(labels)
            print(f"\n{split_name} set:")
            for class_idx, class_name in enumerate(self.class_names):
                count = label_counts.get(class_idx, 0)
                percentage = (count / len(labels)) * 100
                print(f"Class {class_idx} ({class_name}): {count} ({percentage:.2f}%)")