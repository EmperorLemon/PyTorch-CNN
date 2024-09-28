from torch import nn, cuda
from torch import device as TorchDevice
from torchvision import models
from collections import OrderedDict
from typing import List, Any, NamedTuple

## A network model
class Model(nn.Module):
    def __init__(self, 
                 layers: nn.Sequential,
                 device: str = "cuda" if cuda.is_available() else "cpu"):
        super(Model, self).__init__()

        self.device = TorchDevice(device)
        self.to(self.device)

        self.net = layers

    ## Forward step
    def forward(self, X):
        # Compute output given an input X
        return self.net(X.to(self.device))
    
class VGG16(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 device: str = "cuda" if cuda.is_available() else "cpu"):
        super(VGG16, self).__init__()
        
        self.encoder = nn.Sequential(
            # Block 1
                # Conv layer 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
                # Conv layer 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
                # Conv layer 1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
                # Conv layer 2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
                # Conv layer 1
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
                # Conv layer 2
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
                # Conv layer 3
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
                # Conv layer 1
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
                # Conv layer 2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
                # Conv layer 3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
                # Conv layer 1
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
                # Conv layer 2
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
                # Conv layer 3
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        
        self.classifier = nn.Sequential(
            nn.Flatten(),    
                # FC layer 1
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
                # FC layer 2
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
                # FC layer 3 (Output layer)
            nn.Linear(4096, num_classes),
        )

        self.device = TorchDevice(device)
        self.to(self.device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
class PretrainedVGG16(nn.Module):
    def __init__(self, num_classes, 
                 freeze_features: bool = True, 
                 device: str = "cuda" if cuda.is_available() else "cpu"):
        super(PretrainedVGG16, self).__init__()
        
        # Load pretrained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)
        
        if freeze_features:
            # Freeze the features layers
            for param in self.vgg16.features.parameters():
                param.requires_grad = False
        
        # Replace the last fully connected layer
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)

        self.device = TorchDevice(device)
        self.to(self.device)
        
    def forward(self, x):
        return self.vgg16(x)