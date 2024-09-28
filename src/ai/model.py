from torch import nn, cuda
from torch import device as TorchDevice
from torchvision import models
from typing import List

class MLP(nn.Module):
    def __init__(self,
                 input_size: int, 
                 hidden_layers: List[int],
                 num_classes: int,
                 device: str = "cuda" if cuda.is_available() else "cpu"):
        super(MLP, self).__init__()
        
        layers = []
        in_features = input_size
        layers.append(nn.Flatten())
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())
            
            in_features = hidden_size
        
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

        self.device = TorchDevice(device)
        self.to(self.device)

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
            self._conv_block(3, 64, 2), # 2 x conv layers (2)
            self._conv_block(64, 128, 2), # 2 x conv layers (4)
            self._conv_block(128, 256, 3), # 3 x conv layers (7)
            self._conv_block(256, 512, 3), # 3 x conv layers (10)
            self._conv_block(512, 512, 3), # 3 x conv layers (13)
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
        
    def _conv_block(self, in_channels, out_channels, num_convs):
        layers = []
        
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)


    def forward(self, X):
        x = self.encoder(X.to(self.device))
        x = self.classifier(X.to(self.device))
        
        return x
    
class PretrainedVGG16(nn.Module):
    def __init__(self, 
                 num_classes: int, 
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
        
    def forward(self, X):
        return self.vgg16(X.to(self.device))