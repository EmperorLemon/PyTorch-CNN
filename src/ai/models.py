from .model import nn
from typing import Type, List, Union, Tuple

def linear_block(in_features: int, 
                 out_features: int, 
                 activation: nn.Module = nn.ReLU(),
                 dropout_rate: float = 0.5):
    
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features),
        activation,
        nn.Dropout(dropout_rate)
    )

def conv_block(in_channels: int,
               out_channels: int,
               kernel_size: Tuple[int, int] = (3, 3),
               padding: Union[int, Tuple[int, int]] = 1,
               pool_size: Tuple[int, int] = (2, 2),
               use_pooling: bool = True,
               activation: nn.Module = nn.ReLU(),
               dropout_rate: float = 0.5):
    
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        activation
    ]
    
    if use_pooling:
        layers.append(nn.MaxPool2d(pool_size))
    
    layers.append(nn.Dropout2d(dropout_rate))
    
    return nn.Sequential(*layers)

def create_mlp(input_size: int,
               fc_layers: List[int],
               num_classes: int,
               dropout_rate: float = 0.5) -> List[nn.Module]:
    layers = []

    # Input layer
    layers.append(nn.Flatten()) # Flatten the 3D input to 1D
    
    layers.append(nn.Linear(in_features=input_size, out_features=fc_layers[0]))
    layers.append(nn.LayerNorm(fc_layers[0]))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))

    layers.append(nn.Linear(in_features=fc_layers[0], out_features=fc_layers[1]))
    layers.append(nn.LayerNorm(fc_layers[1]))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))

    layers.append(nn.Linear(in_features=fc_layers[-1], out_features=num_classes))

    return layers

def create_cnn(in_channels: int,
               conv_layers: List[int],
               fc_layers: List[int],
               num_classes: int,
               dropout_rate: 0.5) -> List[nn.Module]:
    
    layers = []

    # Convolutional layers
    for i, out_channels in enumerate(conv_layers):
        layers.append(conv_block(
            in_channels=in_channels if i == 0 else conv_layers[i - 1],
            out_channels=out_channels,
            dropout_rate=dropout_rate
        ))

    # Flatten layer
    layers.append(nn.Flatten())

    # Fully connected layer
