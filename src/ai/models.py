from .model import nn
from typing import Type, List, Union, Tuple

def linear_layer(in_features: int, 
                 out_features: int, 
                 activation: nn.Module = nn.ReLU(),
                 dropout_rate: float = 0.5):
    
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features),
        nn.LayerNorm(out_features),
        activation,
        nn.Dropout(dropout_rate)
    )

def conv_layer(in_channels: int,
               out_channels: int,
               kernel_size: Tuple[int, int] = (3, 3),
               padding: Union[int, Tuple[int, int]] = 1,
               activation: nn.Module = nn.ReLU(),
               use_pooling: bool = True,
               pool_size: Tuple[int, int] = (2, 2),
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

    layers = nn.Sequential(
        nn.Flatten(),
        linear_layer(in_features=input_size, out_features=fc_layers[0]),
        linear_layer(in_features=fc_layers[0], out_features=fc_layers[1]),
        nn.Linear(in_features=fc_layers[-1], out_features=num_classes)
    )
    
    # layers = []

    # # Input layer
    # layers.append(nn.Flatten()) # Flatten the 3D input to 1D
    
    # layers.append(nn.Linear(in_features=input_size, out_features=fc_layers[0]))
    # layers.append(nn.LayerNorm(fc_layers[0]))
    # layers.append(nn.ReLU())
    # layers.append(nn.Dropout(dropout_rate))

    # layers.append(nn.Linear(in_features=fc_layers[0], out_features=fc_layers[1]))
    # layers.append(nn.LayerNorm(fc_layers[1]))
    # layers.append(nn.ReLU())
    # layers.append(nn.Dropout(dropout_rate))

    # # Output Layer
    # layers.append(nn.Linear(in_features=fc_layers[-1], out_features=num_classes))

    # return nn.Sequential(*layers)
    return layers

def create_vgg16(num_classes: int):
    # layers = nn.Sequential(
    #     vgg_layer(in_channels=3, conv_block=[64, 64]), # 2 conv layers (total 2)
    #     vgg_layer(in_channels=64, conv_block=[128, 128]), # 2 conv layers (total 4)
    #     vgg_layer(in_channels=128, conv_block=[256, 256, 256]), # 3 conv layers (total 7)
    #     vgg_layer(in_channels=256, conv_block=[512, 512, 512]), # 3 conv layers (total 10)
    #     vgg_layer(in_channels=512, conv_block=[512, 512, 512]), # 3 conv layers (total 13)
    # )

    layers = []

    return layers