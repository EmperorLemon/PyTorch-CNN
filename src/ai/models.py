from .model import nn
from typing import List

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

    # Output Layer
    layers.append(nn.Linear(in_features=fc_layers[-1], out_features=num_classes))

    return nn.Sequential(*layers)