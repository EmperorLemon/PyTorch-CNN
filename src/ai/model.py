from torch import nn, cuda
from torch import device as TorchDevice
from collections import OrderedDict
from typing import List

## A network layer
class Layer():
    def __init__(self, name: str, layer: nn.Module):
        self.name = name
        self.layer = layer

## A network model
class Model(nn.Module):
    def __init__(self, 
                 layers: List[Layer],
                 device: str = "cuda" if cuda.is_available() else "cpu"):
        super().__init__()

        self.device = TorchDevice(device)
        self.to(self.device)

        layer_names = [layer.name for layer in layers]
        if len(layer_names) != len(set(layer_names)):
            raise ValueError("Layer names must be unique")

        # The network is a Sequential model
        self.net = nn.Sequential(OrderedDict([(layer.name, layer.layer) for layer in layers]))

    ## Forward step
    def forward(self, X):
        # Compute output given an input X
        return self.net(X.to(self.device))