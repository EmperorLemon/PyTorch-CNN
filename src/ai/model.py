from torch import nn, cuda
from torch import device as TorchDevice
from collections import OrderedDict
from typing import List, Any, NamedTuple

## A network model
class Model(nn.Module):
    def __init__(self, 
                 layers: List[nn.Module],
                 device: str = "cuda" if cuda.is_available() else "cpu"):
        super().__init__()

        self.device = TorchDevice(device)
        self.to(self.device)

        self.net = nn.Sequential(*layers)

    ## Forward step
    def forward(self, X):
        # Compute output given an input X
        return self.net(X.to(self.device))