import numpy as np
from torch import nn, optim

class Model(nn.Module):
    def __init__(self, in_features : int, out_features : int, lr : float):
        super().__init__()

        # Learning rate
        self.lr = lr

        # The network is a Linear model
        self.net = nn.Linear(in_features=in_features, out_features=out_features)

    ## Forward step
    def forward(self, X):
        # Compute output given an input X
        return self.net(X)
    
    ## Loss function
    def loss(self, y_hat, y):
        fn = nn.CrossEntropyLoss()
        return fn(y_hat, y)
    
    ## Optimization algorithm
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), self.lr)