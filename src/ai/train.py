from torch import optim
from .model import Model, nn, cuda
from typing import Type

class Trainer:
    def __init__(self, 
                 n_epochs : int,
                 loss_fn: Type[nn.Module] = nn.CrossEntropyLoss,
                 lr: float = 0.001,
                 device: str = "cuda" if cuda.is_available() else "cpu"):
        
        self.max_epochs = n_epochs
        self.loss_fn = loss_fn()
        self.lr = lr
        self.device = device
        
    ## Loss function
    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y.to(self.device))
    
    ## Optimization algorithm
    def configure_optimizers(self, model_params):
        return optim.Adam(params=model_params, lr=self.lr)

    ## Fitting step
    def fit(self, model : Model, train_data):
        self.model = model.to(self.device)
        self.train_data = train_data

        # Configure the optimizer
        self.optimizer = self.configure_optimizers(self.model.parameters())

        for epoch in range(self.max_epochs):
            train_loss = self.fit_epoch()
            print(f'Epoch {epoch+1}/{self.max_epochs}, Train Loss: {train_loss:.4f}')

        print("Training process has finished")

    def fit_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Iterate and update network weights, compute loss
        for inputs, labels in self.train_data:
            # Get inputs and the corresponding labels
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Clear gradient buffers
            self.optimizer.zero_grad()

            # Get output from model, given the inputs
            outputs = self.model(inputs)

            # Get loss for predicted outputs
            loss = self.loss(outputs, labels)

            # Get gradients W.R.T the parameters of the model
            loss.backward()

            # Update the parameters (perform optimization)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches != 0 else None