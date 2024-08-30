from torch import nn
from model import Model

class Trainer:
    def __init__(self, n_epochs : int):
        self.max_epochs = n_epochs

    ## Fitting step
    def fit(self, model : Model, data):
        self.data = data

        # Configure the optimizer
        self.optimizer = model.configure_optimizers()
        self.model = model

        for _ in range(self.max_epochs):
            self.fit_epoch()

        print("Training process has finished")

    def fit_epoch(self):
        current_loss = 0.0

        # Iterate and update network weights, compute loss
        for i, data in enumerate(self.data):
            # Get input and the corresponding label
            inputs, label = data

            # Clear gradient buffers
            self.optimizer.zero_grad()

            # Get output from model, given the inputs
            outputs = self.model(inputs)

            # Get loss for predicted output
            loss = self.model.loss(outputs, label)

            # Get gradients W.R.T the parameters of the model
            loss.backward()

            # Update the parameters (perform optimization)
            self.optimizer.step()

            # Display stats
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))
                current_loss = 0.0