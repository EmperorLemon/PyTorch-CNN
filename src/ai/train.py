from torch import optim, no_grad, max
from torch.utils.tensorboard.writer import SummaryWriter
from torch.amp import GradScaler, autocast
from .model import Model, nn, cuda
from .utils import save_state, load_state
from typing import Type, Optional
from tqdm import tqdm

import time

class EarlyStopper():
    def __init__(self, 
                 patience: int = 5,
                 min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer():
    def __init__(self, 
                 n_epochs : int,
                 loss_fn: Type[nn.Module] = nn.CrossEntropyLoss,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 patience: int = 5,
                 min_delta: float = 1e-4,
                 gradient_accumulation_steps: int = 1,
                 mixed_precision: bool = True,
                 device: str = "cuda" if cuda.is_available() else "cpu",
                 writer: Optional[Type[SummaryWriter]] = None):
        
        self.max_epochs = n_epochs
        self.loss_fn = loss_fn()
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.device = device
        self.writer = writer if writer is not None else SummaryWriter()
        
        self.early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)
        self.scaler = GradScaler() if self.mixed_precision else None
        
    ## Optimization algorithm
    def configure_optimizers(self, model_params):
        optimizer = optim.SGD(params=model_params, lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                         mode='min', 
                                                         factor=0.5, 
                                                         patience=self.patience,
                                                         threshold=0.01,
                                                         threshold_mode='rel',
                                                         cooldown=0,
                                                         min_lr=1e-6)

        return optimizer, scheduler

    ## Fitting step
    def fit(self, model : Model, train_loader, val_loader):
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Configure the optimizer
        self.optimizer, self.scheduler = self.configure_optimizers(self.model.parameters())

        best_val_loss = float('inf')
        
        start_time = time.time()

        for epoch in range(self.max_epochs):
            print(f"\nGPU Memory: Allocated: {cuda.memory_allocated() / 1e9:.2f} GB, "
              f"Cached: {cuda.memory_reserved() / 1e9:.2f} GB")

            train_loss = self.fit_epoch()
            val_loss, val_accuracy = self.validate()

            if self.writer is not None:
                # Log to TensorBoard
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                self.writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
            
            print(f"Epoch {epoch+1}/{self.max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            current_lr = self.scheduler.get_last_lr()[epoch]
            print(f"Current learning rate: {current_lr}")

            # Save best model
            if val_loss < best_val_loss and (epoch + 1) % self.patience == 0:
                # save_model(model=self.model)
                # print(f"New best model saved at epoch {epoch+1}")
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_accuracy)

            self.early_stopper(val_loss=val_loss)
            if self.early_stopper.early_stop:
                print(f"Early stopper triggered at epoch {epoch + 1}")
                break

        end_time = time.time()
        print(f"\nTraining process has finished. Total time: {end_time - start_time:.2f} seconds")

    def fit_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")
        # Iterate and update network weights, compute loss
        for i, (inputs, labels) in enumerate(progress_bar):
            # Get inputs and the corresponding labels, move tensors to configured device
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            with autocast(device_type=self.model.device.type, enabled=self.mixed_precision):
                # Get output from model, given the inputs
                outputs = self.model(inputs)

                # Get loss for predicted outputs
                loss = self.loss_fn(outputs, labels)
                loss = loss / self.gradient_accumulation_steps

            # Backward and optimize

            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                # Get gradients W.R.T the parameters of the model
                loss.backward()

            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Update the parameters ((weights)) (perform optimization)
                    self.optimizer.step()

                # Clear gradient buffers
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with autocast(device_type=self.model.device.type, enabled=self.mixed_precision):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)

                total_loss += loss.item()

                _, predicted = max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        return val_loss, accuracy
    
    def save_checkpoint(self, epoch, val_loss, val_accuracy):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }
        
        save_state(checkpoint, f"checkpoint_epoch_{epoch}.pth")
        
        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = load_state(checkpoint_path)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        epoch = checkpoint["epoch"]
        val_loss = checkpoint["val_loss"]
        val_accuracy = checkpoint["val_accuracy"]
        
        print(f"Loaded checkpoint from epoch {epoch} with validation loss {val_loss:.4f} and accuracy {val_accuracy:.2f}%")
        
        return epoch