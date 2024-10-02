from torch import optim, no_grad, max

from tensorboardX import SummaryWriter
from torch.amp import GradScaler, autocast

from .model import nn, cuda
from .utils import get_best_state, load_checkpoint, save_checkpoint
from .data import DataLoader

from config import OptimizerType

from typing import Type, Optional
from tqdm import tqdm

import time

class EarlyStopper:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model_state):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model_state
        else:
            self.counter += 1
            self.countdown = self.patience - self.counter
            print(f"Epochs remaining before stopping early: {self.countdown}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.best_state

class Trainer():
    def __init__(self, 
                 n_epochs: int,
                 lr: float,
                 weight_decay: float,
                 criterion: Type[nn.Module] = nn.CrossEntropyLoss,
                 patience: int = 3,
                 mixed_precision: bool = True,
                 device: str = "cuda" if cuda.is_available() else "cpu",
                 writer: Optional[Type[SummaryWriter]] = None):
        
        self.max_epochs = n_epochs
        self.criterion = criterion()
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.mixed_precision = mixed_precision
        self.device = device
        self.writer = writer
        self.save_frequency = 3
        
        self.early_stopper = EarlyStopper(patience=self.patience * 2, min_delta=1e-4)
        self.scaler = GradScaler() if self.mixed_precision else None
        
    ## Optimization algorithm
    def configure_optimizers(self, model_params, optim_type: OptimizerType):
        
        optimizer = None
        
        if optim_type == OptimizerType.SGD:
            optimizer = optim.SGD(params=model_params, lr=self.lr, momentum=0.9, 
                                  weight_decay=self.weight_decay, nesterov=True)
        elif optim_type == OptimizerType.ADAM:
            optimizer = optim.Adam(params=model_params, lr=self.lr, 
                                   weight_decay=self.weight_decay)
        
        # reduce the learning rate if the model's performance is shit
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                         factor=0.5,
                                                         patience=self.patience,  
                                                         min_lr=1e-6)

        return optimizer, scheduler

    ## Fitting step
    def fit(self, model, train_loader: DataLoader, val_loader: DataLoader, optim_type: OptimizerType, pretrained: bool = True):
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Configure the optimizer
        self.optimizer, self.scheduler = self.configure_optimizers(self.model.parameters(), optim_type=optim_type)
        
        if pretrained:
            # Get the filepath of the best checkpoint state
            best_state = get_best_state()

            if best_state is not None:
                self.checkpoint = load_checkpoint(self.model, self.optimizer, best_state)

        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(self.max_epochs):
            print(f"\nGPU Memory: Allocated: {cuda.memory_allocated() / 1e9:.2f} GB, "
              f"Cached: {cuda.memory_reserved() / 1e9:.2f} GB")

            train_loss = self.fit_epoch()
            val_loss, val_accuracy = self.validate()
            
            print(f"Epoch {epoch}/{self.max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Current learning rate: {current_lr:.2e}")

            if self.writer is not None:
                # Log to TensorBoardX
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                self.writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)

            # Save best model if the validation loss is less than the current lowest val loss.
            # Also, the best model is saved every 3 or 5 epochs (depends what the save frequency is)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                if epoch % self.save_frequency == 0:
                    save_checkpoint(self.model, self.optimizer, 
                                    epoch, train_loss=train_loss, 
                                    val_loss=val_loss, val_accuracy=val_accuracy)

            best_state = self.early_stopper(val_loss, self.model.state_dict())
            if self.early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                save_checkpoint(self.model, self.optimizer, epoch, train_loss=train_loss,
                                val_loss=val_loss, val_accuracy=val_accuracy)
                self.model.load_state_dict(best_state)
                break

        end_time = time.time()
        print(f"\nTraining process has finished. Total time: {end_time - start_time:.2f} seconds")

    def fit_epoch(self):
        total_loss = 0.0
        num_batches = 0

        self.model.train()
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        # Iterate and update network weights, compute loss
        for inputs, labels in progress_bar:
            # Get inputs and the corresponding labels, move tensors to configured device
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Clear gradient buffers
            self.optimizer.zero_grad()

            # Forward pass
            with autocast(device_type=self.model.device.type, enabled=self.mixed_precision):
                # Get output from model, given the inputs
                outputs = self.model(inputs)

                # Get loss for predicted outputs
                loss = self.criterion(outputs, labels)
            
            # Backward and optimize (backpropigation)
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                # self.scaler.unscale_(self.optimizer)
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Get gradients W.R.T the parameters of the model
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            del inputs, labels, outputs
        
        return total_loss / num_batches
    
    def validate(self):
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)

        self.model.eval()
        with no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with autocast(device_type=self.model.device.type, enabled=self.mixed_precision):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                current_accuracy = 100 * correct / total
                progress_bar.set_postfix({
                    "loss": f"{total_loss / (progress_bar.n + 1):.4f}",
                    "accuracy": f"{current_accuracy:.2f}%"
                })

                del inputs, labels, outputs

        val_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        return val_loss, accuracy