from torch import cuda
from torch import save, load

from globals import MODEL_DIR, CHECKPOINT_DIR
from utils import list_files

import os
import re

def check_cuda():
    if cuda.is_available():
        print("CUDA is available")
        print(f"CUDA device count: {cuda.device_count()}")
        print(f"Current CUDA device: {cuda.current_device()}")
        print(f"CUDA device name: {cuda.get_device_name(0)}\n")
    else:
        print("CUDA is not available, using CPU\n")

def save_state(state, model_file: str):
    model_path = os.path.abspath(os.path.join(CHECKPOINT_DIR, model_file))

    save(state, model_path)

def load_state(model_file: str, checkpoint: bool = True):
    if checkpoint:
        checkpoint_path = os.path.abspath(os.path.join(CHECKPOINT_DIR, model_file))
        return load(checkpoint_path, map_location="cpu", weights_only=True)
    else:
        model_path = os.path.abspath(os.path.join(MODEL_DIR, model_file))
        return load(model_path, map_location="cpu", weights_only=True)

def get_best_state() -> str | None:
    # Regex to match pattern
    pattern = r'ep=(\d+)_tl=([\d.]+)_vl=([\d.]+)_va=([\d.]+)\.pth'

    vl_min = float('inf')
    best_state: str = None

    for filename in list_files(CHECKPOINT_DIR):
        # Remove directory path
        basename = os.path.basename(filename)

        match = re.match(pattern, basename)

        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            validation_loss = float(match.group(3))
            validation_accuracy = float(match.group(4))

            if (validation_loss < vl_min):
                vl_min = validation_loss
                best_state = filename

    return best_state

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, val_accuracy):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    }
    
    save_state(checkpoint, f"ep={epoch}_tl={train_loss:.4f}_vl={val_loss:.4f}_va={val_accuracy:.2f}.pth")
    
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = load_state(checkpoint_path)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]
    val_accuracy = checkpoint["val_accuracy"]
    
    print(f"Loaded checkpoint from epoch {epoch} with train loss {train_loss:.4f}, validation loss {val_loss:.4f}, and accuracy {val_accuracy:.2f}%")
    
    return epoch