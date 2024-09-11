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

def get_best_state() -> str:
    # Regex to match pattern
    pattern = r'ep=(\d+)_vl=([\d.]+)_va=([\d.]+)\.pth'

    vl_min = float('inf')
    best_state: str = None

    for filename in list_files(CHECKPOINT_DIR):
        # Remove directory path
        basename = os.path.basename(filename)

        match = re.match(pattern, basename)

        if match:
            epoch = int(match.group(1))
            validation_loss = float(match.group(2))
            validation_accuracy = float(match.group(3))

            if (validation_loss < vl_min):
                vl_min = validation_loss
                best_state = filename

    return best_state