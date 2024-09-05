from torch import cuda
from torch import save, load

from globals import MODEL_DIR

import os

def check_cuda():
    if cuda.is_available():
        print("CUDA is available")
        print(f"CUDA device count: {cuda.device_count()}")
        print(f"Current CUDA device: {cuda.current_device()}")
        print(f"CUDA device name: {cuda.get_device_name(0)}\n")
    else:
        print("CUDA is not available, using CPU\n")

def save_state(state, model_file: str):
    model_path = os.path.abspath(os.path.join(MODEL_DIR, model_file))

    save(state, model_path)

def load_state(model_file: str):
    model_path = os.path.abspath(os.path.join(MODEL_DIR, model_file))

    return load(model_path, map_location="cpu", weights_only=True)

