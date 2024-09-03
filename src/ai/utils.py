from torch import cuda
from torch import save, load

from .model import Model

from globals import MODEL_PATH

def check_cuda():
    if cuda.is_available():
        print("CUDA is available")
        print(f"CUDA device count: {cuda.device_count()}")
        print(f"Current CUDA device: {cuda.current_device()}")
        print(f"CUDA device name: {cuda.get_device_name(0)}")
    else:
        print("CUDA is not available, using CPU")

def save_model(model: Model):
    save(model.state_dict(), MODEL_PATH)

def load_model(model: Model):
    state_dict = load(MODEL_PATH, map_location="cpu", weights_only=True)
    
    model.load_state_dict(state_dict=state_dict)

    model = model.to(model.device)
    
    model.eval()
    
    # print(f"Model loaded from mnist_model.pth and moved to {model.device}")
    
    return model