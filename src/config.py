from enum import Enum

class ModelType(Enum):
    MLP = 0 # Multilayer Perceptron model
    VGG16 = 1 # VGG16 CNN model from scratch
    VGG16_TORCH = 2 # Pytorch VG166 CNN model (pretrained)
    
class ModelMode(Enum):
    INFERENCE = 0
    TRAINING = 1
    
class OptimizerType(Enum):
    SGD = 0 # Stochastic Gradient Descent (SGD)
    ADAM = 1
    
hyperparameters: dict = {
    "batch_size": 64,
    "val_test_split": [0.2, 0.1], # 20% validation, 10% testing 
    "model_type": ModelType.MLP,
    "hidden_layers": [1024, 512], # Only used in MLP
    "learning_rate": 3e-3,
    "num_epochs": 100,
    "optim_type": OptimizerType.SGD,
}

CURRENT_MODEL_MODE = ModelMode.TRAINING