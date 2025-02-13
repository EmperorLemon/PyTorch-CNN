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
    "batch_size": 32,
    "val_test_split": [0.2, 0.1], # 20% validation, 10% testing 
    "model_type": ModelType.VGG16,
    "hidden_layers": [256, 128], # Only used in MLP
    "dropout_rate": 0.5,
    "learning_rate": 5e-4,
    "weight_decay": 1e-9,
    "num_epochs": 100,
    "optim_type": OptimizerType.SGD,
}

CURRENT_MODEL_MODE = ModelMode.INFERENCE