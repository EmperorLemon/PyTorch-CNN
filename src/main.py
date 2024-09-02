from ai.model import Model, Layer, nn
from ai.train import Trainer
from ai.data import Dataset, DataLoader, MNIST
from ai.utils import check_cuda
from ai.visualizer import visualize_results
from ai.test import evaluate_model

from torchvision import transforms
from torch import save, load
from torch.utils.tensorboard.writer import SummaryWriter

import os

BASE_DIR = os.path.abspath(os.path.dirname(os.getcwd()))
LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "logs"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "models"))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "out"))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "data"))
TRAIN_PATH = os.path.abspath(os.path.join(DATA_PATH, "train"))
TEST_PATH = os.path.abspath(os.path.join(DATA_PATH, "test"))

def load_model(model: Model):
    state_dict = load(os.path.abspath(os.path.join(MODEL_DIR, "mnist_model.pth")), map_location="cpu", weights_only=True)
    
    model.load_state_dict(state_dict=state_dict)

    model = model.to(model.device)
    
    model.eval()
    
    # print(f"Model loaded from mnist_model.pth and moved to {model.device}")
    
    return model

def train_model(transform, model: Model, trainer: Trainer):
    # Loading the data
    train_dataset = MNIST(DATA_PATH, train=True, download=True, transform=transform)
    val_dataset = MNIST(DATA_PATH, train=False, download=True, transform=transform)
    
    # print(DATA_PATH)
    # print(train_dataset.classes)
    # print(val_dataset.classes)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    trainer.fit(model=model, train_data=train_loader, val_data=val_loader)

    # Save the model
    save(model.state_dict(), os.path.abspath(os.path.join(MODEL_DIR, "mnist_model.pth")))

def test_model(transform, model: Model, trainer: Trainer):
    # for name, param in model.named_parameters():
    #     print(f"Parameter {name} device: {param.device}")

    # Loading the data
    test_dataset = MNIST(DATA_PATH, train=False, download=True, transform=transform)

    # Create the data loaders
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    evaluate_model(test_data=test_loader, model=model)

    # Visualize some results
    visualize_results(test_dataset, model, OUTPUT_DIR, num_samples=10)

def main() -> int:
    check_cuda()

    writer = SummaryWriter(LOG_DIR)

    input_size = 1 * 28 * 28
    output_size = 10

    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])

    network_layers : list[Layer] = [
        Layer("flatten", nn.Flatten()),
        Layer("fc1", nn.Linear(input_size, 64)),
        Layer("relu1", nn.ReLU()),
        Layer("fc2", nn.Linear(64, 32)),
        Layer("relu2", nn.ReLU()),
        Layer("fc3", nn.Linear(32, output_size))
    ]

    # Create the model
    model = Model(network_layers)
    print(f"Model created with device: {model.device}")

    # Create the trainer
    trainer = Trainer(n_epochs=10, lr=0.001, device=model.device, writer=writer)

    # Ask user if they want to train a new model or load an existing one
    user_choice = input("Do you want to train a new model or load an existing one? (train/load): ")

    if user_choice.lower() == "train":
        # Train the model
        train_model(transform=transform, model=model, trainer=trainer)
    elif user_choice.lower() == "load":
        # Load the model
        if os.path.exists("mnist_model.pth"):
            model = load_model(model)
        else:
            print("No saved model found. Training a new model.")
            train_model(transform=transform, model=model, trainer=trainer)
    else:
        print("Invalid choice. Training a new model.")
        train_model(transform=transform, model=model, trainer=trainer)

    # Test the model
    test_model(transform=transform, model=model, trainer=trainer)

    return 0

if __name__ == "__main__":
    main()