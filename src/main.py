from ai.model import Model, Layer, nn
from ai.train import Trainer
from ai.data import Dataset, DataLoader, MNIST

from torchvision import transforms

from torch import save, load

import os

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "data"))
TRAIN_PATH = os.path.abspath(os.path.join(DATA_PATH, "train"))
TEST_PATH = os.path.abspath(os.path.join(DATA_PATH, "test"))

def train_model(transform, model: Model):
    # Loading the data
    train_dataset = MNIST(DATA_PATH, train=True, download=True, transform=transform)
    
    #print(DATA_PATH)
    print(train_dataset.classes)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1)

    # Create the trainer
    trainer = Trainer(n_epochs=10, lr=0.001)
    trainer.fit(model=model, train_data=train_loader)

    # Save the model
    save(model.state_dict(), "mnist_model.pth")

def load_model(model: Model):
    model.load_state_dict(load("mnist_model.pth", map_location=model.device))
    
    # Set the model to evaluation mode
    model.eval()
    
    print(f"Model loaded from mnist_model.pth and moved to {model.device}")
    
    return model

def test_model(transform, model: Model):
    # Loading the data
    test_dataset = MNIST(DATA_PATH, train=False, download=True, transform=transform)

    # Create the data loaders
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)

    images, labels = next(iter(test_loader))

    output = model(images)

def main() -> int:
    input_size = 28 * 28 * 1
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
    print(f"Using device: {model.device}")

    # Ask user if they want to train a new model or load an existing one
    user_choice = input("Do you want to train a new model or load an existing one? (train/load): ")

    if user_choice.lower() == "train":
        # Train the model
        train_model(transform=transform, model=model)
    elif user_choice.lower() == "load":
        # Load the model
        if os.path.exists("mnist_model.pth"):
            model = load_model(model)
        else:
            print("No saved model found. Training a new model.")
            train_model(transform=transform, model=model)
    else:
        print("Invalid choice. Training a new model.")
        train_model(transform=transform, model=model)

    # Test the model
    test_model(transform=transform, model=model)

    return 0

if __name__ == "__main__":
    main()